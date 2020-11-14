import numpy as np
import models
import matplotlib.pyplot as plt
import pandas as pd
import h5py
import sys
import os
import csv
import shutil
from keras.regularizers import l2
import tensorflow as tf
import keras as K
from tensorflow.keras import metrics
from tensorflow.keras.models import model_from_json,Model
from tensorflow.keras.layers import Dense,concatenate,Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adadelta,Adam
from sklearn.model_selection import train_test_split

### Metrics ###
def top_5_accuracy(y_true,y_pred):
    return metrics.top_k_categorical_accuracy(y_true,y_pred,k=5)

def top_10_accuracy(y_true,y_pred):
    return metrics.top_k_categorical_accuracy(y_true,y_pred,k=10)

def top_50_accuracy(y_true,y_pred):
    return metrics.top_k_categorical_accuracy(y_true,y_pred,k=50)

### CUSTOM LOSS;  mixing cross-entropy with squashed and soft probabilities ###
def KDLoss(y_true,y_pred):
    alpha=0.8 #Weighting
    y_true_hard = tf.one_hot(tf.argmax(y_true, dimension = 1), depth = 256)
    return alpha*categorical_crossentropy(y_true,y_pred)+(1-alpha)*categorical_crossentropy(y_true_hard,y_pred)

### Loading Data ###
def loadData():
    filename = "coords_labels.h5"  #Data frame with vehicles coordinates and LOS/NLOS indicator
    df = pd.read_hdf(filename, key='train')
    X_pos = df[['X', 'Y', 'Z', 'LOS']].to_numpy()
    X_pos=X_pos[:,0:4]
    Y = df[['Labels']].to_numpy() #Channel gains
    Y=np.array(Y[:,0].tolist())
    Y = Y / Y.sum(axis=1)[:, None] #Normalize channel gains so that looks like probabilities
    #Normalizing  column-wise the dataframe, zero mean and unit variance
    for i in range(0,3):
        X_pos[:,i]=X_pos[:,i]-np.mean(X_pos[:,i])
        X_pos[:,i]=X_pos[:,i]/np.sqrt(np.mean(X_pos[:,i]**2))
    #Normalizing lidar will cast all to float making it too big, normalize the input tensor instead
    lidar = np.load('./lidar_008.npz')
    X_lidar = lidar['input']
    return X_pos,X_lidar,Y

def loadDataValidation():
    filename = "coords_labels.h5"  #Data frame with vehicles coordinates and LOS/NLOS indicator
    df = pd.read_hdf(filename, key='val')
    X_pos = df[['X', 'Y', 'Z', 'LOS']].to_numpy()
    X_pos=X_pos[:,0:4]
    Y = df[['Labels']].to_numpy() #Channel gains
    Y=np.array(Y[:,0].tolist())
    Y = Y / Y.sum(axis=1)[:, None] #Normalize channel gains so that looks like probabilities
    m=[757.8467177554851,555.4583686176969,2.1526621404323745]  #Training set means
    c=[4.5697448219753305,68.02014004606737,0.9322994212410762] #Training set variances
    #Normalizing  column-wise the dataframe, zero mean and unit variance
    for i in range(0,3):
        X_pos[:,i]=X_pos[:,i]-m[i]
        X_pos[:,i]=X_pos[:,i]/c[i]
    #Normalizing lidar will cast all to float making it too big, normalize the input tensor instead
    lidar = np.load('./lidar_009.npz')
    X_lidar = lidar['input']
    return X_pos,X_lidar,Y


### Train-Validation split ###
def split_data(X_p,X_l,Y,ratio):
    np.random.seed(12)  #Make split reproducible
    ind = np.arange(X_p.shape[0])
    np.random.shuffle(ind)
    slice=int(ratio*X_p.shape[0])
    return X_p[ind[:slice],:],X_l[ind[:slice],:,:,:],Y[ind[:slice],:],X_p[ind[slice:],:],X_l[ind[slice:],:,:,:],Y[ind[slice:],:]

#train params
num_epochs=50
batch_size=32
X_p_val,X_l_val,Y_val=loadDataValidation()
X_p_train,X_l_train,Y_train=loadData()
X_p_val,X_l_val,Y_val,_,_,_=split_data(X_p_val,X_l_val,Y_val,0.7)
#CNN model for Lidar data
np.random.seed(2)
lidar_model=models.lidarNet(X_l_train[0].shape,2,3)
#FC model for coordinates
pos_model=models.posNet(3)  #Neglecting the LOS info, not sure we can use it
#Concatenate the two outputs and add FC layers
combined_model = concatenate([pos_model.output,lidar_model.output])
reg_val=0.001
layer = Dense(600,activation='relu',kernel_regularizer=l2(reg_val), bias_regularizer=l2(reg_val))(combined_model)
layer = Dense(600,activation='relu',kernel_regularizer=l2(reg_val), bias_regularizer=l2(reg_val))(layer)
layer = Dense(500,activation='relu',kernel_regularizer=l2(reg_val), bias_regularizer=l2(reg_val))(layer)
out = Dense(Y_train.shape[1],activation='softmax')(layer)
model = Model(inputs=[pos_model.input, lidar_model.input], outputs=out)


### Train
opt = Adam()
model.compile(loss=KDLoss,optimizer=opt,metrics=[metrics.categorical_accuracy,top_5_accuracy,top_10_accuracy,top_50_accuracy])
model.summary()


checkpoint = ModelCheckpoint('trained_model.h5', monitor='val_top_10_accuracy', verbose=1,  save_best_only=True, save_weights_only=True, mode='auto', save_frequency=1)
hist = model.fit([X_p_train[:,0:3], X_l_train], Y_train,validation_data=([X_p_val[:,0:3], X_l_val], Y_val), epochs=num_epochs,batch_size=batch_size,callbacks=[checkpoint])
model.save_weights('trained_model.h5')
print(model.metrics_names)

### Plot Validation Acc and Validation Loss
acc = hist.history['categorical_accuracy']
val_acc = hist.history['val_categorical_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']
t5 = hist.history['val_top_5_accuracy']
t10 = hist.history['val_top_10_accuracy']
t50 = hist.history['val_top_50_accuracy']

epochs = range(1, len(acc) + 1)
plt.subplot(121)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.plot(epochs, acc, 'b--', label='accuracy', linewidth=2)
plt.plot(epochs, val_acc, 'g-', label='validation accuracy', linewidth=2)
plt.legend()
plt.subplot(122)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(epochs, loss, 'b--', label='loss', linewidth=2)
plt.plot(epochs, val_loss, 'g--', label='validation loss', linewidth=2)
plt.legend()
plt.savefig('TrainingCurves.png')
plt.show()

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(epochs, t5, 'r--', label='val_top_5_accuracy', linewidth=2)
plt.plot(epochs, t10, 'b--', label='val_top_10_accuracy', linewidth=2)
plt.plot(epochs, t50, 'k--', label='val_top_50_accuracy', linewidth=2)
plt.legend()
plt.savefig('TrainingCurves_1.png')
plt.show()

### Plot Validation Accuracy for LoS and NLoS channels
#NLOS
model.load_weights('trained_model.h5')
NLOSind=np.where(X_p_val[:,3]==0)[0] #Get the NLoS users

preds_gains_NLOS = model.predict([X_p_val[NLOSind,0:3], X_l_val[np.random.randint(np.max(NLOSind), size=NLOSind.shape[0]),:,:,:]]) #Get predictions
pred_NLOS= np.argsort(-preds_gains_NLOS, axis=1) #Descending order
true_NLOS=np.argmax(Y_val[NLOSind,:], axis=1) #Best channel
curve_NLOS=np.zeros(256)
for i in range(0,len(pred_NLOS)):
    curve_NLOS[np.where(pred_NLOS[i,:] == true_NLOS[i])]=curve_NLOS[np.where(pred_NLOS[i,:] == true_NLOS[i])]+1
curve_NLOS=np.cumsum(curve_NLOS)
#LOS
LOSind=np.where(X_p_val[:,3]==1)[0]
preds_gains_LOS = model.predict([X_p_val[LOSind,0:3], X_l_val[np.random.randint(np.max(LOSind), size=LOSind.shape[0]),:,:,:]])
pred_LOS= np.argsort(-preds_gains_LOS, axis=1) #Descending order
true_LOS=np.argmax(Y_val[LOSind,:], axis=1) #Best channel
curve_LOS=np.zeros(256)
for i in range(0,len(pred_LOS)):
    curve_LOS[np.where(pred_LOS[i,:] == true_LOS[i])]=curve_LOS[np.where(pred_LOS[i,:] == true_LOS[i])]+1
curve_LOS=np.cumsum(curve_LOS)
#Plotting
plt.xlabel('K')
plt.ylabel('top-K Accuracy')
plt.ylim((0,1))
plt.xlim((0,256))
plt.grid()
plt.plot(range(0,256),curve_NLOS/curve_NLOS[len(curve_NLOS)-1],'b--', label='NLoS', linewidth=2)
plt.plot(range(0,256),curve_LOS/curve_LOS[len(curve_LOS)-1],'g--', label='LoS', linewidth=2)
plt.plot(range(0,256),(curve_LOS+curve_NLOS)/(curve_NLOS[len(curve_NLOS)-1]+curve_LOS[len(curve_LOS)-1]),'r--', label='avg', linewidth=2)
plt.legend()
plt.savefig('Validation-top-K-Accuracy.png')