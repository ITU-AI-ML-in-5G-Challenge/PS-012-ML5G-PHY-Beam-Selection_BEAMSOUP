import numpy as np
import models
import pandas as pd
from keras.regularizers import l2
from tensorflow.keras.models import model_from_json,Model
from tensorflow.keras.layers import Dense,concatenate,Dropout
def reorder(data, num_rows, num_columns):
    '''
    Reorder a vector obtained from a matrix: read row-wise and write column-wise.
    '''
    original_vector  = np.asarray(data, dtype = np.float)
    #read row-wise
    original_matrix = np.reshape(original_vector, (num_rows, num_columns))
    #write column-wise
    new_vector = np.reshape(original_matrix, num_rows*num_columns, 'F')
    return new_vector


def loadData():
    filename = "coords_labels_test.h5"  #Data frame with vehicles coordinates and LOS/NLOS indicator
    df = pd.read_hdf(filename, key='test')
    X_pos = df[['X', 'Y', 'Z']].to_numpy()
    X_pos=X_pos[:,0:3]
    #Normalizing  column-wise the dataframe, zero mean and unit variance
    m=[757.8467177554851,555.4583686176969,2.1526621404323745]
    c=[4.5697448219753305,68.02014004606737,0.9322994212410762]
    for i in range(0,3):
        X_pos[:,i]=X_pos[:,i]-m[i]
        X_pos[:,i]=X_pos[:,i]/c[i]
    #Normalizing lidar will cast all to float making it too big, normalize the input tensor instead
    lidar = np.load('./lidar_010.npz')
    X_lidar = lidar['input']
    return X_pos,X_lidar

X_p,X_l=loadData()
lidar_model=models.lidarNet(X_l[0].shape,2,3)

pos_model=models.posNet(3)
combined_model = concatenate([pos_model.output,lidar_model.output])
reg_val=0
layer = Dense(600,activation='relu',kernel_regularizer=l2(reg_val), bias_regularizer=l2(reg_val))(combined_model)
layer = Dense(600,activation='relu',kernel_regularizer=l2(reg_val), bias_regularizer=l2(reg_val))(layer)
layer = Dense(500,activation='relu',kernel_regularizer=l2(reg_val), bias_regularizer=l2(reg_val))(layer)
out = Dense(256,activation='softmax')(layer)
model = Model(inputs=[pos_model.input, lidar_model.input], outputs=out)

model.load_weights('trained_model.h5')
preds_gains_NLOS = model.predict([X_p, X_l]) #Get predictions
num_rows = 8
num_columns = 32
to_save=[]
for data in preds_gains_NLOS:
    if len(data) != (num_rows * num_columns):
        raise Exception('Number of elements in this row is not the product num_rows * num_columns')
    new_vector = reorder(data, num_rows, num_columns)
    to_save.append(new_vector)
to_save=np.asarray(to_save)

pred= np.argsort(-to_save, axis=1) #Descending order
np.savetxt('beam_test_pred.csv', pred, delimiter=',', fmt='%s')


