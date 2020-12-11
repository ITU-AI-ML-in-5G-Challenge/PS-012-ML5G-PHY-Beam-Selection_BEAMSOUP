from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import BatchNormalization, LeakyReLU, Conv2D, add,\
    Flatten, MaxPooling2D, Dense, Reshape, Input, Dropout, concatenate
from tensorflow.keras.models import Model, Sequential
import tensorflow.keras.utils
import numpy as np
from keras.regularizers import l2
import tensorflow as tf



###Lidar CNN###
def lidarNet(input_shape,min,max):
    train=True
    input_lid = Input(shape=input_shape)
    layer = tf.to_float(input_lid)
    layer=tf.div(tf.add(layer, min),max) #Scaling to [0,1] interval
    layer=tf.keras.layers.GaussianNoise(0.005)(layer)#0.002
    layer = Conv2D(32, kernel_size=(5, 5), activation='relu', padding="SAME", input_shape=input_shape,trainable=train)(layer)
    layer = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid',trainable=train)(layer)
    layer = Conv2D(32, kernel_size=(5,5), activation='relu', padding="SAME", input_shape=input_shape,trainable=train)(layer)
    layer = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid',trainable=train)(layer)
    layer = Conv2D(32, kernel_size=(5,5),activation='relu',padding="SAME",input_shape=input_shape,trainable=train)(layer)
    layer = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid',trainable=train)(layer)
    layer = Conv2D(16, kernel_size=(3, 3), activation='relu', padding="SAME", input_shape=input_shape,trainable=train)(layer)
    layer = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid',trainable=train)(layer)
    layer = Conv2D(16, kernel_size=(3, 3), activation='relu', padding="SAME", input_shape=input_shape,trainable=train)(layer)
    layer = Flatten()(layer)
    layer = Dense(400, activation='relu',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01),trainable=train)(layer)  #0.02
    architecture = Model(inputs=input_lid, outputs=layer)
    return architecture

###Coordinates DNN###
def posNet(input_shape):
    input_coord = Input(shape = input_shape)
    layer = Dense(128,activation='relu')(input_coord)
    layer = tf.keras.layers.GaussianNoise(0.002)(layer)
    architecture = Model(inputs = input_coord, outputs = layer)
    return architecture


