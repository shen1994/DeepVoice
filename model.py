# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 13:37:44 2018

@author: shen1994
"""

from keras.layers import Input
from keras.layers import BatchNormalization, Conv1D, Bidirectional, GRU
from keras.layers import Dense, TimeDistributed
from keras.models import Model
from keras.activations import relu

def clipped_relu(x):
    
    return relu(x, max_value=20)

def DS2_model(input_shape=(1960, 161), output_shape=(219, )):
    
    input_x = Input(shape=input_shape, name='DS2_input')
    
    # Batch normalize the input
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True, name='BN_1')(input_x)
    
    # 1D convs
    x = Conv1D(filters=512, kernel_size=11, strides=2, padding='valid', activation='relu', name='Conv2D_1')(x)
    x = Conv1D(filters=512, kernel_size=11, strides=2, padding='valid', activation='relu', name='Conv2D_2')(x)
    x = Conv1D(filters=512, kernel_size=11, strides=2, padding='valid', activation='relu', name='Conv2D_3')(x)
 
    # Batch Normalization
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True, name='BN_2')(x)
    
    # BiRNNs
    x = Bidirectional(GRU(512, return_sequences=True, kernel_initializer='glorot_uniform', activation='relu', name='BiGRU_1'), merge_mode='sum')(x)
    x = Bidirectional(GRU(512, return_sequences=True, kernel_initializer='glorot_uniform', activation='relu', name='BiGRU_2'), merge_mode='sum')(x)
    x = Bidirectional(GRU(512, return_sequences=True, kernel_initializer='glorot_uniform', activation='relu', name='BiGRU_3'), merge_mode='sum')(x)
    x = Bidirectional(GRU(512, return_sequences=True, kernel_initializer='glorot_uniform', activation='relu', name='BiGRU_4'), merge_mode='sum')(x)
    x = Bidirectional(GRU(512, return_sequences=True, kernel_initializer='glorot_uniform', activation='relu', name='BiGRU_5'), merge_mode='sum')(x)
    x = Bidirectional(GRU(512, return_sequences=True, kernel_initializer='glorot_uniform', activation='relu', name='BiGRU_6'), merge_mode='sum')(x)
    x = Bidirectional(GRU(512, return_sequences=True, kernel_initializer='glorot_uniform', activation='relu', name='BiGRU_7'), merge_mode='sum')(x)
    
    # Batch Normalization
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True, name='BN_3')(x)
    
    # Fully Connection
    x = TimeDistributed(Dense(512, activation=clipped_relu, name='FC'))(x)
    output_x = TimeDistributed(Dense(output_shape[0], activation='softmax', name='DS2_output'))(x)
    
    return Model(inputs=input_x, outputs=output_x)
    
    
    
    
    
    
    
    
    
    
    
    
    
    