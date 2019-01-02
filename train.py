# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 14:30:20 2018

@author: shen1994
"""

import os
import keras
from keras.layers import Input
from keras.layers import Lambda
from keras.models import Model
from keras import backend as K
from model import DS2_model
from generate import Generator
    
def ctc_loss_func(args):
    
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

if __name__ == "__main__":
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    
    model_path = 'model'
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    epochs = 10000
    batch_size = 128
    # input_shape.shape[0] is too small, , this may lead to inf of ctc loss
    input_shape=(1960, 161)
    output_shape=(219, ) # 218 + 1(space) + 1(remian)
    true_len = 120 # true_len(116) < pred_len

    # define input
    label = Input(name='label', shape=[true_len], dtype='int32') 
    i_len = Input(name='i_len', shape=[1], dtype='int32')
    o_len = Input(name='o_len', shape=[1], dtype='int32')
    
    # define model
    base_model = DS2_model(input_shape=input_shape, output_shape=output_shape)
    pred_len = base_model.output.get_shape()[1]

    loss_out = Lambda(ctc_loss_func, output_shape=(1,), name='ctc')([base_model.output, label, i_len, o_len])
    network = Model(inputs=[base_model.input, label, i_len, o_len], outputs=[loss_out])
    network.load_weights('model/weights.170.hdf5', by_name=True)

    # learning rate is too large, this may lead to inf of ctc loss
    opt = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
    network.compile(loss={'ctc': lambda y_true, y_pred: y_pred},  optimizer=opt)
    
    # define training
    callbacks = [keras.callbacks.ModelCheckpoint('model/weights.{epoch:02d}.hdf5',
                                                  verbose=1,
                                                  save_weights_only=True)]
    history = network.fit_generator(generator=Generator(path='speechs/train', 
                                                        feature_maxlen=input_shape[0], 
                                                        feature_ndim=input_shape[1], 
                                                        label_true_maxlen=true_len, 
                                                        label_pred_maxlen=pred_len).generate(batch_size=batch_size), 
                                    epochs=epochs,
                                    steps_per_epoch=1000,
                                    verbose=1,
                                    initial_epoch=0,
                                    callbacks=callbacks,
                                    workers=1)   
    
    
    