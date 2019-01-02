# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 11:06:54 2018

@author: shen1994
"""

import os
import numpy as np
import tensorflow as tf
from keras import backend as K
from model import DS2_model
from utils import spectrogram_from_file
from utils import mfcc_to_maxlen
from utils import load_albet_dict
from utils import label_to_albet
from language import load_text_dict
from language import albet2text
from language import load_vector_dict

def ctc_decode(args):
    y_pred, input_length =args
    seq_len = tf.squeeze(input_length,axis=1)

    return K.ctc_decode(y_pred=y_pred, input_length=seq_len, 
                        greedy=True, beam_width=100, top_paths=1)
    

if __name__ == "__main__":
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    input_shape=(1960, 161)
    output_shape=(219, )
    
    # define text model
    albet_dict = load_albet_dict()
    text_dict = load_text_dict()
    text_vector_dict, text_words_bag = load_vector_dict()
    
    # define speech model
    base_model = DS2_model(input_shape=input_shape, output_shape=output_shape) 
    base_model.load_weights('model/weights.170.hdf5', by_name=True)

    sess = K.get_session()
    speech_input = sess.graph.get_tensor_by_name("DS2_input:0")
    speech_output = sess.graph.get_tensor_by_name("time_distributed_2/Reshape_1:0")
    speech_decode = K.ctc_decode(speech_output, input_length=np.ones(1) * int(speech_output.get_shape()[1]))
    
    while(True): 
        
        wav_path = input("Enter your wav path: ")
        
        if wav_path == '':
            break

        try:
            
            speech_in = spectrogram_from_file(wav_path)
            speech_in = mfcc_to_maxlen(speech_in, input_shape[0]) 
        
            out = sess.run(speech_decode, feed_dict={speech_input: [speech_in]})

            # out = K.get_value(K.ctc_decode(speech_out, \
            #                   input_length=np.ones(speech_out.shape[0]) \
            #                               * speech_out.shape[1])[0][0])

            albet = label_to_albet(out[0][0][0], albet_dict)
        
            print('\nPICO predicting albet is: %s' %albet)
            
            text = albet2text(albet, text_dict, text_vector_dict, text_words_bag)
            
            print('PICO predicting text is: %s\n' %text)
            
        except Exception:
            
            print('no such path or wav is too large!')

    sess.close()
    