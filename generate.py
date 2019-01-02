# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 16:38:53 2018

@author: shen1994
"""

import os
import cv2
import codecs
import pickle
import numpy as np
from scipy.io import wavfile
from speech_utils import spectrogram_from_wave

class Generator(object):
    
    def __init__(self, 
                 path,
                 feature_maxlen=1960,
                 feature_ndim=161,
                 label_true_maxlen=88,
                 label_pred_maxlen=408): # label_true_maxlen < label_pred_maxlen
        
        with codecs.open('speechs/text_to_label.pkl', 'rb') as fin:
            text_to_label = pickle.load(fin)
            exit_text = text_to_label.keys()
        with codecs.open('speechs/dict_word2label.pkl', 'rb') as fin:
            dict_word2label = pickle.load(fin)
            space_encode = dict_word2label[' ']
        
        self.feature_maxlen = feature_maxlen # 1s 1000ms/10ms=100
        self.feature_ndim = feature_ndim # n_input * (n_context * 2 + 1)
        self.label_true_maxlen = label_true_maxlen
        self.label_pred_maxlen = label_pred_maxlen
        self.space_encode = space_encode
        
        self.sounds = []
        self.labels = []
        self.sample_len = 0    
        for s in os.listdir(path):
            s_path = path + os.sep + s
            for u in os.listdir(s_path):
                u_path = s_path + os.sep + u
                if u[:-4] in exit_text:
                    self.sounds.append(u_path)
                    self.labels.append(text_to_label[u[:-4]])
                    self.sample_len += 1
                    
    def random_argument(self, data):
        
        # sound tune
        if np.random.randint(0, 2) == 1:
            ly = len(data) 
            y_tune = cv2.resize(data, (1, int(len(data) * 1.2))).squeeze() 
            lc = len(y_tune) - ly 
            y_tune = y_tune[int(lc / 2):int(lc / 2) + ly]
        else:
            y_tune = data
         
        # sound noise  
        if np.random.randint(0, 2) == 1:             
            wn = np.random.normal(0, 1, len(y_tune)) 
            y_tune = np.where(y_tune != 0.0, y_tune.astype('float64') + 0.003 * wn, 0.0).astype(np.float32)
        
        return y_tune
        
    def mfcc_to_maxlen(self, a_mfcc):
        
        mfcc_len = len(a_mfcc)
        if mfcc_len >= self.feature_maxlen:
            return a_mfcc[:self.feature_maxlen, :]
        else:
            padding = np.zeros((self.feature_maxlen - mfcc_len, a_mfcc.shape[1]), dtype=np.float32)
            a_mfcc = np.concatenate((a_mfcc, padding))
            return a_mfcc
        
    def label_to_maxlen(self, a_label):
        
        label_len = len(a_label)
        if label_len >= self.label_true_maxlen:
            return a_label[:self.label_true_maxlen]
        else:
            padding = np.full((self.label_true_maxlen - label_len,), self.space_encode, dtype=np.int32)
            a_label = np.concatenate((a_label, padding))
            return a_label
            
    def generate(self, batch_size=32):
        
        while(True):
            indexes = np.arange(0, self.sample_len)
            np.random.shuffle(indexes)
            wav_paths = [self.sounds[index] for index in indexes]
            wav_labels = [self.labels[index] for index in indexes]

            print('\nGenerate %d once...' %self.sample_len)
            
            counter = 0
            batch_counter = 0
            mfccs, labels, labels_true_lens, labels_pred_lens = [], [], [], []
            for i in range(self.sample_len):
                sample_rate, audio = wavfile.read(wav_paths[i])    
                audio = audio / np.sqrt(np.sum(np.square(audio))) 
                audio = self.random_argument(audio)
                a_mfcc = spectrogram_from_wave(audio, sample_rate)
                a_mfcc = self.mfcc_to_maxlen(a_mfcc)
                mfccs.append(a_mfcc)
                a_label = self.label_to_maxlen(np.array(wav_labels[i]))
                labels.append(a_label)
                labels_true_lens.append(self.label_true_maxlen)
                labels_pred_lens.append(self.label_pred_maxlen)
                counter += 1
                
                if (batch_counter + 1) * batch_size > self.sample_len:
                    counter = 0
                    batch_counter = 0
                    mfccs, labels, labels_true_lens, labels_pred_lens = [], [], [], []
                    break
                
                if counter >= batch_size:
                    yield [np.array(mfccs), np.array(labels), np.array(labels_pred_lens), np.array(labels_true_lens)], \
                          [np.array(labels)]
                    counter = 0
                    batch_counter +=  1
                    mfccs, labels, labels_true_lens, labels_pred_lens = [], [], [], []
    
    
    
    
    
    
    
    
    
    