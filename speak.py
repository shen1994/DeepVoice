# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 15:39:33 2018

@author: shen1994
"""

import os
import time
import wave
import struct
import numpy as np
import tensorflow as tf
from keras import backend as K
from pyaudio import PyAudio
from pyaudio import paInt16
import matplotlib.pyplot as plt

from utils import spectrogram_from_wav
from utils import mfcc_to_maxlen
from utils import load_albet_dict
from utils import label_to_albet
from language import load_text_dict
from language import albet2text
from language import load_vector_dict

def save_wave_file(filename, data):
    
    wf = wave.open(filename, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(16000)
    wf.writeframes(b"".join(data))
    wf.close()
    
def load_speech_model():
    
    ASR_graph_def = tf.GraphDef()
    ASR_graph_def.ParseFromString(open("model/pico_DeepDS2_model.pb", "rb").read())
    tf.import_graph_def(ASR_graph_def, name="")
    ASR_sess = tf.Session()
    ASR_sess.graph.get_operations()
    ASR_x = ASR_sess.graph.get_tensor_by_name("DS2_input:0")
    ASR_y = ASR_sess.graph.get_tensor_by_name("time_distributed_2/Reshape_1:0")
    ASR_d = K.ctc_decode(ASR_y, input_length=np.ones(1) * int(ASR_y.get_shape()[1]))
    
    return ASR_x, ASR_d, ASR_sess
    
def load_text_model():

    albet_dict = load_albet_dict()
    text_dict = load_text_dict()
    text_vector_dict, text_words_bag = load_vector_dict()
    
    return albet_dict, text_dict, text_vector_dict, text_words_bag
    
def to_albet(data, rate, ASR_x, ASR_d, ASR_sess, albet_dict):
    
    speech_in = spectrogram_from_wav(data, rate)
    speech_in = mfcc_to_maxlen(speech_in, 1960) 
    speech_out = ASR_sess.run(ASR_d, feed_dict={ASR_x: [speech_in]})
                
    albet = label_to_albet(speech_out[0][0][0], albet_dict)
    
    return albet
    
def to_text(data, rate, ASR_x, ASR_d, ASR_sess, 
            albet_dict, text_dict, text_vector_dict, text_words_bag):
    
    speech_in = spectrogram_from_wav(data, rate)
    speech_in = mfcc_to_maxlen(speech_in, 1960) 
    speech_out = ASR_sess.run(ASR_d, feed_dict={ASR_x: [speech_in]})
                
    albet = label_to_albet(speech_out[0][0][0], albet_dict)
                
    text = albet2text(albet, text_dict, text_vector_dict, text_words_bag)
                
    return text
    
def wav_fourier_show(wav_file):
    
    wf = wave.open(wav_file, 'rb')
    nframes = wf.getnframes()
    framerate = wf.getframerate()
    str_data = wf.readframes(nframes)
    wf.close()
    
    wave_data = np.fromstring(str_data, dtype=np.short)
    
    df = float(framerate) / float(nframes - 1) # 分辨率
    freq = [df * n for n in range(0,nframes)]

    c = np.fft.fft(wave_data) * 2 / nframes
    
    plt.plot(freq, abs(c), 'r')
    plt.show()
    
def wav_lowpass_filter(data, nframes, framerate):
    
    df = float(framerate) / float(nframes - 1) # 分辨率
    freq = [df * n for n in range(0,nframes)]
            
    c = np.fft.fft(data)
    
    # 低通滤波
    c = [0 if freq[i] > 6000 and freq[i] < 10000 else c[i] for i in range(nframes)]
   
    wave_data_back = np.fft.ifft(c)
    wave_data_back = np.int16(np.real(wave_data_back))
    
    return wave_data_back

if __name__ == "__main__":
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # model init
    ASR_x, ASR_d, ASR_sess = load_speech_model()
    albet_dict, text_dict, text_vector_dict, text_words_bag = load_text_model()
    
    print("PICO MODEL IS OK!")

    # speech init
    CHUNK = 4000
    RATE = 16000
    CHANNELS = 1
    LEVEL = 1500
    COUNT_NUM = 200 # threshold
    SAVE_LENGTH = 8
    start_flag = False
    time_flag = False
    once_flag = True
    last_data = None
    audio_sum = 0
    save_count = 0
    save_buffer = []
    
    audio_obj = PyAudio()
    stream = audio_obj.open(format=paInt16, 
                            channels=CHANNELS, 
                            rate=RATE, 
                            input=True, 
                            frames_per_buffer=CHUNK) 
    
    while True:
        
        data = stream.read(CHUNK)
        audio_data = np.fromstring(data, dtype=np.short)
        large_count = np.sum(audio_data > LEVEL)
        audio_temp = np.max(audio_data)
        if audio_temp > 2000 and not start_flag:
            start_flag = True
            begin = time.time()
        if start_flag:
            if time.time() - begin > 5:
                time_flag = True
            if large_count > COUNT_NUM:
                save_count = SAVE_LENGTH
            else:
                save_count -= 1
            if save_count < 0:
                save_count = 0
            if save_count > 0:
                if once_flag and last_data is not None:
                    once_flag = False
                    save_buffer.append(last_data)
                save_buffer.append(data)
            else:
                if time_flag and len(save_buffer) < 5:
                    save_buffer = []
                    start_flag = False
                    time_flag = False
                    once_flag = True
                if len(save_buffer) > 0 or time_flag:

                    try:
                        audio_data = np.fromstring(np.array(save_buffer), dtype=np.short)
                        albet = to_albet(audio_data, RATE, ASR_x, ASR_d, ASR_sess, albet_dict)
                        print('\nPICO predicting albet is: %s' %albet)
                        # audio_data = wav_lowpass_filter(audio_data, len(audio_data), 16000)
                        # albet = to_albet(audio_data, RATE, ASR_x, ASR_d, ASR_sess, albet_dict)
                        # print('\nPICO predicting albet is: %s' %albet)
                        # text = to_text(audio_data, RATE, ASR_x, ASR_d, ASR_sess, 
                        #                albet_dict, text_dict, text_vector_dict, text_words_bag)
                        # print('PICO predicting text is: %s\n' %text)
                    except Exception:
                        save_buffer = []
                        start_flag = False
                        time_flag = False
                        once_flag = True

                    save_buffer = []
                    start_flag = False
                    time_flag = False
                    once_flag = True
                    
        last_data = data
    