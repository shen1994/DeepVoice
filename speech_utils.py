# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 11:12:32 2018

@author: shen1994
"""

import os
import codecs
import numpy as np
from scipy.io import wavfile
from numpy.lib.stride_tricks import as_strided

def spectrogram(samples, fft_length=256, sample_rate=2, hop_length=128):
    """
    Compute the spectrogram for a real signal.
    The parameters follow the naming convention of
    matplotlib.mlab.specgram
    Args:
        samples (1D array): input audio signal
        fft_length (int): number of elements in fft window
        sample_rate (scalar): sample rate
        hop_length (int): hop length (relative offset between neighboring
        fft windows).
    Returns:
        x (2D array): spectrogram [frequency x time]
        freq (1D array): frequency of each row in x
        Note:
    This is a truncating computation e.g. if fft_length=10,
    hop_length=5 and the signal has 23 elements, then the
    last 3 elements will be truncated.
    """
    assert not np.iscomplexobj(samples), "Must not pass in complex numbers"
    
    window = np.hanning(fft_length)[:, None]
    window_norm = np.sum(window ** 2)
    
    # The scaling below follows the convention of
    # matplotlib.mlab.specgram which is the same as
    # matlabs specgram.
    scale = window_norm * sample_rate
    
    trunc = (len(samples) - fft_length) % hop_length
    x = samples[:len(samples) - trunc]
    
    # "stride trick" reshape to include overlap
    nshape = (fft_length, (len(x) - fft_length) // hop_length + 1)
    nstrides = (x.strides[0], x.strides[0] * hop_length)
    x = as_strided(x, shape=nshape, strides=nstrides)
    
    # window stride sanity check
    assert np.all(x[:, 1] == samples[hop_length:(hop_length + fft_length)])
    
    # broadcast window, compute fft over columns and square mod
    x = np.fft.rfft(x * window, axis=0)
    x = np.absolute(x) ** 2
    
    # scale, 2.0 for everything except dc and fft_length/2
    x[1:-1, :] *= (2.0 / scale)
    x[(0, -1), :] /= scale
    
    freqs = float(sample_rate) / fft_length * np.arange(x.shape[0])
    
    return x, freqs
    
def spectrogram_from_wave(audio, sample_rate, step=10, window=20, max_freq=None,
                              eps=1e-14, time_up=12, time_down=2):
    """ Calculate the log of linear spectrogram from FFT energy
        Params:
            filename (str): Path to the audio file
            step (int): Step size in milliseconds between windows
            window (int): FFT window size in milliseconds
            max_freq (int): Only FFT bins corresponding to frequencies between
                [0, max_freq] are returned
            eps (float): Small value to ensure numerical stability (for ln(x))
    """
    if audio.ndim >= 2:
        audio = np.mean(audio, 1)
    if max_freq is None:
        max_freq = sample_rate / 2
    if max_freq > sample_rate / 2:
        raise ValueError("max_freq must not be greater than half of "
                         " sample rate")
    if step > window:
        raise ValueError("step size must not be greater than window size")
    hop_length = int(0.001 * step * sample_rate)
    fft_length = int(0.001 * window * sample_rate)
    pxx, freqs = spectrogram(
        audio, fft_length=fft_length, sample_rate=sample_rate,
        hop_length=hop_length)
    ind = np.where(freqs <= max_freq)[0][-1] + 1

    return np.transpose(np.log(pxx[:ind, :] + eps))
    
    
def spectrogram_from_file(filename, step=10, window=20, max_freq=None,
                              eps=1e-14, time_up=12, time_down=2):
    """ Calculate the log of linear spectrogram from FFT energy
        Params:
            filename (str): Path to the audio file
            step (int): Step size in milliseconds between windows
            window (int): FFT window size in milliseconds
            max_freq (int): Only FFT bins corresponding to frequencies between
                [0, max_freq] are returned
            eps (float): Small value to ensure numerical stability (for ln(x))
    """
    sample_rate, audio = wavfile.read(filename)    
    audio = audio / np.sqrt(np.sum(np.square(audio)))
    if audio.ndim >= 2:
        audio = np.mean(audio, 1)
    if max_freq is None:
        max_freq = sample_rate / 2
    if max_freq > sample_rate / 2:
        raise ValueError("max_freq must not be greater than half of "
                         " sample rate")
    if step > window:
        raise ValueError("step size must not be greater than window size")
    hop_length = int(0.001 * step * sample_rate)
    fft_length = int(0.001 * window * sample_rate)
    pxx, freqs = spectrogram(
        audio, fft_length=fft_length, sample_rate=sample_rate,
        hop_length=hop_length)
    ind = np.where(freqs <= max_freq)[0][-1] + 1

    # audio record time limit
    is_saved = False
    sample_time = int(len(audio) / sample_rate * 1000)
    if sample_time <= time_up * 1000 and sample_time >= time_down * 1000:
        is_saved = True

    return np.transpose(np.log(pxx[:ind, :] + eps)), is_saved
    
def max_shape_of_speech(path):
    
    train_path = path
    lengths, dims = [], []
    for s in os.listdir(train_path):
        s_path = train_path + '/' + s
        print('%s ---> path load is ok!' %s_path)
        for u in os.listdir(s_path):
            u_path = s_path + '/' + u
            try:
                a, f = spectrogram_from_file(u_path)
                if f:
                    lengths.append(a.shape[0])
                    dims.append(a.shape[1])
                else:
                    os.remove(u_path)
            except Exception:
                os.remove(u_path)

    max_length = np.max(lengths)
    max_dim = np.max(dims)

    return (max_length, max_dim)    
    
if __name__ == "__main__":
    
    train_max_l, train_max_d = max_shape_of_speech('speechs/train')
    valid_max_l, valid_max_d = max_shape_of_speech('speechs/test')   
    
    with codecs.open('speechs/speech_info.txt', 'w') as fout:
        fout.write('[PICO ASR AI]\n')
        fout.write('train: max_length %s, max_dim %s\n' %(str(train_max_l), str(train_max_d)))
        fout.write('valid: max_length %s, max_dim %s\n' %(str(valid_max_l), str(valid_max_d)))
    