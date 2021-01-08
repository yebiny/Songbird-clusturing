import os, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import librosa

c_list=['r','salmon','chocolate',
        'darkorange','gold','yellow',
        'yellowgreen','lightgreen','green',
        'turquoise','c','deepskyblue','royalblue',
        'slateblue','darkviolet','m','deeppink']
c_base='greenyellow'


def read_wav(wav_file):
    y, sr = librosa.load(wav_file, sr=32000)
    x = np.linspace(0, len(y)/sr, len(y)) # time axis
    return x, y

def draw_split_wav(wav_list, syllable_list, idx, save_name=None):
    x, y = read_wav(wav_list[idx])
    
    fig = plt.figure(figsize=(15,3))
    plt.plot(x, y, color = c_base, alpha = 0.5)
   
    syllable = syllable_list[syllable_list[:,0]==idx] 
    for i in range(len(syllable)):
        start= syllable[i][1]
        end  = start + syllable[i][2]
        color = c_list[i%len(c_list)]
        plt.plot( x[start:end],y[start:end], color = color, alpha = 0.75)

    plt.title(wav_list[idx])
    if save_name == None:
        plt.show()
    else:
        plt.savefig(save_name)

def draw_spectrogram(wav_file):
    x, y = wavfile.read(wav_file)
    fig = plt.figure(figsize=(15,3))
    plt.specgram(y[:,0][:], Fs = x)
    plt.show()

def draw_wav(wav_file):
    x, y = read_wav(wav_file)
    fig = plt.figure(figsize=(15,3))
    plt.plot(y, color = c_base, label='speech waveform', alpha=0.6)
    plt.show()


def hist_syllable_lenth(sylla_len, axis=None):
   if axis !=None:
       plt.hist(sylla_len, color = c_base, range = axis, alpha = 0.6, bins=50)
   else:
       plt.hist(sylla_len, color = c_base, alpha = 0.6, bins=50)
   plt.title('Syllables Lenth Histogram')
   plt.show()
