import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os, sys

def if_not_exit(path):
	if not os.path.exists(path):
		print(path, 'is not exist.')
		exit()
		
def if_not_make(path):
	if not os.path.exists(path):
		os.makedirs(path)

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

def draw_specgram(wav_file):
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
