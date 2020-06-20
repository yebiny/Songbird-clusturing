from def_dict import *
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import librosa

c_list=['r','salmon','chocolate',
        'darkorange','gold','yellow',
        'yellowgreen','lightgreen','green',
        'turquoise','c','deepskyblue','royalblue',
        'slateblue','darkviolet','m','deeppink']
c_base = 'greenyellow'


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

def get_noise(y, noise_y_cond=0.006, noise_len_cond=500):

    noise_count=0
    noise=[]
    for i in range(len(y)):
        val = y[i]
        if abs(val)<noise_y_cond:
            noise_count+=1
        else:
            if noise_count > noise_len_cond:
                noise.append([i-noise_count, i])
            noise_count=0
    noise.append([len(y)-noise_count, len(y)])

    return noise

def get_syllable(noise, syllable_extralen_cond=300, syllable_len_cond=500):

    syllable=[]
    for i in range(len(noise)-1):
        syllable_start, voise_end = noise[i][1], noise[i+1][0]
        syllable_lenth = voise_end-syllable_start
        if syllable_lenth > syllable_len_cond:
            syllable.append([syllable_start-syllable_extralen_cond, syllable_lenth+2*syllable_extralen_cond])

    return syllable

def draw_split_wav(wav, title, save_name=None):
    x, y = read_wav(wav)
    noise = get_noise(y)
    syllable = get_syllable(noise)
    
    fig = plt.figure(figsize=(15,3))
    plt.plot(x, y, color = c_base, alpha = 0.5)
    
    for i in range(len(syllable)):
        start= syllable[i][0]
        end  = start + syllable[i][1]
        color = c_list[i%len(c_list)]
        plt.plot( x[start:end],y[start:end], color = color, alpha = 0.75)

    plt.title(title)
    if save_name == None:
        plt.show()
    else:
        plt.savefig(save_name)


def check_wav_img(idx, wav_list):
    wav = wav_list[idx] 
    wav_name = wav.split('/')[-1].split('.')[0]
    
    x, y = read_wav(wav)
    noise = get_noise(y)
    syllable = get_syllable(noise)
    
    draw_specgram(wav)
    draw_wav(wav)
    draw_split_wav(wav, 'Idx %i: '%idx+wav_name)

def get_syllable_list(wav_list, wav_name ):
    syllable_list = []
    for wav in wav_list:
        name = wav.split('/')[-1].split('.')[0]
        idx = np.where(wav_name == name)[0][0]

        x, y = read_wav(wav)
        noise = get_noise(y)
        syllables = get_syllable(noise)
        
        for syllable in syllables:
            syllable_list.append([idx, syllable[0], syllable[1]])
        if idx % 10 == 0: 
            print('* Finish Idx %i wav file : '%idx, name)
        
    syllable_list = np.array(syllable_list)
        
    return syllable_list


def hist_syllable_lenth(sylla_len, axis=None):
   if axis !=None:
       plt.hist(sylla_len, color = c_base, range = axis, alpha = 0.6, bins=50)
   else:
       plt.hist(sylla_len, color = c_base, alpha = 0.6, bins=50)
   plt.title('Syllables Lenth Histogram')
   plt.show()

def get_abnormal(syllable_list, condition):
   abnormal_list = syllable_list[syllable_list[:, 2] > condition]
   abnormal_idx = abnormal_list[:, 0]
   abnormal_len = abnormal_list[:, 2]
   return abnormal_idx, abnormal_len

def save_abnormal_img(wav_list, abnormal_idx, save_path):
    save_path = save_path+'/abnormal/'
    if_not_make(save_path)

    for idx in abnormal_idx:
        draw_split_wav(wav_list[idx], idx, '%s/%s'%(save_path, idx))
        plt.close('all')

def remove_abnormal_syllable(syllable_list, drop_wav_list, lenth_limit=10000):
    syllable_selected = []
    for syllable in syllable_list:
        if (syllable[0] in drop_wav_list) or (syllable[2] > lenth_limit): continue
        else: syllable_selected.append(syllable)
    return np.array(syllable_selected)
