import sys, glob
import scipy.signal as signal
import math
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import os

def if_not_make(path):
    if not os.path.exists(path):
        os.makedirs(path)
    
class DataFlow():
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.save_dir = data_dir.replace('data','result')

        self.plot_dir = self.save_dir +'/1-plot'
        self.split_dir = self.save_dir +'/2-split'
        self.compo_dir = self.save_dir +'/3-compo'

        if_not_make(self.save_dir)
        if_not_make(self.plot_dir)
        if_not_make(self.split_dir)
        if_not_make(self.compo_dir)
    
        self.data_files = sorted(glob.glob(self.data_dir + '/*'))
        
        self.noise_y_cond, self.noise_len_cond = 0.004, 200 
        self.extra_lenth, self.voice_len_cond = 200, 500
        
    def get_split_plot(self):

        for f in self.data_files:

            save_name = f.split('/')[-1].split('.')[0] 
            x, y = read_wav(f)
            plot_wav(y, self.plot_dir, save_name)
            plt.close()
            
            noise_space = self.get_noise_space(y)
            voice_space = self.get_voice_space(noise_space)
            plot_split(x, y, voice_space, self.split_dir, save_name)
            plt.close()

    def get_lenth(self):

        filterd_files = sorted(glob.glob(self.split_dir + '/*'))
        len_list =[]
        for filterd_file in filterd_files:
            f = filterd_file.split('/')[-1].replace('png','wav')
            f = self.data_dir+'/'+f
            
            x , y = read_wav(f)
            noise_space = self.get_noise_space(y)
            voice_space = self.get_voice_space(noise_space)
            
            for j in range(len(voice_space)):
                len_list.append(voice_space[j][1])
                
        return np.array(len_list)
    
    def get_compo_list(self, comp_max_len, comp_filter_len):
    
        filterd_files = sorted(glob.glob(self.split_dir + '/*'))
        comp_list = []
        for filterd_file in filterd_files:
            f = filterd_file.split('/')[-1].replace('png','wav')
            f = self.data_dir+'/'+f
            
            x , y = read_wav(f)
            noise_space = self.get_noise_space(y)
            voice_space = self.get_voice_space(noise_space)

            for j in range(len(voice_space)):
               
                comp_start = voice_space[j][0]
                comp_len = voice_space[j][1]
                comp_end = comp_start+comp_len
                comp = list(y[comp_start:comp_end])
               
                if len(comp)> comp_filter_len: continue
                if len(comp)> comp_max_len:
                    comp = comp[0::2]
                
                comp.extend([0 for k in range(comp_max_len-len(comp))])
                comp_list.append(comp)
                
        comp_list = np.array(comp_list)
        np.save('%s/compo'%(self.save_dir), comp_list)    
        return comp_list
    
    def get_noise_space(self, y):

        noise_count=0
        noise_space=[]
        for i in range(len(y)):
            val = y[i]
            if abs(val)<self.noise_y_cond:
                noise_count+=1
            else:
                if noise_count > self.noise_len_cond:
                    noise_space.append([i-noise_count, i])
                noise_count=0
        noise_space.append([len(y)-noise_count, len(y)])

        return noise_space

    def get_voice_space(self, noise_space):

        voice_space=[]
        for i in range(len(noise_space)-1):
            voice_start, voise_end = noise_space[i][1], noise_space[i+1][0]
            voice_lenth = voise_end-voice_start
            if voice_lenth > self.voice_len_cond:
                voice_space.append([voice_start-self.extra_lenth, voice_lenth+self.extra_lenth]) 

        return voice_space

def read_wav(f):
    y, sr = librosa.load(f, sr=32000)
    x = np.linspace(0, len(y)/sr, len(y)) # time axis
    return x, y

def plot_wav(y, save_dir, save_name, figsize=(20,5)):
    fig = plt.figure(figsize=figsize)
    plt.plot(y, color = 'steelblue', label='speech waveform')
    fig.savefig('%s/%s.png'%(save_dir, save_name))

def plot_split(x, y, voice_space, save_dir, save_name, figsize=(20,5)):
    fig = plt.figure(figsize=figsize)
    plt.plot(x, y, color = 'steelblue')
    c_list=['r','salmon','chocolate','darkorange','gold','yellow','yellowgreen','lightgreen','green','turquoise','c','deepskyblue','royalblue','slateblue','darkviolet','m','deeppink']
    for i in range(len(voice_space)):
        start= voice_space[i][0]
        end  = start + voice_space[i][1]
        color = c_list[i%len(c_list)]
        plt.plot( x[start:end],y[start:end], color = color)

    fig.savefig('%s/%s'%(save_dir, save_name))
