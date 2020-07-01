from def_dict import *
from DataCleaning import *

import glob, os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


class PreProcess():

    def __init__(self, org_path, load_path, save_path):
        self.org_path =org_path
        self.load_path=load_path
        self.save_path=save_path

        self.wav_name = np.load('%s/wav_name.npy'%(load_path))
        self.wav_info = np.load('%s/syllables.npy'%(load_path))

    def get_wav_path(self, idx):
        this_wav=self.wav_name[idx]
        this_name = this_wav.split('_')[1]
        wav_path = '%s/%s.wav'%(self.org_path+this_name, this_wav)
        return wav_path

    def draw_splited_wav(self, idx):

        wav_path = self.get_wav_path(idx)
        draw_wav(wav_path)
        draw_specgram(wav_path)
        
        x, y = wavfile.read(wav_path)
        target = self.wav_info[self.wav_info[:,0]==idx]
        fig = plt.figure(figsize=(20,4))
        for i in range(len(target)):
            start = target[i][1]
            end = start+target[i][2]
            
            plt.subplot(1,len(target),i+1)
            plt.title(i)

            plt.xticks([])
            plt.yticks([])
            plt.specgram(y[:,0][start:end], Fs = x)

        plt.show()

    def get_syllable_value(self, syllable):
    
        idx = syllable[0]
        start = syllable[1]
        lenth = syllable[2]
        end = start+lenth
    
        wav_path = self.get_wav_path(idx)
        x, y = wavfile.read(wav_path)
        val = y[:,0][start:end]
    
        return x, val
    
    def draw_syllable_wav(self, syllable):

        x, val = self.get_syllable_value(syllable)

        fig = plt.figure(figsize=(1,2))
        plt.plot(val, color='k')
        plt.axis('off'), plt.xticks([]), plt.yticks([])
        plt.tight_layout()
        plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0, wspace = 0)
        plt.show()
        plt.close('all')
    
    def draw_syllable_spectogram(self, syllable, save_name=None):
        x, val = self.get_syllable_value(syllable)
        xsize = round(round(len(val)/1000, 2)/6,2)
        ysize = 12/6
        print(xsize, ysize)
        fig = plt.figure(figsize=(xsize,ysize))
        plt.specgram(val, Fs = x)
        plt.axis('off'), plt.xticks([]), plt.yticks([])
        plt.tight_layout()
        plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0, wspace = 0)
    
        if save_name !=None:
            plt.savefig(save_name, bbox_inces='tight',
                    pad_inches=0,
                    dpi=100)
        else:
            plt.show()
    
        plt.close('all')
