import os, sys, io
import numpy as np
import cv2
from scipy.io import wavfile
import matplotlib.pyplot as plt
#from splitWavProcess import *


class DataProcess():

    def __init__(self, wav_list, sylla_list, width=64, height=128):
        self.wav_list = wav_list
        self.sylla_list = sylla_list
        self.width=width
        self.height=height

    def get_syllable_value(self, idx):
    
        wav_idx = self.sylla_list[idx][0]
        start = self.sylla_list[idx][1]
        lenth = self.sylla_list[idx][2]
        end = start+lenth
    
        wav = self.wav_list[wav_idx]
        x, y = wavfile.read(wav)
        val = y[:,0][start:end]
    
        return x, val

    def get_img_from_fig(self, fig, dpi=100):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi)
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        img = cv2.imdecode(img_arr, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
     
        return img
    
    def img2arr(self, idx, opt='gray'):
        
        x, val = self.get_syllable_value(idx)
        
        xlen = self.width/100
        ylen = self.height/100
        
        fig = plt.figure(figsize=(xlen, ylen))
        
        if opt =='gray':
            plt.specgram(val, Fs = x, cmap='gray')
        else: 
            plt.specgram(val, Fs = x)
        plt.axis('off'), plt.xticks([]), plt.yticks([])
        plt.tight_layout()
        plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0, wspace = 0)
        plt.close('all')
        
        arr = self.get_img_from_fig(fig)    
        if opt=='gray':
            arr=arr[:,:,0]

        return arr

    def get_dataset(self):
        dataset = []
        print('* Start process...  total syllables: %i'%len(self.sylla_list))
        for idx in range(len(self.sylla_list)):
            arr = self.img2arr(idx)
            dataset.append(arr)
            if idx%100==0:
                print(idx)
        dataset = np.array(dataset)
        return dataset
