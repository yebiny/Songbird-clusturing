import glob, os
import numpy as np
import librosa

class SyllableGenerator():
    
    def __init__(self, data_path):
        self.wav_list = sorted(glob.glob(data_path+'/*.wav'))
        self.wav_name = np.array([wav.split('/')[-1].split('.')[0] for wav in self.wav_list])
    
    def read_wav(self, wav_file):
        y, sr = librosa.load(wav_file, sr=32000)
        x = np.linspace(0, len(y)/sr, len(y)) # time axis
        return x, y
    
    def get_noise(self, y, noise_y_cond=0.006, noise_len_cond=500):
    
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
    
    def get_syllable(self, noise, syllable_extralen_cond=300, syllable_len_cond=500):
    
        syllable=[]
        for i in range(len(noise)-1):
            syllable_start, voise_end = noise[i][1], noise[i+1][0]
            syllable_lenth = voise_end-syllable_start
            if syllable_lenth > syllable_len_cond:
                syllable.append([syllable_start-syllable_extralen_cond, 
                                syllable_lenth+2*syllable_extralen_cond])
    
        return syllable

    def get_syllable_list(self, wav_list):
        syllable_list = []
        n = 0
        for wav in wav_list:
            name = wav.split('/')[-1].split('.')[0]
            idx = np.where(self.wav_name == name)[0][0]
    
            x, y = self.read_wav(wav)
            noise = self.get_noise(y)
            syllables = self.get_syllable(noise)
            
            for syllable in syllables:
                syllable_list.append([idx, syllable[0], syllable[1]])
            if n % 10 == 0: 
                print('* Finish Idx %i wav file : '%n, name)
            n=n+1

        syllable_list = np.array(syllable_list)
        return syllable_list

    def draw_wav_img(self, idx):
        wav = self.wav_list[idx] 
        name = wav.split('/')[-1].split('.')[0]
        
        draw_split_wav(wav, 'Idx %i: %s'%(idx,name))
        draw_specgram(wav)
        draw_wav(wav)
