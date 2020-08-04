
import glob
from splitWavProcess import *
##
class DataClean():
    
    def __init__(self, data_path):
        self.wav_list = sorted(glob.glob(data_path+'/*.wav'))
        self.wav_name = np.array([wav.split('/')[-1].split('.')[0] for wav in self.wav_list])

    def check_wav_img(self, idx):
        wav = self.wav_list[idx] 
        name = wav.split('/')[-1].split('.')[0]
        
        draw_split_wav(wav, 'Idx %i: %s'%(idx,name))
        draw_specgram(wav)
        draw_wav(wav)

    def get_syllable_list(self, wav_list):
        syllable_list = []
        n = 0
        for wav in wav_list:
            name = wav.split('/')[-1].split('.')[0]
            idx = np.where(self.wav_name == name)[0][0]
    
            x, y = read_wav(wav)
            noise = get_noise(y)
            syllables = get_syllable(noise)
            
            for syllable in syllables:
                syllable_list.append([idx, syllable[0], syllable[1]])
            if n % 10 == 0: 
                print('* Finish Idx %i wav file : '%n, name)
            n=n+1

        syllable_list = np.array(syllable_list)
        return syllable_list


class Abnormal():
    def __init__(self, syllable_list, wav_list):
        self.syllable_list = syllable_list
        self.wav_idx = syllable_list[:,0]
        self.lenth = syllable_list[:,2]
        self.wav_list = wav_list
    
    def get_overlen(self, condition):
        overlen_idx =  self.syllable_list[self.lenth > condition]
        return overlen_idx[:,0]
    
    def save_abnormal_img(self, idx_list, save_path):
        save_path = save_path+'/abnormal/'
        if_not_make(save_path)
    
        for idx in idx_list:
            draw_split_wav(self.wav_list[idx], idx, '%s/%s'%(save_path, idx))
            plt.close('all')

    def remove_abnormal(self, idx_list, lenth_limit=10000):
        syllable_selected = []
        for syllable in self.syllable_list:
            if (syllable[0] in idx_list) or (syllable[2] > lenth_limit): continue
            else: syllable_selected.append(syllable)
        return np.array(syllable_selected)
