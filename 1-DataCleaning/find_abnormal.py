import glob, os
from draw_tools import *

class FindAbnormal():
    def __init__(self, wav_list, syllable_list):
        self.wav_list = wav_list
        self.syllable_list = syllable_list
        self.wav_idx = syllable_list[:,0]
        self.lenth = syllable_list[:,2]
    
    def get_overlen(self, condition):
        overlen_idx =  self.syllable_list[self.lenth > condition]
        return overlen_idx[:,0]
    
    def save_abnormal_img(self, idx_list, save_path):
        if not os.path.exists(save_path): os.makedirs(save_path)
    
        for idx in idx_list:
            draw_split_wav(self.wav_list, self.syllable_list, idx, '%s/%s'%(save_path, idx))
            plt.close('all')

    def remove_abnormal(self, idx_list, lenth_limit=10000):
        syllable_selected = []
        for syllable in self.syllable_list:
            if (syllable[0] in idx_list) or (syllable[2] > lenth_limit): continue
            else: syllable_selected.append(syllable)
        return np.array(syllable_selected)
