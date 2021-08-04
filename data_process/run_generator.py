from syllable_generator import *
from find_abnormal import *
from data_process import *

org_path = '/Volumes/BPlus/dataset/SongBird/bengal/org/'
save_path = '../3-Analysis/bengal/0728_test/' 
lenth_limit = 8000


sg = SyllableGenerator(org_path)

wav_list = sg.wav_list
print('* Wav list : ', len(wav_list))
print(wav_list)

#wav list -> syllable list
syllable_list = sg.get_syllable_list(wav_list)
print('* Syllable list shape: ', syllable_list.shape)
print(syllable_list[0])

# 0: idx, 1: start, 2: lenth
hist_syllable_lenth(syllable_list[:,2], save='%s/hist_len'%save_path)

# wav list & syllable list -> final syllable list
fa=FindAbnormal(wav_list, syllable_list)
final_syllable_list= fa.remove_abnormal([], lenth_limit=lenth_limit)
print('* Final syllable list: ', final_syllable_list.shape)


# wav list & final syllable list -> valset, imgset
dp = DataProcess(wav_list, final_syllable_list, width=64, height=128)
valset, imgset = dp.get_dataset()

print('* datasets: ', valset.shape, imgset.shape)

print('* Save at , ', save_path)
np.save("%s/label"%save_path, final_syllable_list)
np.save("%s/imgset"%save_path, imgset)
np.save("%s/valset"%save_path, valset)
