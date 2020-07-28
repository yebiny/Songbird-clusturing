from Draw import *

def read_wav(wav_file):
    y, sr = librosa.load(wav_file, sr=32000)
    x = np.linspace(0, len(y)/sr, len(y)) # time axis
    return x, y

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

