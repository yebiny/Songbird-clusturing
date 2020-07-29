import io
import cv2
from def_dict import *
from splitWavProcess import *

def get_img_from_fig(fig, dpi=100):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 
    return img

class PreProcess():

    def __init__(self, path):
        self.wav_list = np.load('%s/wav_list.npy'%(path))
        self.sylla_list = np.load('%s/sylla_list.npy'%(path))

    def draw_wav(self, idx):

        wav=self.wav_list[idx]        
        name = wav.split('/')[-1].split('.')[0]
        
        draw_split_wav(wav, 'Idx %i: %s'%(idx,name))
        draw_specgram(wav)
        
        x, y = wavfile.read(wav)
        target = self.sylla_list[self.sylla_list[:,0]==idx]
        fig = plt.figure(figsize=(20,3))
        for i in range(len(target)):
            start = target[i][1]
            end = start+target[i][2]
            
            plt.subplot(1,len(target),i+1)
            plt.title(i)

            plt.xticks([])
            plt.yticks([])
            plt.specgram(y[:,0][start:end], Fs = x)

        plt.show()

    def get_syllable_value(self, idx):
    
        wav_idx = self.sylla_list[idx][0]
        start = self.sylla_list[idx][1]
        lenth = self.sylla_list[idx][2]
        end = start+lenth
    
        wav = self.wav_list[wav_idx]
        x, y = wavfile.read(wav)
        val = y[:,0][start:end]
    
        return x, val
    
    def draw_syllable(self, idx):
        
        x, val = self.get_syllable_value(idx)

        fig = plt.figure(figsize=(2,3))

        plt.subplot(1,2,1)
        plt.specgram(val, Fs = x)
        plt.axis('off'), plt.xticks([]), plt.yticks([])
        plt.tight_layout()
        plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0, wspace = 0)
        
        plt.subplot(1,2,2)
        plt.plot(val, color='k')
        plt.axis('off'), plt.xticks([]), plt.yticks([])
        plt.tight_layout()
        plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0, wspace = 0)
        
        plt.show()
        plt.close('all')
        
        xsize = round(round(len(val)/1000, 2)/6,2)
        ysize = 12/6
        print(xsize, ysize)

    def get_img_arr(self, idx, xsize, ysize, opt='gray'):
        
        x, val = self.get_syllable_value(idx)
        
        xlen = xsize/100
        ylen = ysize/100
        
        fig = plt.figure(figsize=(xlen, ylen))
        
        if opt =='gray':
            plt.specgram(val, Fs = x, cmap='gray')
        else: 
            plt.specgram(val, Fs = x)
        plt.axis('off'), plt.xticks([]), plt.yticks([])
        plt.tight_layout()
        plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0, wspace = 0)
        #plt.show()
        plt.close('all')
        
        # you can get a high-resolution image as numpy array!!
        arr = get_img_from_fig(fig)    
        if opt=='gray':
            arr=arr[:,:,0]

        return arr
