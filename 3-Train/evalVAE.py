from models import *
import matplotlib.pyplot as plt
import numpy as np

def get_z_rec(x_test, model, save_path=None):
    encoder = model.layers[1]
    z = encoder.predict(x_test)[2]
    x_rec = model.predict(x_test)
    
    if save_path:
        np.save('%s/x_lat'%save_path, z)
        np.save('%s/x_rec'%save_path, x_rec)
        print('save dataset')
    return z, x_rec

def plot_recimg(x_test, x_rec, g=1):
    
    def plot_img(img, title=None):
        plt.yticks([])
        plt.xticks([])
        plt.title(title)
        plt.imshow(img)

    orgimg = x_test.reshape(x_test.shape[:-1])
    recimg = x_rec.reshape(x_rec.shape[:-1])

    w, h = 8,2
    figure = plt.figure(figsize=(w*2,h*4))
    for idx in range(w):

        gidx = idx+w*g
        
        plt.subplot(2,w,idx+1)
        plot_img(orgimg[gidx], title=gidx)
        plt.subplot(2,w,idx+w+1)
        plot_img(recimg[gidx])

    plt.show()

def main():
    model_path='../5-Results/bengal/0812/train2/model.h5'
    data_path ='../5-Results/bengal/0812/pre/data.npy'
    
    data = np.load(data_path)
    model = tf.keras.models.load_model(model_path, compile=False)
    model.summary()
    
if __name__=='__main__':
    main()
