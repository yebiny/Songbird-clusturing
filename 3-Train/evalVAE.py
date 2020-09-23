from models import *
import matplotlib.pyplot as plt
import numpy as np
import argparse
def get_z_rec(x_data, model, save_path=None):
    encoder = model.layers[1]
    z = encoder.predict(x_data)[2]
    x_rec = model.predict(x_data)
    
    if save_path:
        np.save('%s/x_lat'%save_path, z)
        np.save('%s/x_rec'%save_path, x_rec)
        print('** save dataset')
    return z, x_rec

def plot_recimg(x_data, x_rec, save_path, g=1):
    
    def plot_img(img, title=None):
        plt.yticks([])
        plt.xticks([])
        plt.title(title)
        plt.imshow(img)

    orgimg = x_data.reshape(x_data.shape[:-1])
    recimg = x_rec.reshape(x_rec.shape[:-1])

    w, h = 8,2
    figure = plt.figure(figsize=(w*2,h*4))
    for idx in range(w):

        gidx = idx+w*g
        
        plt.subplot(2,w,idx+1)
        plot_img(orgimg[gidx], title=gidx)
        plt.subplot(2,w,idx+w+1)
        plot_img(recimg[gidx])

    if save_path:
        plt.savefig(save_path+'/plot_recimg_%i'%g)
        print('** save plot')
    else:
        plt.show()

def main():
    opt = argparse.ArgumentParser()
    opt.add_argument(dest='path', type=str, help='model path')
    opt.add_argument('-s',  dest='save', type=str, default='y', help='save or not')
    argv = opt.parse_args()
    print('* Working space: ', argv.path)
    
    model_path='%s/model.h5'%argv.path
    data_path ='%s/../pre/data.npy'%argv.path
    test_path ='%s/../pre/x_test.npy'%argv.path

    data = np.load(data_path)
    x_test = np.load(test_path)
    print('* Load dataset: ', data.shape, x_test.shape)
    model = tf.keras.models.load_model(model_path, compile=False)
    model.summary()
   
    print('* Save reconstructed image with testset')
    z, x_rec = get_z_rec(x_test, model)
    plot_recimg(x_test, x_rec, argv.path)
    
    print('* Save latent vectors')
    if argv.save == 'y':
        z, x_rec = get_z_rec(data, model, argv.path)

if __name__=='__main__':
    main()
