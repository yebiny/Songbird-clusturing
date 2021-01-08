from models import *
from tensorflow.keras import callbacks
from tensorflow.keras.utils import plot_model

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os, argparse

def plot_loss(history, save_path):
    plt.plot(history.history['loss'], marker='o')
    plt.grid(True)
    plt.legend(('loss'))
    plt.savefig('%s/loss.png'%save_path)

def trainVAE(x_train, epochs, save_path, z_dim, batch_size=64):
    if not os.path.exists(save_path): os.makedirs(save_path)
    print(x_train.shape)

    # Set model
    encoder, decoder, vae = build_vae(x_train, z_dim)
    vae.summary()
   
    # Custom vae_loss
    def vae_loss(x, rec_x):
        z_mean, z_log_var, z = encoder(x)
        # 1.reconstruct loss
        rec_x = decoder(z)
        rec_loss = tf.keras.losses.binary_crossentropy(x, rec_x)
        rec_loss = tf.reduce_mean(rec_loss)
        rec_loss *= (128*64)
        # 2. KL Divergence loss
        kl_loss = 1+z_log_var-tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = -0.5*tf.reduce_mean(kl_loss)
    
        total_loss = rec_loss+kl_loss
        return total_loss

    # Compile with custom loss
    vae.compile(optimizer='adam', loss=vae_loss)

    # Set callbacks
    ckp = callbacks.ModelCheckpoint(filepath=save_path+'/model.h5', monitor='loss', verbose=1, save_best_only=True)
    csv_logger = callbacks.CSVLogger(save_path + '/logger.csv')
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=1e-5)
    
    # Train
    history = vae.fit(
              x_train, x_train, epochs=epochs, batch_size=batch_size,
              callbacks=[ckp, reduce_lr, csv_logger]                   
            )

    # Plotining
    plot_loss(history, save_path)
    plot_model(encoder, to_file=save_path+'/vae_encoder.png', show_shapes=True)
    plot_model(decoder, to_file=save_path+'/vae_decoder.png', show_shapes=True)

def main():
    opt = argparse.ArgumentParser()
    opt.add_argument(dest='data_path', type=str, help='datasets path')
    opt.add_argument(dest='save_path', type=str, help='save path')
    opt.add_argument('-e',  dest='epochs', type=int, default=5, help='epochs')
    opt.add_argument('-z',  dest='z_dim', type=int, default=10, help='z dimension')
    argv = opt.parse_args()
    print(argv.save_path, argv.epochs)

    x_train  = np.load(argv.data_path)
    trainVAE(x_train, argv.epochs, argv.save_path, argv.z_dim)

if __name__=='__main__':
    main()
