import tensorflow as tf
import os, sys
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model
from tensorflow.keras import layers, models
from tensorflow.keras import backend as K

class VAE():

    def __init__(self, encoder, decoder, img_shape):
        self.encoder = encoder
        self.decoder = decoder
        self.img_shape = img_shape

    def compile(self, optimizer = tf.keras.optimizers.Adam(0.001)):
        inputs = layers.Input(shape=self.img_shape, name = 'vae_inputs')
        z_log_var, z_mean, z = self.encoder(inputs)
        outputs = self.decoder(z)
        self.vae = models.Model(inputs, outputs, name = 'vae')
        self.optimizer = optimizer
    
    def _make_dataset(self, x_val, x_len,  batch_size):
        train_ds = tf.data.Dataset.from_tensor_slices((x_val, x_len)).batch(batch_size)
        return train_ds
    
    def _get_rec_loss(self, inputs, predictions):
        mask = K.all(K.equal(inputs, 0), axis=-1)
        mask = 1-K.cast(mask, K.floatx())
        
        rec_loss = tf.keras.losses.mse(inputs, predictions) * mask
        rec_loss = tf.reduce_mean(rec_loss)
        return rec_loss
    
    def _get_kl_loss(self, z_log_var, z_mean):
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = tf.reduce_mean(kl_loss)
        kl_loss *= -0.5
        return kl_loss
    
    @tf.function
    def _train_step(self, vals, lens):
        with tf.GradientTape() as tape:
    
            # Get model ouputs
            z_log_var, z_mean, z = self.encoder(vals)
            rec_vals = self.decoder(z)
         
            rec_loss = self._get_rec_loss(vals, rec_vals)
            kl_loss = self._get_kl_loss(z_log_var, z_mean)
            loss = rec_loss + kl_loss
    
        # Compute gradients
        varialbes = self.vae.trainable_variables
        gradients = tape.gradient(loss, varialbes)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, varialbes))
    
        return loss, rec_loss, kl_loss 
    

    def _save_best_model(self, mn_loss, history,save_path):
        base_loss = history['loss'][-1]
        
        if mn_loss >= base_loss: 
            self.save_model(save_path)
            mn_loss = base_loss
            print('save model')
        
        return mn_loss

    def fit(self, x_val, x_len,  epochs=1, batch_size=16, img_iter=1, save_path=None):
        
        train_ds = self._make_dataset(x_val, x_len, batch_size)
        loss_name = ['loss', 'rec_loss', 'kl_loss']
        
        ## set history ##
        history={ name:[] for name in loss_name}
                
        ## epoch ## 
        for epoch in range(1, 1+epochs):
            self.epoch=epoch
            for h in history: history[h].append(0)
            
            ## batch-trainset ##
            for batch_vals, batch_lens in train_ds:
                losses = self._train_step(batch_vals, batch_lens)
                 
                for name, loss in zip(loss_name, losses): history[name][-1]+=loss
            
            
            ## print losses ##
            print('* %i / %i : loss: %f'%(epoch, epochs, history['loss'][-1]))
            
            ## save sample image ##
            if epoch%img_iter==0:
                print(batch_vals.shape)
                self.plot_sample_imgs(batch_vals, save_path=save_path)

            ## save best model ##
            if save_path:
                if epoch==1: mn_loss=base_loss
                mn_loss = self._save_best_model(mn_loss, history, save_path)
        
        return history 

    def plot_sample_imgs(self, imgs, n=10, save_path=None):
        plt.figure(figsize=(n,2))
        rec_vals = self.vae.predict(imgs[:n])
        print(imgs.shape, rec_vals.shape)
        for i, (img, rec_img) in enumerate(zip(imgs, rec_vals)):
            plt.subplot(2,n,i+1)
            plt.plot(img[:,0], c='orange', alpha=0.7)
            plt.xticks([])
            plt.yticks([])
            plt.subplot(2,n,n+i+1)
            plt.plot(rec_img[:,0], c='blue', alpha=0.7)
            plt.xticks([])
            plt.yticks([])
        
        if save_path!=None: 
            plt.savefig('%s/sample_%i'%(save_path, self.epoch))
        else: plt.show()

    def save_model(self, save_path):
        self.encoder.save('%s/encoder.h5'%save_path)
        self.decoder.save('%s/decoder.h5'%save_path)
        self.vae.save('%s/vae.h5'%save_path)

    def plot_model(self, save_path):
        plot_model(self.encoder, to_file='%s/encoder.png'%save_path, show_shapes=True)
        plot_model(self.decoder, to_file='%s/decoder.png'%save_path, show_shapes=True)
        plot_model(self.vae, to_file='%s/vae.png'%save_path, show_shapes=True)
    
    def load_weight(self, weight_path):
        self.vae.load_weights(weight_path)
