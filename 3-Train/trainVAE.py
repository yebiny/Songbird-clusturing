import tensorflow as tf
import numpy as np
import pandas as pd
import os

from models import *
from basic import *

def reduce_lr(pre_v_loss, v_loss, count, lr, patience, factor, min_lr):
    if v_loss < pre_v_loss:
        count = 0
    else:
        count += 1
        if count >= patience: 
            lr = lr*factor
            if lr < min_lr: 
                lr = min_lr
            count = 0
            print('reduce learning rate..', lr)    
    return count, lr

class TrainVAE():

    def __init__(self, latent_dim, data_path, save_dir,  ckp='y'):
        self.data_path = data_path
        self.save_path = '%s/%s'%(data_path, save_dir)
        if_not_make(self.save_path)

        self.x_train = np.load(data_path+'pre/x_train.npy')
        self.x_test = np.load(data_path+'pre/x_test.npy')
        self.ckp_dir = self.save_path+'/ckp/'
        self.npy_dir = self.save_path+'/npy/'
        if_not_make(self.npy_dir)

        encoder, decoder, vae = build_vae(self.x_train, latent_dim)
        self.encoder = encoder
        self.decoder = decoder
        self.vae = vae
        
        self.checkpoint = tf.train.Checkpoint(step=tf.Variable(1), encoder=self.encoder, decoder=self.decoder, vae=self.vae)
        if ckp=='y':
            self.checkpoint.restore(tf.train.latest_checkpoint(self.ckp_dir))

    def get_rec_loss(self, inputs, predictions):
        rec_loss = tf.keras.losses.binary_crossentropy(inputs, predictions)
        rec_loss = tf.reduce_mean(rec_loss)
        rec_loss *= self.x_train.shape[1]*self.x_train.shape[2]
        return rec_loss
    
    def get_kl_loss(self, z_log_var, z_mean):
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = tf.reduce_mean(kl_loss)
        kl_loss *= -0.5
        return kl_loss
  
    @tf.function
    def train_step(self, inputs, optimizer, train_loss):
        with tf.GradientTape() as tape:
    
            # Get model ouputs
            z_log_var, z_mean, z = self.encoder(inputs)
            predictions = self.decoder(z)
    
            # Compute losses
            rec_loss = self.get_rec_loss(inputs, predictions)
            kl_loss = self.get_kl_loss(z_log_var, z_mean)
            loss = rec_loss + kl_loss
    
        # Compute gradients
        varialbes = self.vae.trainable_variables
        gradients = tape.gradient(loss, varialbes)
        # Update weights
        optimizer.apply_gradients(zip(gradients, varialbes))
    
        # Update train loss
        train_loss(loss)

    def train(self, epochs, batch_size, init_lr=0.001):
        
        train_ds = tf.data.Dataset.from_tensor_slices((self.x_train, self.x_train)).batch(batch_size)
        csv_logger = tf.keras.callbacks.CSVLogger(self.save_path+'/training.log')

        optimizer = tf.keras.optimizers.Adam(init_lr)
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        valid_loss = tf.keras.metrics.Mean(name='valid_loss') 
        
        # Initialize values
        best_loss, count = float('inf'), 0
        
        # Start epoch loop
        for epoch in range(epochs):
            for inputs, outputs in train_ds:
                self.train_step(inputs, optimizer, train_loss)
            
            # Get loss and leraning rate at this epoch
            t_loss = train_loss.result().numpy() 
            l_rate = optimizer.learning_rate.numpy()
        
            # Control learning rate
            count, lr  = reduce_lr(best_loss, t_loss, count, l_rate, 5, 0.2, 0.00001)
            optimizer.learning_rate = lr
            
            # Save checkpoint if best v_loss 
            if t_loss < best_loss:
                best_loss = t_loss
                self.checkpoint.save(file_prefix=os.path.join(self.save_path+'/ckp/', 'ckp'))
            
            # Save loss, lerning rate
            print("* %i * loss: %f,  best_loss: %f, l_rate: %f, lr_count: %i"%(epoch, t_loss, best_loss, l_rate, count ))
            df = pd.DataFrame({'epoch':[epoch], 'loss':[t_loss], 'best_loss':[best_loss], 'l_rate':[l_rate]  } )
            df.to_csv(self.save_path+'/process.csv', mode='a', header=False)
            
            # Reset loss
            train_loss.reset_states()   

    def save_latent_val(self, save=None):

        rec = self.vae.predict(self.x_train)
        lat = self.encoder.predict(self.x_train)[2]
        rec_test = self.vae.predict(self.x_test)
        lat_test = self.encoder.predict(self.x_test)[2]

        if save !=None:
            np.save(self.npy_dir+'/rec', rec)
            np.save(self.npy_dir+'/lat', lat)
            np.save(self.npy_dir+'/rec_test', rec_test)
            np.save(self.npy_dir+'/lat_test', lat_test)
            print(rec.shape, lat.shape, rec_test.shape, lat_test.shape)
        
        return rec, lat, rec_test, lat_test


    def plot_recimg(self, idx):
        org = self.x_train[idx]
        rec = self.vae.predict(org[np.newaxis, :])[0]

        orgimg = org.reshape(org.shape[:-1])
        recimg = rec.reshape(rec.shape[:-1])
        
        figure = plt.figure(figsize=(4,4))
        
        plt.subplot(1,2,1)
        plt.yticks([])
        plt.xticks([])
        plt.title('Org') 
        plt.imshow(orgimg)

        plt.subplot(1,2,2)
        plt.yticks([])
        plt.xticks([])
        plt.title('Rec')
        plt.imshow(recimg)
        
        plt.show()
        
