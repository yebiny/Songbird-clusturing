import tensorflow as tf
import numpy as np
import pandas as pd
import os

from models import *

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

    def __init__(self, x_train , latent_dim, save_path):
        self.x_train = x_train
        self.save_path = save_path
    
        encoder, decoder, vae = build_vae(x_train, latent_dim)
        self.encoder = encoder
        self.decoder = decoder
        self.vae = vae

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

    def train(self, epochs, batch_size):
        
        train_ds = tf.data.Dataset.from_tensor_slices((self.x_train, self.x_train)).batch(batch_size)
        
        ckp_dir = self.save_path+'/ckp/'
        checkpoint = tf.train.Checkpoint(step=tf.Variable(1), encoder=self.encoder, decoder=self.decoder, vae=self.vae)
        checkpoint.restore(tf.train.latest_checkpoint(ckp_dir))
        csv_logger = tf.keras.callbacks.CSVLogger(self.save_path+'/training.log')

        optimizer = tf.keras.optimizers.Adam(0.001)
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
                checkpoint.save(file_prefix=os.path.join(self.save_path+'/ckp/', 'ckp'))
            
            # Save loss, lerning rate
            print("* %i * loss: %f,  best_loss: %f, l_rate: %f, lr_count: %i"%(epoch, t_loss, best_loss, l_rate, count ))
            df = pd.DataFrame({'epoch':[epoch], 'loss':[t_loss], 'best_loss':[best_loss], 'l_rate':[l_rate]  } )
            df.to_csv(self.save_path+'/process.csv', mode='a', header=False)
            
            # Reset loss
            train_loss.reset_states()    
