import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras import backend as K


class WaveVAE():
    def __init__(self, img_shape, val_shape, z_dim):
        self.img_shape = img_shape
        self.val_shape = val_shape
        self.z_dim = z_dim
        
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        
    def _sampling(self, args):
        """Reparameterization function by sampling from an isotropic unit Gaussian.
        # Arguments:
            args (tensor): mean and log of variance of Q(z|X)
        # Returns:
            z (tensor): sampled latent vector
        """
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean=0 and std=1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon


    def _Conv1DTranspose(self, input_tensor, filters, kernel_size, strides=2, dr=1, padding='same', activation=None):
        """
            input_tensor: tensor, with the shape (batch_size, time_steps, dims)
            filters: int, output dimension, i.e. the output tensor will have the shape of (batch_size, time_steps, filters)
            kernel_size: int, size of the convolution kernel
            strides: int, convolution step size
            padding: 'same' | 'valid'
        """
        x = layers.Lambda(lambda x: K.expand_dims(x, axis=2))(input_tensor)
        x = layers.Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1), 
                                   strides=(strides, 1), dilation_rate=(dr, dr), padding=padding,
                                  activation=activation)(x)
        x = layers.Lambda(lambda x: K.squeeze(x, axis=2))(x)
        return x

    
    def build_encoder(self, n_filters=20):
        img_inputs = layers.Input(shape=self.img_shape, name = 'img_inputs')
        

        y = layers.Conv2D(32, 3, strides=2, padding="same")(img_inputs)
        y = layers.LeakyReLU()(y)
        y = layers.Conv2D(32, 3, strides=2, padding="same")(y)
        y = layers.LeakyReLU()(y)
        y = layers.Conv2D(32, 3, strides=2, padding="same")(y)
        y = layers.LeakyReLU()(y)
        y = layers.Conv2D(32, 3, strides=2, padding="same")(y)
        y = layers.LeakyReLU()(y)
        self.y_shape = y.shape

        y = layers.Flatten()(y)
        y = layers.Dense(256, activation='relu')(y)
        img_y = layers.Dense(256, activation='relu')(y)
        
        img_encoder = models.Model(img_inputs, img_y, name='ImgEncoder')
        
        val_inputs = layers.Input(shape=self.val_shape, name = 'val_inputs')
        y = layers.Masking(mask_value=0.0)(val_inputs)

        y = layers.Conv1D(n_filters, 2, 
                          dilation_rate=1, 
                          padding='causal',
                          name = 'dilated_conv_1')(y)
        for dr in (2,4,8,16):
            y = layers.Conv1D(n_filters, 2, 
                              dilation_rate=dr, 
                              padding='causal',
                              name='dilated_conv_%i'%dr)(y)
        
        y = layers.AveragePooling1D(20, padding='same')(y)
        y = layers.Conv1D(20, 100, padding='same', activation='relu')(y)
        y = layers.Conv1D(20, 100, padding='same', activation='relu')(y)
        y = layers.AveragePooling1D(20, padding='same')(y)
        val_y = layers.Reshape((y.shape[1]*y.shape[2],))(y)
        val_encoder = models.Model(val_inputs, val_y, name='ValEncoder')

        y_img = img_encoder(img_inputs)
        y_val = val_encoder(val_inputs)
        y_img = layers.Dropout(0.5)(y_img)
        
        y = layers.Concatenate()([y_img, y_val])
        z_mean = layers.Dense(self.z_dim)(y)
        z_log_var = layers.Dense(self.z_dim)(y)
        z = layers.Lambda(self._sampling)([z_mean, z_log_var])
        model = models.Model([img_inputs, val_inputs], [z_mean, z_log_var, z] )                          
        return model
                                  
                            
        
    def build_decoder(self):
        x = layers.Input(shape=(self.z_dim,))

        y = layers.Dense(256, activation='relu')(x)
        y = layers.Dense(256, activation='relu')(y)
        y = layers.Dense(self.y_shape[1]*self.y_shape[2]*self.y_shape[3], activation="relu")(y)
        y = layers.Reshape(self.y_shape[1:])(y)

        y = layers.Conv2DTranspose(32, 3, strides=2, padding="same")(y)
        y = layers.LeakyReLU()(y)
        y = layers.Conv2DTranspose(32, 3, strides=2, padding="same")(y)
        y = layers.LeakyReLU()(y)
        y = layers.Conv2DTranspose(32, 3, strides=2, padding="same")(y)
        y = layers.LeakyReLU()(y)
        y = layers.Conv2DTranspose(self.img_shape[-1], 3, strides=2, padding="same")(y)
        y = layers.Activation('sigmoid')(y)
        
        model = models.Model(x, y, name='Decoder')
        return model


def main():
    IMG_SHAPE = (128,64,1)
    VAL_SHAPE = (9000,1)

    wv = WaveVAE(IMG_SHAPE, VAL_SHAPE, 10)

    wv.encoder.summary()
    wv.decoder.summary()

if __name__=='__main__':
    main()
