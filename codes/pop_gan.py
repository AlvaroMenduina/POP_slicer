# ==================================================================
#       Physical Optics Propagation (POP) - Machine Learning
#             Generative Adversarial Networks (GAN)
# ==================================================================
#
# Very preliminary attempt to use GAN to generate realistic NCPA maps
#
# Based on the Keras-GAN package at: https://github.com/eriklindernoren/Keras-GAN

import os
import numpy as np
import matplotlib.pyplot as plt
import pop_methods as pop

from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import RMSprop

import keras.backend as K

def data_augmentation(data_set):
    copy_set = data_set.copy()
    extended_data_set = data_set.copy()
    photon = [100., 1000., 10000.]
    for f in photon:
        noise = np.random.poisson(lam=f * copy_set) / f
        extended_data_set = np.concatenate((extended_data_set, noise), axis=0)
    return extended_data_set


class WGAN():
    def __init__(self, image_shape):
        self.N, self.M = image_shape
        self.channels = 1
        self.img_shape = (self.N, self.M, self.channels)
        self.latent_dim = 100

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        self.clip_value = 0.01
        optimizer = RMSprop(lr=0.00005)

        # Build and compile the critic
        self.critic = self.build_critic()
        self.critic.compile(loss=self.wasserstein_loss,
                            optimizer=optimizer,
                            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generated imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.critic.trainable = False

        # The critic takes generated images as input and determines validity
        valid = self.critic(img)

        # The combined model  (stacked generator and critic)
        self.combined = Model(z, valid)
        self.combined.compile(loss=self.wasserstein_loss,
                              optimizer=optimizer,
                              metrics=['accuracy'])

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self):

        model = Sequential()

        # model.add(Dense(256 * 7 * 7, activation="relu", input_dim=self.latent_dim))
        # model.add(Reshape((7, 7, 256)))
        # model.add(UpSampling2D())

        model.add(Dense(128 * 7 * 7, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((7, 7, 128)))
        model.add(UpSampling2D())

        model.add(Conv2D(128, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())

        model.add(Conv2D(64, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(Conv2D(self.channels, kernel_size=4, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_critic(self):

        model = Sequential()

        model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, sample_interval=50):

        data_set = os.path.join(path_results, 'GAN_PSF.npz')
        compressed = np.load(data_set)
        X_train = compressed['psf']

        # Crop the arrays to 28x28
        N, M = X_train.shape[1], X_train.shape[2]
        pix_min = N // 2 - 28//2
        pix_max = M // 2 + 28//2
        X_train = X_train[:, pix_min:pix_max, pix_min:pix_max]

        # Run Data Augmentation (Poisson Noise)
        X_train = data_augmentation(X_train)

        # Show some random PSFs
        idx = np.random.randint(0, X_train.shape[0], 25)
        imgs = X_train[idx]
        print('\n a')
        print(imgs.shape)

        fig, axs = plt.subplots(5, 5)
        cnt = 0
        for i in range(5):
            for j in range(5):
                axs[i, j].imshow(imgs[cnt, :, :], cmap='jet')
                axs[i, j].axis('off')
                cnt += 1
        file_name = os.path.join(path_results, "PSFs.png")
        fig.savefig(file_name)
        plt.close()

        # Rescale -1 to 1
        X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))

        for epoch in range(epochs):

            for _ in range(self.n_critic):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                imgs = X_train[idx]
                # print(np.mean(imgs))
                # print('Batch shape: ', imgs.shape)

                # Sample noise as generator input
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                # print('Noise shape: ', noise.shape)

                # Generate a batch of new images
                gen_imgs = self.generator.predict(noise)
                # print('Gen shape: ', gen_imgs.shape)

                # Train the critic
                d_loss_real = self.critic.train_on_batch(imgs, valid)
                d_loss_fake = self.critic.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

                # Clip critic weights
                for l in self.critic.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                    l.set_weights(weights)

            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print("%d [D loss: %f] [G loss: %f]" % (epoch, 1 - d_loss[0], 1 - g_loss[0]))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 1

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='jet')
                axs[i, j].axis('off')
                cnt += 1
        file_name = os.path.join(path_results, "images_%d.png" % epoch)
        fig.savefig(file_name)
        plt.close()

""" PARAMETERS """

path_real_data = os.path.join('zemax_files', 'ML')
path_results = os.path.join('results', 'GAN')

if __name__ == "__main__":

    image_shape = (28, 28)
    wgan = WGAN(image_shape)
    wgan.train(epochs=5000, batch_size=128, sample_interval=50)