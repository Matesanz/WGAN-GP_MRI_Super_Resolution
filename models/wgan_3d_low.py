from os import path
import nibabel as nib
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Conv3DTranspose, Reshape, LeakyReLU, Flatten, Conv3D
from tensorflow.keras.optimizers import Adam
from utils.training import DataGenerator
from utils.imutils import process_mnc_and_reduce
import numpy as np
from utils.imutils import pad_image
from models.wgan_3d import DCWGAN
from scipy.ndimage import zoom
from utils.minc_viewer import Viewer


if __name__ == '__main__':

        init = tf.initializers.RandomNormal(stddev=0.02)
        generator = Sequential(
                [       # Input
                        Input(shape=(10,), name='z_input'),
                        # 1st Deconvolution
                        Dense(2 * 2 * 2 * 128),
                        LeakyReLU(alpha=0.2, name='lrelu_1'),
                        Reshape((2, 2, 2, 128), name="conv_1"),
                        # 2nd Deconvolution
                        Conv3DTranspose(
                                filters=64,
                                kernel_size=5,
                                strides=(2, 2, 2),
                                kernel_initializer=init,
                                use_bias=True,
                                padding="same",
                                name="conv_2"),
                        LeakyReLU(alpha=0.2, name='lrelu_2'),
                        # 3rd Deconvolution
                        Conv3DTranspose(
                                filters=32,
                                kernel_size=5,
                                strides=(2, 2, 2),
                                kernel_initializer=init,
                                use_bias=True,
                                padding='same',
                                name="conv_3"),
                        LeakyReLU(alpha=0.2, name='lrelu_3'),
                        # Output
                        Conv3DTranspose(
                                filters=1,
                                kernel_size=5,
                                strides=(2, 2, 2),
                                kernel_initializer=init,
                                activation='linear',
                                use_bias=True,
                                padding='same',
                                name='output')
                ],
                name="generator",
        )

        critic = Sequential(
                [       # Input
                        Input(shape=(16, 16, 16, 1), name='input'),
                        # 1st Convolution
                        Conv3D(
                                filters=32,
                                kernel_size=5,
                                strides=(2, 2, 2),
                                kernel_initializer=init,
                                use_bias=True,
                                padding="same",
                                name="conv_1"),
                        LeakyReLU(alpha=0.2, name='lrelu_1'),
                        # 2nd Convolution
                        Conv3D(
                                filters=64,
                                kernel_size=5,
                                strides=(2, 2, 2),
                                kernel_initializer=init,
                                use_bias=True,
                                padding='same',
                                name="conv_2"),
                        LeakyReLU(alpha=0.2, name='lrelu_2'),
                        # 3rd Convolution
                        Conv3D(
                                filters=128,
                                kernel_size=5,
                                strides=(2, 2, 2),
                                kernel_initializer=init,
                                use_bias=True,
                                padding='same',
                                name='conv_3'),
                        LeakyReLU(alpha=0.2, name='lrelu_3'),
                        # Output
                        Flatten(),
                        Dense(1, activation=None, name='output', kernel_initializer=init)
                ],
                name="critic",
        )

        wgan = DCWGAN(generator=generator, critic=critic)

        IMAGE_PATH = path.join('..', 'resources', 'mri')
        data_generator = DataGenerator(IMAGE_PATH, process_mnc_and_reduce)

        # --------------------
        #  PARAMETER INIT
        # --------------------

        batch_size = 4  # Samples every epoch
        n_epochs = 10  # Training Epochs
        plot_interval = 10  # Every plot_interval create a graph with real and generated data distribution
        c_loops = 5  # number of loops to train critic every epoch
        z_control = tf.random.normal((batch_size, wgan.z_units))

        # --------------------
        #  TENSORBOARD SETUP
        # --------------------
        generator_train_loss = tf.keras.metrics.Mean('generator_train_loss', dtype=tf.float32)
        critic_train_loss = tf.keras.metrics.Mean('critic_train_loss', dtype=tf.float32)

        for epoch in range(n_epochs):

                # --------------------
                #     TRAINING
                # --------------------

                # Train Critic
                for _ in range(c_loops):
                        # TODO: Import real images Batch
                        batch = data_generator.get_batch(batch_size)
                        c_loss = wgan.train_critic(batch)

                # Train Generator
                g_loss = wgan.train_generator(batch_size)  # Train our model on real distribution points
                # c_loss = wgan.compute_critic_loss(batch)  # Get batch loss to track data
                # g_loss = wgan.compute_generator_loss(batch)

                # -----------------------
                #  TENSORBOARD TRACKING
                # ------------------------

                # Save generator and critic losses
                generator_train_loss(g_loss)
                critic_train_loss(c_loss)

                # track data through console
                template = 'Epoch {}, Gen Loss: {}, Dis Loss {}'
                print(template.format(epoch + 1,
                                      generator_train_loss.result(),
                                      critic_train_loss.result()))

        fake = generator(z_control)[0]
        fake = np.squeeze(fake, 3)
        print(fake.min(), fake.max())
        Viewer(fake)
