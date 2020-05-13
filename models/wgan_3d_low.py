from os import path
from numpy import squeeze
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input, Conv3DTranspose, Reshape, LeakyReLU, Flatten, Conv3D
from tensorflow.keras.optimizers import Adam
from utils.training import DataGenerator, save_models
from utils.imutils import process_mnc_and_reduce
from utils.minc_viewer import Viewer
from models.wgan_3d import DCWGAN

# Initialize NN weights
init = tf.initializers.RandomNormal(stddev=0.02)

# Create generator Graph
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

# Create critic Graph
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

if __name__ == '__main__':

        # Create adversarial graph
        gen_opt = Adam()
        critic_opt = Adam()
        wgan = DCWGAN(generator=generator, critic=critic, g_opt=gen_opt, c_opt=critic_opt)

        # Path to mnc files
        IMAGES_PATH = path.join('..', 'resources', 'mri')
        data_generator = DataGenerator(IMAGES_PATH, process_mnc_and_reduce)

        # --------------------
        #  PARAMETER INIT
        # --------------------

        batch_size = 4  # Samples every epoch
        n_epochs = 1  # Training Epochs
        plot_interval = 10  # Every plot_interval create a graph with real and generated data distribution
        c_loops = 5  # number of loops to train critic every epoch
        z_control = tf.random.normal((batch_size, wgan.z_units))  # Vector to feed gen and control training evolution

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

                        batch = data_generator.get_batch(batch_size)  # Collects Batch of real images
                        c_loss = wgan.train_critic(batch)  # Train and get critic loss

                # Train Generator
                g_loss = wgan.train_generator(batch_size)  # Train our model on real distribution points

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

        # save models after training
        save_models(critic, generator, None, "dc_wgan")

        # generate fake sample to visualize
        fake = generator(z_control)[0]
        fake = squeeze(fake, 3)
        print(fake.min(), fake.max())
        Viewer(fake)
