from time import strftime, localtime, time

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.layers import Add, BatchNormalization, Conv2D, Dense, Flatten, Input, LeakyReLU, PReLU, \
        Lambda, Conv3D, Conv3DTranspose, Reshape
from tensorflow.keras.losses import MeanSquaredError
from numpy import load, squeeze, expand_dims
from os import path
from utils.minc_viewer import Viewer
from utils.training import DataGenerator


#  Code based on https://github.com/krasserm/super-resolution/blob/master/model/srgan.py
from utils.imutils import process_mnc


def upsample(x_in, num_filters):
        x = Conv2D(num_filters, kernel_size=3, padding='same')(x_in)
        # x = Lambda(pixel_shuffle(scale=2))(x)
        return PReLU(shared_axes=[1, 2])(x)


def res_block(x_in, num_filters, momentum=0.8):
        x = Conv2D(num_filters, kernel_size=3, padding='same')(x_in)
        x = BatchNormalization(momentum=momentum)(x)
        x = PReLU(shared_axes=[1, 2])(x)
        x = Conv2D(num_filters, kernel_size=3, padding='same')(x)
        x = BatchNormalization(momentum=momentum)(x)
        x = Add()([x_in, x])
        return x

def _vgg(output_layer):
        vgg = VGG19(input_shape=(None, None, 3), include_top=False)
        return Model(vgg.input, vgg.layers[output_layer].output)



class SRWGAN(Model):

        """
        Keras Model that takes Generator and Critic Graphs as input
        and trains them adversarially
        """

        def __init__(self,
                     generator,
                     critic,
                     cut_layer=4,
                     g_opt=Adam(lr=5e-5, beta_1=0.5),
                     c_opt=Adam(lr=5e-5, beta_1=0.5),
                     gradient_penalty_weight=10):

                super(SRWGAN, self).__init__()

                self.gradient_penalty_weight = gradient_penalty_weight  # interpolated image loss weight

                # Generator and Critic Optimizers
                self.generator_optimizer = g_opt
                self.critic_optimizer = c_opt
                self.mean_squared_error = MeanSquaredError()

                # Build Generator
                self.generator = generator

                # Build Discriminator
                self.critic = critic

                # build vgg form content loss
                self.comparator = Model(self.critic.input, self.critic.layers[cut_layer].output)

        def _gradient_penalty_loss(self, hr, sr):

                """
                Computes Interpolated Loss of WGAN training
                :param real_data: batch of real data to be interpolated
                :param generated_data: batch of generated data to be interpolated
                :return: interpolated loss
                """

                # Get Number of instances of real data == Batch size
                batch_size = hr.shape[0]
                alpha = tf.random.uniform((batch_size, 1, 1, 1, 1))  # alpha shape matches 3d data
                inter_data = (alpha * hr) + ((1 - alpha) * sr)

                with tf.GradientTape() as g:
                        g.watch(inter_data)
                        logits_inter_data = self.critic(inter_data)

                gradients = g.gradient(logits_inter_data, inter_data)

                # compute the euclidean norm by squaring ...
                gradients_sqr = tf.square(gradients)
                #   ... summing over the rows ...
                gradients_sqr_sum = tf.reduce_sum(gradients_sqr, axis=0)
                #   ... and sqrt
                gradient_l2_norm = tf.sqrt(gradients_sqr_sum)
                # compute lambda * (1 - ||grad||)^2 still for each single sample
                gradient_penalty = tf.square(1 - gradient_l2_norm)
                # return the mean as loss over all the batch samples
                return tf.reduce_mean(gradient_penalty)

        @tf.function
        def _content_loss(self, hr, sr):
                # hr = preprocess_input(hr)
                # sr = preprocess_input(sr)
                hr_features = self.comparator(hr)  #/ 12.75
                sr_features = self.comparator(sr)  #/ 12.75
                return self.mean_squared_error(hr_features, sr_features)

        @tf.function
        def train_step(self, lr, hr):

                with tf.GradientTape() as gen_tape, tf.GradientTape() as critic_tape:

                        lr = tf.convert_to_tensor(lr, dtype=tf.float32)
                        hr = tf.convert_to_tensor(hr, dtype=tf.float32)
                        sr = self.generator(lr, training=True)  # feed generator and get super resolution data

                        logits_sr = self.critic(sr, training=True)  # get generator logits from generated data, SR data
                        logits_hr = self.critic(hr, training=True)  # get critic logits from real data, HR data

                        generator_adversarial_loss = tf.reduce_mean(logits_sr)  # SR with label 1
                        content_loss = self._content_loss(hr, sr)
                        perception_loss = content_loss + 0.001 * generator_adversarial_loss  # Generator loss

                        critic_regularizer_loss = self._gradient_penalty_loss(hr, sr)  # gradient penalty
                        critic_loss = (
                                        tf.reduce_mean(logits_hr)
                                        - generator_adversarial_loss
                                        + critic_regularizer_loss
                                        * self.gradient_penalty_weight
                        )

                gen_gradients = gen_tape.gradient(
                        perception_loss,
                        self.generator.trainable_variables)

                critic_gradients = critic_tape.gradient(
                        critic_loss,
                        self.critic.trainable_variables)

                self.generator_optimizer.apply_gradients(
                        zip(gen_gradients, self.generator.trainable_variables))
                self.critic_optimizer.apply_gradients(
                        zip(critic_gradients, self.critic.trainable_variables))

                return perception_loss, critic_loss


if __name__ == '__main__':

        # import numpy as np
        # import matplotlib.pyplot as plt
        #
        # img = np.random.uniform(0, 1, size=(64, 64, 64))
        # # plt.imshow(img)
        # # plt.show()
        #
        # mod = _vgg(5)
        # mod.summary()
        #
        # batch = np.expand_dims(img, 0)
        #
        # res = mod.predict_on_batch(batch)

        # Initialize NN weights
        init = tf.initializers.RandomNormal(stddev=0.02)
        st_filters = 16

        # Create generator Graph
        generator = Sequential(
                [       # Input
                        Input(shape=(24, 32, 24, 1), name='z_input'),
                        # Dense(4 * 4 * 4 * 128),
                        # LeakyReLU(alpha=0.2, name='lrelu_1'),
                        # Reshape((4, 4, 4, 128), name="conv_1"),
                        # 1st Deconvolution
                        Conv3DTranspose(
                                filters=st_filters*4,
                                kernel_size=5,
                                strides=(2, 2, 2),
                                kernel_initializer=init,
                                use_bias=True,
                                padding="same",
                                name="conv_1"),
                        LeakyReLU(alpha=0.2, name='lrelu_1'),
                        # 2nd Deconvolution
                        Conv3DTranspose(
                                filters=st_filters*2,
                                kernel_size=5,
                                strides=(2, 2, 2),
                                kernel_initializer=init,
                                use_bias=True,
                                padding='same',
                                name="conv_2"),
                        LeakyReLU(alpha=0.2, name='lrelu_2'),
                        # 3rd Deconvolution
                        Conv3DTranspose(
                                filters=st_filters,
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
                                strides=(1, 1, 1),
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
                        Input(shape=(192, 256, 192, 1), name='input'),
                        # 1st Convolution
                        Conv3D(
                                filters=st_filters,
                                kernel_size=5,
                                strides=(2, 2, 2),
                                kernel_initializer=init,
                                use_bias=True,
                                padding="same",
                                name="conv_1"),
                        LeakyReLU(alpha=0.2, name='lrelu_1'),
                        # 2nd Convolution
                        Conv3D(
                                filters=st_filters*2,
                                kernel_size=5,
                                strides=(2, 2, 2),
                                kernel_initializer=init,
                                use_bias=True,
                                padding='same',
                                name="conv_2"),
                        LeakyReLU(alpha=0.2, name='lrelu_2'),
                        # 3rd Convolution
                        Conv3D(
                                filters=st_filters*4,
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

        # Create adversarial graph
        gen_opt = Adam(1e-5)
        critic_opt = Adam(1e-5)
        wgan = SRWGAN(generator=generator, critic=critic, cut_layer=3, g_opt=gen_opt, c_opt=critic_opt)

        # Path to mnc files
        IMAGES_LOW_PATH = "D:\\Matesanz\\Imágenes\\training\\low"
        IMAGES_HIGH_PATH = "D:\\Matesanz\\Imágenes\\training\\high"
        datagen_low = DataGenerator(IMAGES_LOW_PATH, load)  # Collects LR images for training
        datagen_high = DataGenerator(IMAGES_HIGH_PATH, load)  # Collects HR images for training

        folder_path = path.join('weights', 'dc_wgan_sr')
        generator_name = 'generator_dc_wgan_sr2.h5'
        critic_name = 'critic_dc_wgan_sr2.h5'
        generator_path = path.join(folder_path, generator_name)
        critic_path = path.join(folder_path, critic_name)

        generator.load_weights(generator_path)
        critic.load_weights(critic_path)
        # --------------------
        #  PARAMETER INIT
        # --------------------

        batch_size = 1  # Samples every epoch
        n_epochs = 100  # Training Epochs
        plot_interval = 10  # Every plot_interval create a graph with real and generated data distribution
        c_loops = 5  # number of loops to train critic every epoch
        # z_control = tf.random.normal((batch_size, wgan.z_units))  # Vector to feed gen and control training evolution

        # --------------------
        #  TENSORBOARD SETUP
        # --------------------

        # Set Tensorboard Directory to track data
        time_now = strftime("%d-%b-%H%M", localtime())
        log_dir = path.join('..', 'logs', 'dc_wgan_sr', time_now)

        # Start model data tracing (logs)
        summary_writer = tf.summary.create_file_writer(log_dir)
        tf.summary.trace_on()

        generator_train_loss = tf.keras.metrics.Mean('generator_train_loss', dtype=tf.float32)
        critic_train_loss = tf.keras.metrics.Mean('critic_train_loss', dtype=tf.float32)

        for epoch in range(n_epochs):

                start_time = time()


                # --------------------
                #     TRAINING
                # --------------------

                batch_lr = datagen_low.get_batch(batch_size)
                batch_hr = datagen_high.get_batch(batch_size)

                b_time = time()

                print("BATCH took {} seconds".format(round(b_time - start_time, 2)))

                g_loss, c_loss = wgan.train_step(batch_lr, batch_hr)

                t_time = time()

                print("TRAIN took {} seconds".format(round(t_time - b_time, 2)))


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


                # -----------------------
                #  TENSORBOARD PLOTTING
                # ------------------------

                with summary_writer.as_default():

                        # Write losses
                        tf.summary.scalar('Generator Loss',
                                          generator_train_loss.result(),
                                          step=epoch)

                        tf.summary.scalar('Discriminator Loss',
                                          critic_train_loss.result(),
                                          step=epoch)

                print("Epoch took {} seconds".format(round(time() - start_time, 2)))

        # ---------------
        #  SAVE WEIGHTS
        # ---------------


        generator.save(generator_path)
        critic.save(critic_path)

        # generate fake sample to visualize
        img_path = IMAGES_LOW_PATH + "\\brainweb  (1).npy"
        low = load(img_path)
        low = expand_dims(low, 0)
        fake = generator(low)[0]
        fake = squeeze(fake, 3)
        print("Min and Max values of fake image are:", fake.min(), fake.max())
        Viewer(fake)