from os import path
import nibabel as nib
from time import strftime, localtime
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Conv3DTranspose, Reshape, LeakyReLU, Flatten, Conv3D
from tensorflow.keras.optimizers import Adam
from utils.simple_data_generator import get_quadratic_data_sample, plot_quadratic_data
import numpy as np
from utils.imutils import pad_image


def build_critic(ncf=32, kernel_size=5, layer_n=3, activation=None, input_dims=(192, 256, 192, 1)):

        # TODO: CLEAN DISCRIMINATOR CODE

        # Input Layer
        model_name = 'critic'
        layer_name = model_name + '_input'
        nnin = Input(shape=input_dims, name=layer_name)

        # init model weights
        init = tf.initializers.RandomNormal(stddev=0.02)

        # First Hidden Layer
        layer_name = model_name + '_conv_' + str(1)
        nn = Conv3D(filters=ncf,
                    kernel_size=kernel_size,
                    strides=(2, 2, 2),
                    padding='same',
                    use_bias=True,
                    kernel_initializer=init,
                    name=layer_name)(nnin)

        nn = LeakyReLU(alpha=0.1)(nn)

        for i in range(layer_n - 1):
                # New Hidden Layer
                layer_name = model_name + '_conv_' + str(i + 2)
                filter_n = ncf * 2 ** (i + 1)
                nn = Conv3D(filters=filter_n,
                            kernel_size=kernel_size,
                            strides=(2, 2, 2),
                            padding='same',
                            use_bias=True,
                            kernel_initializer=init,
                            name=layer_name)(nn)

                nn = LeakyReLU(alpha=0.1)(nn)

        # Flatten
        nn = Flatten()(nn)

        # Output Layer
        layer_name = model_name + '_output_layer'
        nnout = Dense(
                units=1,
                activation=activation,
                name=layer_name,
                kernel_initializer=init
        )(nn)

        # Create Model
        model = Model(nnin, nnout, name=model_name)

        # Return Model
        return model


def build_generator(ngf=32, kernel_size=5, layer_n=3, activation="linear", z_dims=(10,)):

        # TODO: CLEAN GENERATOR CODE

        # Input Layer
        model_name = 'generator'
        layer_name = model_name + '_z_input'
        nnin = Input(shape=z_dims, name=layer_name)

        # init model weights
        init = tf.initializers.RandomNormal(stddev=0.02)

        nn = Dense(4*4*4*ngf*(2**(layer_n)), kernel_initializer=init, use_bias=True)(nnin)
        nn = Reshape((4, 4, 4, ngf*(2**layer_n)))(nn)
        nn = LeakyReLU(alpha=0.1)(nn)

        # Convolutional Layers
        for i in range(layer_n - 1):
                # New Hidden Layer
                layer_name = model_name + '_conv_' + str(i + 1)
                filter_n = ngf * 2 ** (layer_n - i - 1)
                nn = Conv3DTranspose(filters=filter_n,
                                     kernel_size=kernel_size,
                                     strides=(4, 4, 4),
                                     padding='same',
                                     use_bias=True,
                                     kernel_initializer=init,
                                     name=layer_name)(nn)

                nn = LeakyReLU(alpha=0.1)(nn)

        # Output Layer
        layer_name = model_name + '_output_layer'
        nnout = Conv3DTranspose(
                filters=1,
                kernel_size=kernel_size,
                strides=(3, 4, 3),
                padding='same',
                activation=activation,
                use_bias=True,
                kernel_initializer=init,
                name=layer_name)(nn)

        # Create Model
        model = Model(nnin, nnout, name=model_name)

        # Return Model
        return model


class DCWGAN(Model):

        def __init__(self, generator, critic, g_lr=5e-5, c_lr=5e-5, gradient_penalty_weight = 10):
                super(DCWGAN, self).__init__()

                # Set Parameters for training

                self.z_units = generator.layers[0].output_shape[0][1]
                self.g_lr = g_lr  # generator learning rate
                self.c_lr = c_lr  # discriminator learning rate
                self.gradient_penalty_weight = gradient_penalty_weight  # interpolated image loss weight
                # critic_loops = 5  # number of iterations to train discriminator on every epoch

                self.generator_optimizer = Adam(lr=self.g_lr, beta_1=0.5)
                self.critic_optimizer = Adam(lr=self.c_lr, beta_1=0.5)

                # Build Generator
                self.generator = generator

                # Build Discriminator
                self.critic = critic

        def set_trainable(self, model, val):
                model.trainable = val
                for layer in model.layers:
                        layer.trainable = val

        def gradient_penalty_loss(self, real_data, generated_data):

                # Get Number of instances of real data == Batch size
                batch_size = real_data.shape[0]
                alpha = tf.random.uniform((batch_size, 1))
                inter_data = (alpha * real_data) + ((1 - alpha) * generated_data)

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

        def generate_data(self, n):

                # Create random z vectors to feed generator
                random_z_vectors = tf.random.normal(shape=(n, self.z_units))
                generated_data = self.generator(random_z_vectors)
                return generated_data

        def compute_generator_loss(self, batch_size):

                # Get fake data to feed generator
                generated_data = self.generate_data(batch_size)
                # feed generator and get logits
                logits_generated_data = self.critic(generated_data)
                # losses of fake with label "1"
                generator_loss = tf.reduce_mean(logits_generated_data)

                return generator_loss

        def compute_generator_gradients(self, batch_size):

                with tf.GradientTape() as gen_tape:
                        gen_loss = self.compute_generator_loss(batch_size)
                # compute gradients
                gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
                return gen_gradients

        def apply_generator_gradients(self, gradients):

                """
                Apply calculated gradients to update generator
                :param gradients: generator gradients
                """
                # Optimizer applies gradients on trainable weights
                self.generator_optimizer.apply_gradients(
                        zip(gradients, self.generator.trainable_variables)
                )

        def compute_critic_loss(self, real_data):

                """
                passes through the network and computes loss
                """

                # Get Number of instances of real data == Batch size
                batch_size = real_data.shape[0]

                # Convert numpy training data to tensor
                real_data = tf.convert_to_tensor(real_data, dtype=tf.float32)

                # Create random z vectors to feed generator
                random_z_vectors = tf.random.normal(shape=(batch_size, self.z_units))
                generated_data = self.generator(random_z_vectors)

                # discriminate x and x_gen
                logits_real_data = self.critic(real_data)
                logits_generated_data = self.critic(generated_data)

                # gradient penalty
                critic_regularizer = self.gradient_penalty_loss(real_data, generated_data)

                # losses
                critic_loss = (
                        tf.reduce_mean(logits_real_data)
                        - tf.reduce_mean(logits_generated_data)
                        + critic_regularizer
                        * self.gradient_penalty_weight
                )

                return critic_loss

        def compute_critic_gradients(self, real_data):

                """
                Compute Gradients to update generator and discriminator
                :param real_data:
                :return:
                """

                # pass through network
                with tf.GradientTape() as critic_tape:
                        critic_loss = self.compute_critic_loss(real_data)

                # compute gradients
                critic_gradients = critic_tape.gradient(critic_loss, self.critic.trainable_variables)

                return critic_gradients

        def apply_critic_gradients(self, gradients):

                """
                Apply calculated gradients to update critic
                :param gradients: critic gradients
                """

                # Optimizer applies gradients on trainable weights
                self.critic_optimizer.apply_gradients(
                        zip(gradients, self.critic.trainable_variables)
                )

        @tf.function
        def train_generator(self, batch_size):
                gen_gradients = self.compute_generator_gradients(batch_size)
                self.apply_generator_gradients(gen_gradients)

        @tf.function
        def train_critic(self, real_data):
                critic_gradients = self.compute_critic_gradients(real_data)
                self.apply_critic_gradients(critic_gradients)


if __name__ == '__main__':

        gen = build_generator()
        critic = build_critic()

        wgan = DCWGAN(generator=gen, critic=critic)

        IMAGE_FILE = 'brain.mnc'
        IMAGE_PATH = path.join('..', 'resources', 'mri', IMAGE_FILE)

        # --------------------
        #  PARAMETER INIT
        # --------------------

        batch_size = 1  # Samples every epoch
        n_epochs = 10  # Training Epochs
        plot_interval = 10  # Every plot_interval create a graph with real and generated data distribution
        c_loops = 5  # number of loops to train critic every epoch

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
                        img = nib.load(IMAGE_PATH)
                        data = img.get_fdata()
                        data = pad_image(data)
                        data = data.reshape((192, 256, 192, 1))
                        training_data = np.expand_dims(data, 0)
                        training_data = training_data.astype(np.float32)

                        wgan.train_critic(training_data)

                # Train Generator
                wgan.train_generator(batch_size)  # Train our model on real distribution points
                c_loss = wgan.compute_critic_loss(training_data)  # Get batch loss to track training
                g_loss = wgan.compute_generator_loss(batch_size)

                # -----------------------
                #  TENSORBOARD TRACKING
                # ------------------------

                # Save generator and critic losses
                generator_train_loss(g_loss)
                critic_train_loss(c_loss)

                # track training through console
                template = 'Epoch {}, Gen Loss: {}, Dis Loss {}'
                print(template.format(epoch + 1,
                                      generator_train_loss.result(),
                                      critic_train_loss.result()))
