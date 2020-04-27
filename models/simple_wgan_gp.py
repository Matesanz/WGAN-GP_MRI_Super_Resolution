from time import localtime, strftime
from os import path
import tensorflow as tf
from models.simple_gan import build_simple_nn, build_gan
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Layer, Input, Lambda
from functools import partial
from tensorflow.keras.models import Model

import keras.backend as K
import matplotlib.pyplot as plt
from numpy import clip, arange
from utils.save_models import save_models
from utils.simple_data_generator import get_quadratic_data_sample, plot_quadratic_data, save_plot_to_buffer


class RandomWeightedAverage(Layer):
        """
        Taken from David Foster GDL repo: https://github.com/davidADSP/GDL_code/blob/tensorflow_2/models/WGANGP.py
        Provides a (random) weighted average between real and generated image samples
        """

        def __init__(self, batch_size):
                super().__init__()
                self.batch_size = batch_size

        def call(self, inputs, **kwargs):
                alpha = K.random_uniform((self.batch_size, 1))
                return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])


def grad(y, x):
        """
        Taken from David Foster GDL repo: https://github.com/davidADSP/GDL_code/blob/tensorflow_2/models/WGANGP.py
        """

        V = Lambda(lambda z: K.gradients(
                z[0], z[1]), output_shape=[1])([y, x])
        return V


def gradient_penalty_loss(y_true, y_pred, interpolated_samples):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        gradients = grad(y_pred, interpolated_samples)[0]

        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)


def wasserstein_loss(y, y_hat):
        """
        Compute Wasserstein loss
        The loss will normally be less than 0
        :param y: expected output
        :param y_hat: predicted output
        :return: product mean
        """
        return -K.mean(y * y_hat)


def build_adversarial(generator, critic, batch_size):

        """
        Builds critic and generator graph
        :param generator: generator Keras Model
        :param critic: critic Keras Model
        :param batch_size: int, length of batch durin training
        :return: Critic and Generator Models
        """

        # -------------------------------
        # Construct Computational Graph
        #       for the Critic
        # -------------------------------

        # Get generator input
        generator_input_dim = generator.input.shape[1]
        critic_input_dim = critic.input.shape[1]

        # Freeze Critic Weights
        generator.trainable = False

        # Set Z vector Input
        # Generator will take random vectors from this Layer
        z_input = Input(
                shape=(generator_input_dim,),
                name='z_input'
        )

        # Set critic Fake data input
        # critic will take fake data samples from this graph
        fake_input = generator(z_input)

        # Set Critic Real data input
        # Critic will take real data samples from this Layer
        real_input = Input(
                shape=(critic_input_dim),
                name='real_input'
        )

        # Set Critic Interpolated data input
        # Critic will take interpolated data samples from this Layer
        interpolated_input = RandomWeightedAverage(batch_size)([real_input, fake_input])

        # Build Adversarial Graph with real, fake and interpolated Inputs
        real_graph = critic(real_input)
        fake_graph = critic(fake_input)
        interpolated_graph = critic(interpolated_input)

        # Taken from David Foster GDL repo:
        # https://github.com/davidADSP/GDL_code/blob/tensorflow_2/models/WGANGP.py
        # Use Python partial to provide loss function with additional
        # 'interpolated_samples' argument
        partial_gp_loss = partial(gradient_penalty_loss,
                                  interpolated_samples=interpolated_input)
        partial_gp_loss.__name__ = 'gradient_penalty'  # Keras requires function names

        # Build Critic Graph
        critic_graph = Model(
                inputs=[real_input, z_input],
                outputs=[real_graph, fake_graph, interpolated_graph],
                name='Critic Graph'
        )

        # Compile Critic
        critic_graph.compile(
                optimizer=Adam(lr=5e-5, beta_1=0.5),  # Optimizer described in paper
                loss=[wasserstein_loss, wasserstein_loss, partial_gp_loss],  # loss for real, fake and interpol images
                loss_weights=[1, 1, 10]  # Every loss has a weighting value
        )

        # -------------------------------
        # Construct Computational Graph
        #       for the Generator
        # -------------------------------

        # Critic will be used for inference so freeze critic weights
        critic.trainable = False
        generator.trainable = True
        # Set input for generator graph: z vector input
        generator_graph_input = Input(shape=(generator_input_dim,))  # z input
        # Set body for generator graph: D of G
        generator_graph_output = critic(generator(generator_graph_input))  # D of G in Vanilla GAN
        # Build generator Graph
        generator_graph = Model(generator_graph_input, generator_graph_output, name='generator graph')
        # Compile generator Graph
        generator_graph.compile(
                optimizer=Adam(lr=5e-5, beta_1=0.5),
                loss=wasserstein_loss
        )

        # Return Models
        return critic_graph, generator_graph


def generator_train_step(batch_size, generator, generator_graph):
        """
        Perform Batch training step on Generator
        :param batch_size: int, batch length
        :param generator: generator keras model
        :param generator_graph: discriminator(generator(z_input)) keras model
        :return: generator Loss
        """

        # Get generator input
        generator_input_dim = generator.input.shape[1]
        # true_labels = tf.zeros((batch_size, 1))
        true_labels = tf.ones((batch_size, 1))

        # --------------------
        # GENERATOR TRAINING
        # --------------------

        # Create random z vectors to feed generator
        random_z_vectors = tf.random.normal(shape=(batch_size, generator_input_dim))
        # Train generator and get loss for training tracking
        generator_loss = generator_graph.train_on_batch(random_z_vectors, true_labels)

        # Return losses to track training
        return generator_loss


def critic_train_step(data, critic_graph, generator):
        """
        Performs Batch training on Critic
        :param data: array of 2d coordinates, real data sample
        :param critic_graph: gan model
        :param generator: generator model
        :return: critic losses
        """

        # Get Number of instances of real data == Batch size
        batch_size = tf.shape(data)[0]
        # Get length of z vector
        generator_input_dim = generator.input.shape[1]

        # Convert training data to tensor
        real_data = tf.convert_to_tensor(data, dtype=tf.float32)
        # Create z vectors to feed generator # tip: use normal dist, not uniform.
        random_z_vectors = tf.random.normal(shape=(batch_size, generator_input_dim))
        # Create generated data
        generated_data = generator.predict_on_batch(random_z_vectors)

        # Labels for real (ones vector) and fake (minus ones vector) data
        true_labels = tf.ones((batch_size, 1))
        fake_labels = -true_labels
        dummy_labels = tf.zeros((batch_size, 1))


        # Train critic and get loss for training tracking
        # discriminator_loss = discriminator.train_on_batch(combined_data, labels)
        critic_loss = critic_graph.train_on_batch(
                [real_data, generated_data],  # feeding inputs of the graph
                [true_labels, fake_labels, dummy_labels]  # feeding outputs of the graph to calculate loss
        )

        return critic_loss


if __name__ == '__main__':

        # Set Parameters for training

        epochs = 20001  # Training Epochs
        z_dim = 10  # Generator Input units
        layers = 4  # Number of hidden layers
        g_out_dim = 2  # Generator Output Units == Discriminator Input Units
        batch_size = 32  # Define Batch Size
        plot_interval = 100  # Every plot_interval create a graph with real and generated data distribution
        units_per_layer = 16  # units on hidden layers
        g_lr = 5e-5  # generator learning rate
        d_lr = 5e-5  # discriminator learning rate
        d_epochs = 5  # number of iterations to train discriminator on every epoch

        # Build Generator
        generator = build_simple_nn(
                input_units=z_dim,
                output_units=g_out_dim,
                layer_number=3,
                units_per_layer=units_per_layer,
                activation='linear',
                model_name='generator'
        )

        generator.summary()

        # Build Discriminator
        discriminator = build_simple_nn(
                input_units=g_out_dim,
                output_units=1,
                layer_number=layers,
                units_per_layer=units_per_layer,
                activation=None,
                model_name='discriminator'
        )

        discriminator.summary()

        # Compile Discriminator
        discriminator.compile(
                optimizer=Adam(lr=d_lr, beta_1=0.5),
                loss=wasserstein_loss)  # DIFFERENCE WITH SIMPLE_GAN

        # Create Keras GAN Model
        critic_graph, generator_graph = build_adversarial(generator, discriminator, batch_size)

        critic_graph.summary()  # Show GAN Architecture
        generator_graph.summary()  # Show GAN Architecture
