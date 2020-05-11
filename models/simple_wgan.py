from time import localtime, strftime
from os import path
import tensorflow as tf
from models.simple_gan import build_simple_nn, build_gan
from tensorflow.keras.optimizers import Adam
import keras.backend as K
import matplotlib.pyplot as plt
from numpy import clip
from utils.save_models import save_models
from utils.simple_data_generator import get_quadratic_data_sample, plot_quadratic_data, save_plot_to_buffer


def wasserstein_loss(y, y_hat):
        """
        Compute Wasserstein loss
        The loss will normally be less than 0
        :param y: expected output
        :param y_hat: predicted output
        :return: product mean
        """
        return K.mean(y * y_hat)


def generator_train_step(batch_size, generator, d_of_g):

        """
        Perform Batch data step on Generator
        :param batch_size: int, batch length
        :param generator: generator keras model
        :param d_of_g: discriminator(generator) keras model
        :return: generator Loss
        """

        # Get generator input
        generator_input_dim = generator.input.shape[1]
        # true_labels = tf.zeros((batch_size, 1))
        true_labels = -tf.ones((batch_size, 1))

        # --------------------
        # GENERATOR TRAINING
        # --------------------

        # Create random z vectors to feed generator
        random_z_vectors = tf.random.normal(shape=(batch_size, generator_input_dim))
        # Train generator and get loss for data tracking
        generator_loss = d_of_g.train_on_batch(random_z_vectors, true_labels)

        # Return losses to track data
        return generator_loss


def discriminator_train_step(data, generator, discriminator, clip_value=0.01):

        """
        Performs Batch data on Discriminator
        :param data: array of 2d coordinates, real data sample
        :param generator: generator model
        :param discriminator: discriminator model
        :param d_of_g: gan model
        :param clip_value: float, limit weights values on discrimintor weights
        :param d_epochs: int, number of iters to train discriminator on every epoch
        :return: generator and discriminator losses
        """

        # Convert data data to tensor
        real_data = tf.convert_to_tensor(data, dtype=tf.float32)
        # Get Number of instances of real data == Batch size
        batch_size = tf.shape(data)[0]
        # Get generator input
        generator_input_dim = generator.input.shape[1]
        # Create z vectors to feed generator # tip: use normal dist, not uniform.
        random_z_vectors = tf.random.normal(shape=(batch_size, generator_input_dim))
        # Create generated data
        generated_data = generator.predict_on_batch(random_z_vectors)
        # Combine generated and real data to train on same batch
        combined_data = tf.concat([real_data, generated_data], axis=0)
        # Labels for real (ones vector) and fake (minus ones vector) data
        fake_labels = tf.ones((batch_size, 1))
        true_labels = -fake_labels

        # --------------------
        # IMPORTANT: Clip critic weights to satisfy Lipschitz condition
        # --------------------
        for layer in discriminator.layers:
                weights = layer.get_weights()
                weights = [clip(w, -clip_value, clip_value) for w in weights]
                layer.set_weights(weights)

        # Train discriminator and get loss for data tracking
        # discriminator_loss = discriminator.train_on_batch(combined_data, labels)
        true_discriminator_loss = discriminator.train_on_batch(real_data, true_labels)
        fake_discriminator_loss = discriminator.train_on_batch(generated_data, fake_labels)
        discriminator_loss = -true_discriminator_loss + fake_discriminator_loss

        return discriminator_loss


if __name__ == '__main__':

        # Set Parameters for data

        epochs = 20001  # Training Epochs
        z_dim = 10  # Generator Input units
        layers = 4  # Number of hidden layers
        g_out_dim = 2  # Generator Output Units == Discriminator Input Units
        batch_size = 64  # Define Batch Size
        plot_interval = 100  # Every plot_interval create a graph with real and generated data distribution
        units_per_layer = 32  # units on hidden layers
        g_lr = 5e-5  # generator learning rate
        d_lr = 5e-5  # discriminator learning rate
        d_epochs = 5  # number of iterations to train discriminator on every epoch

        # Build Generator
        generator = build_simple_nn(
                input_units=z_dim,
                output_units=g_out_dim,
                layer_number=layers,
                units_per_layer=units_per_layer,
                activation='linear',
                model_name='generator'
        )

        # Build Discriminator
        discriminator = build_simple_nn(
                input_units=g_out_dim,
                output_units=1,
                layer_number=layers,
                units_per_layer=units_per_layer,
                activation='linear',
                model_name='discriminator'
        )

        # Compile Discriminator
        discriminator.compile(
                optimizer=Adam(lr=d_lr, beta_1=0.5),
                loss=wasserstein_loss)  # DIFFERENCE WITH SIMPLE_GAN

        # Create Keras GAN Model
        gan = build_gan(generator, discriminator)

        # Compile GAN Model
        gan.compile(
                optimizer=Adam(lr=g_lr, beta_1=0.5),
                loss=wasserstein_loss)  # DIFFERENCE WITH SIMPLE_GAN

        gan.summary()  # Show GAN Architecture

        # Set Parameters for data tracking and data evaluation
        # Random vectors to control generator evolution
        control_z_vectors = tf.random.normal(shape=(batch_size, 10))
        # Get quadratic distributed data (-1, 1)
        real_distribution = get_quadratic_data_sample(batch_size, add_noise=False)

        # Set Tensorboard Directory to track data
        time = strftime("%d-%b-%H%M", localtime())
        log_dir = path.join('..', 'logs', 'simple_wgan', time)

        # Initialize Keras metrics to track data
        generator_train_loss = tf.keras.metrics.Mean('generator_train_loss', dtype=tf.float32)
        discriminator_train_loss = tf.keras.metrics.Mean('discriminator_train_loss', dtype=tf.float32)

        # Create Scope in order to set Tensorboard hyperparameters
        with tf.name_scope("Simple WGan Training") as scope:

                # Start model data tracing (logs)
                tf.summary.trace_on()
                summary_writer = tf.summary.create_file_writer(log_dir)

                # --------------------
                # START BATCH TRAINING
                # --------------------
                for e in range(epochs):

                        # --------------------
                        # DISCRIMINATOR TRAINING
                        # --------------------

                        # Unfreeze discriminator weights
                        discriminator.trainable = True
                        # Train discriminator more than generator
                        for _ in range(d_epochs):
                                # Collect data
                                real_data = get_quadratic_data_sample(batch_size)
                                # Train gan: Perform batch data on D with the collected data
                                d_loss = discriminator_train_step(real_data, generator, discriminator, clip_value=0.2)

                        # --------------------
                        # GENERATOR TRAINING
                        # --------------------
                        # Freeze discriminator weights to train generator
                        # Discriminator will be used only for inference here
                        discriminator.trainable = False
                        g_loss = generator_train_step(batch_size, generator, gan)

                        # --------------------
                        # TRACK TRAINING
                        # Create an image comparing Real Data Distribution and (quadratic distribution)
                        # and generator Distribution (Distribution we want to approximate)
                        # --------------------

                        # Save generator and discriminator losses
                        generator_train_loss(g_loss)
                        discriminator_train_loss(d_loss)

                        # Visual tracking: Every n epochs create a plot comparing real and fake data distribution
                        if e % plot_interval == 0:
                                # Create Matplotlib figure
                                fig = plt.figure()
                                # Points generated by generator with control vectors
                                control_generated_data_points = generator.predict_on_batch(control_z_vectors)
                                # Plot Points generated by generator with control vectors
                                plot_quadratic_data(control_generated_data_points, show=False)
                                # Plot real distribution
                                plot_quadratic_data(real_distribution, show=False)
                                # Save plot image to buffer (no file creation needed)
                                buf = save_plot_to_buffer()
                                # Close plt figure
                                plt.close(fig)
                                # Convert PNG buffer to TensorBoard image
                                image = tf.image.decode_png(buf.getvalue(), channels=4)
                                # Add the batch dimension
                                image = tf.expand_dims(image, 0)
                                # track data through console
                                template = 'Epoch {}, Gen Loss: {}, Dis Loss {}'
                                print(template.format(e + 1,
                                                      generator_train_loss.result(),
                                                      discriminator_train_loss.result()))

                        # Write data into Tensorboard
                        with summary_writer.as_default():
                                # Write losses
                                tf.summary.scalar('Generator Loss',
                                                  g_loss,
                                                  # generator_train_loss.result(),
                                                  step=e)

                                tf.summary.scalar('Discriminator Loss',
                                                  d_loss,
                                                  # discriminator_train_loss.result(),
                                                  step=e)

                                # Write image of real and generated data distribution into Tensorboard
                                if e % plot_interval == 0: tf.summary.image('Comparison', image, step=e)

        # Stop data tracing
        tf.summary.trace_off()
        save_models(discriminator, generator, gan, 'simple_wgan')
