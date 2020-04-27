from time import localtime, strftime
from os import path
import tensorflow as tf
from models.simple_gan import build_simple_nn, build_gan
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from models.simple_wgan import wasserstein_loss, discriminator_train_step, generator_train_step
from utils.save_models import save_models
from utils.simple_data_generator import get_quadratic_data_sample, plot_quadratic_data, save_plot_to_buffer

if __name__ == '__main__':

        # Set Parameters for training

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

        # Set Parameters for training tracking and training evaluation
        # Random vectors to control generator evolution
        control_z_vectors = tf.random.normal(shape=(batch_size, 10))
        # Get quadratic distributed data (-1, 1)
        real_distribution = get_quadratic_data_sample(batch_size, add_noise=False)

        # Set Tensorboard Directory to track training
        time = strftime("%d-%b-%H%M", localtime())
        log_dir = path.join('..', 'logs', 'simple_wgan', time)

        # Initialize Keras metrics to track training
        generator_train_loss = tf.keras.metrics.Mean('generator_train_loss', dtype=tf.float32)
        discriminator_train_loss = tf.keras.metrics.Mean('discriminator_train_loss', dtype=tf.float32)

        # Create Scope in order to set Tensorboard hyperparameters
        with tf.name_scope("Simple WGan Training") as scope:

                # Start model training tracing (logs)
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
                                # Train gan: Perform batch training on D with the collected data
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
                                # track training through console
                                template = 'Epoch {}, Gen Loss: {}, Dis Loss {}'
                                print(template.format(e + 1,
                                                      generator_train_loss.result(),
                                                      discriminator_train_loss.result()))

                        # Write training into Tensorboard
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

        # Stop training tracing
        tf.summary.trace_off()
        save_models(discriminator, generator, gan, 'simple_wgan')
