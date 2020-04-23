from os import path
import tensorflow as tf
from models.simple_gan import SimpleGan
import matplotlib.pyplot as plt
from utils.simple_data_generator import get_quadratic_data_sample, save_plot_to_buffer, plot_quadratic_data
from time import strftime, localtime

if __name__ == '__main__':

        # Set Parameters for training
        # Training Epochs
        epochs = 1001
        # create gan model
        gan = SimpleGan()
        # Define Batch Size
        batch_size = 100
        # Every plot_epoch create a graph with real and generated data distribution
        plot_interval = 100

        # Set Parameters for training tracking and training evaluation
        # Random vectors to control generator evolution
        control_z_vectors = tf.random.normal(shape=(batch_size, 10))
        # Returns quadratic distributed data, with and without noise
        # Take not noisy data
        _, real_distribution = get_quadratic_data_sample(batch_size, x_min=-1, x_max=1)
        # Set Tensorboard Directory to track training
        time = strftime("%d-%b-%H%M", localtime())
        log_dir = path.join('..', 'logs', 'simple_gan', time)
        # generator_log_dir = path.join(log_dir, 'generator')
        # discriminator_log_dir = path.join(log_dir, 'discriminator')
        # Initialize Keras metrics to track training
        generator_train_loss = tf.keras.metrics.Mean('generator_train_loss', dtype=tf.float32)
        discriminator_train_loss = tf.keras.metrics.Mean('discriminator_train_loss', dtype=tf.float32)

        # Create Scope in order to set Tensorboard hyperparameters
        with tf.name_scope("Simple Gan Training") as scope:

                # Start model training tracing (logs)
                tf.summary.trace_on()
                summary_writer = tf.summary.create_file_writer(log_dir)
                # START TRAINING
                for e in range(epochs):
                        # Collect data
                        _, real_data = get_quadratic_data_sample(batch_size, x_min=-1, x_max=1)
                        # Train gan: Perform batch training with the collected data
                        d_loss, g_loss = gan.train_step(real_data)
                        # Save generator and discriminator losses
                        generator_train_loss(g_loss)
                        discriminator_train_loss(d_loss)

                        # TRACK TRAINING
                        # Create an image comparing Real Data Distribution and (quadratic distribution)
                        # and generator Distribution (Distribution we want to approximate)
                        # Visual tracking: Every n epochs create a plot comparing real and fake data distribution
                        if e % plot_interval == 0:
                                # Create Matplotlib figure
                                fig = plt.figure()
                                # Points generated by generator with control vectors
                                control_generated_data_points = gan.generator.predict_on_batch(control_z_vectors)
                                # Plot Points generated by generator with control vectors
                                control_generated_data_points_plot = plot_quadratic_data(
                                        control_generated_data_points,
                                        plot_type='points')
                                # Plot real distribution
                                real_distribution_plot = plot_quadratic_data(real_distribution, plot_type='line')
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
                                print(template.format(e+1,
                                                      generator_train_loss.result(),
                                                      discriminator_train_loss.result()))

                        # Write training into Tensorboard
                        with summary_writer.as_default():
                                # Write losses
                                tf.summary.scalar('Generator Loss',
                                                  generator_train_loss.result(),
                                                  step=e)

                                tf.summary.scalar('Discriminator Loss',
                                                  discriminator_train_loss.result(),
                                                  step=e)

                                # Write image of real and generated data distribution into Tensorboard
                                if e % plot_interval == 0: tf.summary.image('Comparison', image, step=e)

        # Stop training tracing
        tf.summary.trace_off()

        # Save Models
        gan_name = 'gan_model_' + time + '.h5'
        generator_name = 'generator_model_' + time + '.h5'
        discriminator_name = 'discriminator_model_' + time + '.h5'
        gan_path = path.join('weights', 'simple_gan', gan_name)
        generator_path = path.join('weights', 'simple_gan', generator_name)
        discriminator_path = path.join('weights', 'simple_gan', discriminator_name)
        gan.D_of_G.save(gan_path)
        gan.generator.save(generator_path)
        gan.discriminator.save(discriminator_path)
