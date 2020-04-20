import matplotlib.pyplot as plt
from os import path, mkdir
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, LeakyReLU, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from utils.simple_data_generator import get_quadratic_data_sample, save_plot_to_buffer, plot_simple_data
from time import strftime, localtime


class SimpleGan:
        """
        GAN for 1D Data: it tries to approximate simple quadratic data distribution
        Creates Generator and Discriminator. Compiles them.
        train_step(data): performs batch training on GAN
        """

        def __init__(self):
                # Create generator and Discriminator
                self.generator = self.build_generator((10,), 2)
                self.discriminator = self.build_discriminator((2,))

                # Compile Discriminator
                self.discriminator.compile(
                        optimizer=Adam(),
                        loss='binary_crossentropy')

                # Create Gan Model: combine generator and discriminator
                self.D_of_G = self.build_gan()
                # Compile GAN model
                self.D_of_G.compile(
                        optimizer=Adam(),
                        loss='binary_crossentropy')

        def build_generator(self, input_shape, output_units):
                """
                Returns generator neural network
                :param input_shape: tuple input of generator
                :param output_units: int number of output units
                :return: Generator Keras Model
                """

                # Input
                gin = Input(
                        shape=input_shape,
                        name='generator_input')

                # Hidden Layers
                g = Dense(units=16, kernel_initializer='he_uniform')(gin)
                g = BatchNormalization()(g)
                g = LeakyReLU(alpha=0.1)(g)
                g = Dense(units=16, kernel_initializer='he_uniform')(g)
                g = BatchNormalization()(g)
                g = LeakyReLU(alpha=0.1)(g)

                # Output
                gout = Dense(
                        units=output_units,
                        name='generator_output',
                        activation='linear'
                )(g)
                # Create Keras Model
                model = Model(gin, gout, name='generator')
                # Return Model
                return model

        def build_discriminator(self, input_shape):
                """
                Returns Discriminator NN
                :param input_shape: tuple shape generator input
                :return: Discriminator Keras Model
                """

                # Input
                din = Input(
                        shape=input_shape,
                        name='discriminator_input')
                # Hidden Layers
                d = Dense(units=16, kernel_initializer='he_uniform')(din)
                d = LeakyReLU()(d)
                d = BatchNormalization()(d)
                d = Dense(units=16, kernel_initializer='he_uniform')(d)
                d = BatchNormalization()(d)

                d = LeakyReLU()(d)

                # Output
                dout = Dense(
                        units=1,
                        activation='sigmoid',
                        name='discriminator_output')(d)

                # Create Model
                model = Model(din, dout, name='discriminator')
                # Return Model
                return model

        def build_gan(self):

                # Freeze Discriminator Weights
                self.discriminator.trainable = False
                # Set Gan Input
                gan_input = Input(shape=(10,), name='gan_input')
                # Set Gan Body (combine discriminator and generator)
                # Set input on generator is necessary in Keras Model Class
                gan_output = self.discriminator(self.generator(gan_input))
                # Create Keras GAN Model
                gan = Model(gan_input, gan_output)
                # Return Model
                return gan

        def train_step(self, data):

                # Convert training data to tensor
                real_data = tf.convert_to_tensor(data, dtype=tf.float32)
                # Get Number of instances of real data == Batch size
                batch_size = tf.shape(data)[0]
                # Create z vectors to feed generator
                random_z_vectors = tf.random.normal(shape=(batch_size, 10))
                # Create generated data
                generated_data = self.generator.predict_on_batch(random_z_vectors)
                # Combine generated and real data to train on same batch
                combined_data = tf.concat([real_data, generated_data], axis=0)
                # Labels for real (ones vector) and fake (zeros vector) data
                true_labels = tf.ones((batch_size, 1))
                fake_labels = tf.zeros((batch_size, 1))

                # F Chollet Trick - Important!
                # Adding noise to labels increases model generalization
                true_labels += 0.05 * tf.random.uniform(tf.shape(true_labels))
                fake_labels += 0.05 * tf.random.uniform(tf.shape(fake_labels))

                # Combine labels for real and fake data
                labels = tf.concat([true_labels,
                                    fake_labels], axis=0)

                # START BATCH TRAINING
                # Unfreeze discriminator weights
                self.discriminator.trainable = True

                # Train discriminator and get loss for training tracking
                discriminator_loss = self.discriminator.train_on_batch(combined_data, labels)
                # Freeze discriminator weights to train generator
                # Discriminator will be used only for inference here
                self.discriminator.trainable = False
                # Create random z vectors to feed generator
                random_z_vectors = tf.random.normal(shape=(batch_size, 10))
                # Train generator and get loss for training tracking
                generator_loss = self.D_of_G.train_on_batch(random_z_vectors, true_labels)

                # Return losses to track training
                return discriminator_loss, generator_loss


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
                                control_generated_data_points_plot = plot_simple_data(
                                        control_generated_data_points,
                                        plot_type='points')
                                # Plot real distribution
                                real_distribution_plot = plot_simple_data(real_distribution, plot_type='line')
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
