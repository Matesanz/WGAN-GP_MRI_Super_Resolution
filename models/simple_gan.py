from os import path
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, LeakyReLU, BatchNormalization
from tensorflow.keras.optimizers import Adam
from utils.simple_data_generator import get_quadratic_data_sample, save_plot_to_buffer, plot_quadratic_data
from utils.training import save_models
import matplotlib.pyplot as plt
from time import strftime, localtime


def build_simple_nn(input_units, output_units, activation, layer_number=4, units_per_layer=16, model_name='nn'):

        """
        Build simple n layers fully connected neural network
        :param input_units: int, number of input units
        :param output_units: int, number of output units
        :param activation: str, activation function
        :return: Keras Model
        """

        # Input Layer
        layer_name = model_name + '_input_layer'
        nnin = Input(shape=(input_units,), name=layer_name)

        # First Hidden Layer
        layer_name = model_name + '_hidden_layer_' + str(1)
        nn = Dense(units=units_per_layer, name=layer_name)(nnin)
        nn = BatchNormalization()(nn)
        nn = LeakyReLU(alpha=0.1)(nn)

        for i in range(layer_number-1):

                # New Hidden Layer
                layer_name = model_name + '_hidden_layer_' + str(i + 2)
                nn = Dense(
                        units=units_per_layer,
                        name=layer_name,
                        kernel_initializer='he_uniform'
                )(nn)
                nn = BatchNormalization()(nn)
                nn = LeakyReLU(alpha=0.1)(nn)

        # Output Layer
        layer_name = model_name + '_output_layer'
        nnout = Dense(
                units=output_units,
                activation=activation,
                name=layer_name,
                kernel_initializer='he_uniform'
        )(nn)

        # Create Model
        model = Model(nnin, nnout, name=model_name)

        # Return Model
        return model


def build_gan(generator, discriminator):

        # Get generator input
        generator_input_dim = generator.input.shape[1]
        # Freeze Discriminator Weights
        discriminator.trainable = False
        # Set Gan Input
        gan_input = Input(
                shape=(generator_input_dim,),
                name='gan_input'
        )

        # Set GAN Body (combine discriminator and generator)
        # Set input on generator is necessary in Keras Model Class
        gan_output = discriminator(generator(gan_input))
        # Create Keras GAN Model
        gan = Model(gan_input, gan_output, name='gan')

        # Return Model
        return gan


def train_step(data, generator, discriminator, d_of_g):

        """
        Performs Batch data on GAN
        :param data: array of 2d coordinates, real data sample
        :param generator: generator model
        :param discriminator: discriminator model
        :param d_of_g: gan model
        :return: generator and discriminator losses
        """

        # Convert data data to tensor
        real_data = tf.convert_to_tensor(data, dtype=tf.float32)
        # Get Number of instances of real data == Batch size
        batch_size = tf.shape(data)[0]
        # Get generator input
        generator_input_dim = generator.input.shape[1]
        # Create z vectors to feed generator
        random_z_vectors = tf.random.normal(shape=(batch_size, generator_input_dim))
        # Create generated data
        generated_data = generator.predict_on_batch(random_z_vectors)
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
        discriminator.trainable = True

        # Train discriminator and get loss for data tracking
        discriminator_loss = discriminator.train_on_batch(combined_data, labels)
        # Freeze discriminator weights to train generator
        # Discriminator will be used only for inference here
        discriminator.trainable = False
        # Create random z vectors to feed generator
        random_z_vectors = tf.random.normal(shape=(batch_size, generator_input_dim))
        # Train generator and get loss for data tracking
        generator_loss = d_of_g.train_on_batch(random_z_vectors, true_labels)

        # Return losses to track data
        return discriminator_loss, generator_loss


if __name__ == '__main__':

        # Set Parameters for data

        epochs = 1001 # Training Epochs
        z_dim = 10  # Generator Input units
        layers = 4  # Number of hidden layers
        g_out_dim = 2  # Generator Output Units == Discriminator Input Units
        batch_size = 100  # Define Batch Size
        plot_interval = 100  # Every plot_interval create a graph with real and generated data distribution

        # Build Generator
        generator = build_simple_nn(
                input_units=z_dim,
                output_units=g_out_dim,
                layer_number=layers,
                activation='linear',
                model_name='generator'
        )
        
        # Build Discriminator
        discriminator = build_simple_nn(
                input_units=g_out_dim,
                output_units=1,
                layer_number=layers,
                activation='sigmoid',
                model_name='discriminator'
        )

        # Compile Discriminator
        discriminator.compile(
                optimizer=Adam(),
                loss='binary_crossentropy')

        # Create Keras GAN Model
        gan = build_gan(generator, discriminator)

        # Compile GAN Model
        gan.compile(
                optimizer=Adam(),
                loss='binary_crossentropy')

        # Set Parameters for data tracking and data evaluation
        # Random vectors to control generator evolution
        control_z_vectors = tf.random.normal(shape=(batch_size, 10))
        # Returns quadratic distributed data, with and without noise
        # Take not noisy data
        real_distribution = get_quadratic_data_sample(batch_size, add_noise=False)

        # Set Tensorboard Directory to track data
        time = strftime("%d-%b-%H%M", localtime())
        log_dir = path.join('..', 'logs', 'simple_gan', time)

        # Initialize Keras metrics to track data
        generator_train_loss = tf.keras.metrics.Mean('generator_train_loss', dtype=tf.float32)
        discriminator_train_loss = tf.keras.metrics.Mean('discriminator_train_loss', dtype=tf.float32)

        # Create Scope in order to set Tensorboard hyperparameters
        with tf.name_scope("Simple Gan Training") as scope:

                # Start model data tracing (logs)
                tf.summary.trace_on()
                summary_writer = tf.summary.create_file_writer(log_dir)
                # START TRAINING
                for e in range(epochs):
                        # Collect data
                        real_data = get_quadratic_data_sample(batch_size)
                        # Train gan: Perform batch data with the collected data
                        d_loss, g_loss = train_step(real_data, generator, discriminator, gan)
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
                                print(template.format(e+1,
                                                      generator_train_loss.result(),
                                                      discriminator_train_loss.result()))

                        # Write data into Tensorboard
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

        # Stop data tracing
        tf.summary.trace_off()

        # Save Models
        save_models(discriminator, generator, gan, 'simple_gan')
