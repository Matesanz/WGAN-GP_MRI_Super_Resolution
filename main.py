from os import path
from numpy import squeeze
from time import strftime, localtime, time
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from utils.training import DataGenerator
from utils.imutils import process_mnc_and_reduce
from utils.minc_viewer import Viewer
from models.wgan_3d_low import critic, generator
from models.wgan_3d import DCWGAN


if __name__ == '__main__':
        
        # load weights
        critic.load_weights("models/weights/dc_wgan_low/critic_dc_wgan_low.h5")
        generator.load_weights("models/weights/dc_wgan_low/generator_dc_wgan_low.h5")
        
        # Create adversarial graph
        gen_opt = Adam()
        critic_opt = Adam()
        wgan = DCWGAN(generator=generator, critic=critic, g_opt=gen_opt, c_opt=critic_opt)

        # Path to mnc files
        IMAGES_PATH = path.join('resources', 'mri')
        data_generator = DataGenerator(IMAGES_PATH, process_mnc_and_reduce)

        # --------------------
        #  PARAMETER INIT
        # --------------------

        batch_size = 4  # Samples every epoch
        n_epochs = 1000  # Training Epochs
        plot_interval = 100  # Every plot_interval create a graph with real and generated data distribution
        c_loops = 5  # number of loops to train critic every epoch
        z_control = tf.random.normal((batch_size, wgan.z_units))  # Vector to feed gen and control training evolution

        # --------------------
        #  TENSORBOARD SETUP
        # --------------------

        generator_train_loss = tf.keras.metrics.Mean('generator_train_loss', dtype=tf.float32)
        critic_train_loss = tf.keras.metrics.Mean('critic_train_loss', dtype=tf.float32)

        # Set Tensorboard Directory to track data
        time_now = strftime("%d-%b-%H%M", localtime())
        log_dir = path.join('logs', 'dc_wgan_low', time_now)
        # Start model data tracing (logs)
        summary_writer = tf.summary.create_file_writer(log_dir)
        tf.summary.trace_on()

        for epoch in range(n_epochs):

                start_time = time()

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

        # save models after training
        folder_path = path.join('models', 'weights', 'dc_wgan_low')
        generator_name = 'generator_dc_wgan_low.h5'
        critic_name = 'critic_dc_wgan_low.h5'
        generator_path = path.join(folder_path, generator_name)
        critic_path = path.join(folder_path, critic_name)
        generator.save(generator_path)
        critic.save(critic_path)

        # generate fake sample to visualize
        fake = generator(z_control)[0]
        fake = squeeze(fake, 3)
        print("Min and Max values of fake image are:", fake.min(), fake.max())
        Viewer(fake)
