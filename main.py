from models.simple_wgan_gp_II import Simple_WGAN
from utils.simple_data_generator import get_quadratic_data_sample, plot_quadratic_data

if __name__ == '__main__':

        # --------------------
        #  PARAMETER INIT
        # --------------------

        # Training Parameters
        n_epochs = 1000  # Training Epochs
        batch_size = 64  # Samples every epoch
        c_loops = 5  # number of loops to train critic every epoch, 5 according to paper

        # Networks Parameters
        layers_n = 4  # number of generator and critic hidden layers
        hidden_layer_units = 16  # number of generator and critic hidden layers units

        # Build GAN
        wgan = Simple_WGAN(
                layers_n=4,
                hidden_units=hidden_layer_units,
        )

        # --------------------
        #     TRAINING
        # --------------------

        for epoch in range(n_epochs):

                # --------------------
                #     TRAIN CRITIC
                # --------------------

                for _ in range(c_loops):
                        training_data = get_quadratic_data_sample(batch_size)  # Get points from real distribution
                        wgan.train_critic(training_data)

                # ----------------------
                #     TRAIN GENERATOR
                # ----------------------

                wgan.train_generator(batch_size)  # Train our model on real distribution points
                c_loss= wgan.compute_critic_loss(training_data)  # Get critic batch loss to track data
                g_loss = wgan.compute_generator_loss(batch_size)  # Get generator batch loss to track data

                # ----------------------
                #     PRINT TRAINING
                # ----------------------

                template = 'Epoch {}, Gen Loss: {}, Dis Loss {}'
                print(template.format(epoch + 1, g_loss, c_loss))

        # ------------------------------------------------
        #     PLOT DISTRIBUTIONS AT THE END OF TRAINING
        # ------------------------------------------------

        real = get_quadratic_data_sample(batch_size)
        fake = wgan.generate_data(batch_size)
        plot_quadratic_data(real, show=False)
        plot_quadratic_data(fake.numpy())
