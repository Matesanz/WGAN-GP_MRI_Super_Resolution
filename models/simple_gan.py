from keras.models import Model
from keras.layers import Dense, Input
from keras.optimizers import Adam



def get_generator(input_shape, output_units):
        """
        Returns generator neural network
        :param input_shape: tuple input of generator
        :param output_units: int number of output units
        :return: Keras Model
        """

        # Input
        gin = Input(
                shape=input_shape,
                name='generator_input'
        )

        # Hidden Layers
        g = Dense(units=64)(gin)
        g = Dense(units=64)(g)
        g = Dense(units=64)(g)

        # Output
        gout = Dense(
                units=output_units,
                activation='relu',
                name='discriminator_output'
        )(g)

        model = Model(gin, gout)

        return model


def get_discriminator(input_shape):

        """
        Returns Discriminator NN
        :param input_shape: tuple shape generator input
        :return: Keras Model
        """

        # Input
        din = Input(
                shape=input_shape,
                name='discriminator_input'
        )

        # Hidden Layers
        d = Dense(units=64)(din)
        d = Dense(units=64)(d)
        d = Dense(units=64)(d)

        # Output
        dout = Dense(
                units=1,
                activation='sigmoid',
                name='discriminator_output'
        )(d)

        # Create Model
        model = Model(din, dout)

        return model


if __name__ == '__main__':
        
        z_dim = (10,)
        gen_out_units = 2
        dis_input_dim = (gen_out_units,)

        generator = get_generator(z_dim, gen_out_units)
        discriminator = get_discriminator(dis_input_dim)

        discriminator.compile(
                optimizer=Adam(),
                loss='binary_crossentropy',
                metrics=['accuracy']
        )

        discriminator.trainable = False

        gan_input = Input(
                shape=z_dim,
                name='gan_input'
        )

        gan_output = discriminator(generator(gan_input))
        gan = Model(gan_input, gan_output)
        gan.compile(
                 optimizer=Adam(),
                 loss='binary_crossentropy',
                 metrics=['accuracy']
        )

        gan.summary()
