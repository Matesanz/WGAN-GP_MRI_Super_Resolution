from time import strftime, localtime
from os import path


def save_models(d, g, d_of_g, folder_name):

        time = strftime("%d-%b-%H%M", localtime())
        # Save Models
        gan_name = 'gan_model_' + time + '.h5'
        generator_name = 'generator_model_' + time + '.h5'
        discriminator_name = 'discriminator_model_' + time + '.h5'
        gan_path = path.join('weights', folder_name, gan_name)
        generator_path = path.join('weights', folder_name, generator_name)
        discriminator_path = path.join('weights', folder_name, discriminator_name)
        d_of_g.save(gan_path)
        g.save(generator_path)
        d.save(discriminator_path)
