from time import strftime, localtime
from os import path, listdir
from utils.imutils import process_mnc


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


class DataGenerator:

        def __init__(self, folder_path):
                self.data_generator = (process_mnc(path.join(folder_path, filename)) for filename in listdir(folder_path))

        def get_batch(self, batch_size):

                batch = []
                for _ in range(batch_size):

                        sample = next(self.data_generator)
                        batch.append(sample)

                return batch


if __name__ == '__main__':

        PATH = path.join('..', 'resources', 'mri')
        data_generator = DataGenerator(PATH)

        data_generator.get_batch(2)
        data_generator.get_batch(2)
        data_generator.get_batch(2)

