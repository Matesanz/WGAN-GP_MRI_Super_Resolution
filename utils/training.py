from time import strftime, localtime
from os import path, listdir
from utils.imutils import process_mnc
from numpy import asarray


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
                self.folder_path = folder_path
                self.data_generator = (process_mnc(path.join(folder_path, filename)) for filename in listdir(folder_path))
                self.data_length = len(listdir(folder_path))
                self.data_taken = 0

        def get_batch(self, batch_size):

                batch = []
                for _ in range(batch_size):

                        if self.data_taken >= self.data_length: self.reset_generator()
                        sample = next(self.data_generator)
                        batch.append(sample)
                        self.data_taken += 1

                return asarray(batch)

        def reset_generator(self):
                # print("generator reseted")
                self.data_taken = 0
                self.data_generator = (
                        process_mnc(path.join(self.folder_path, filename)) for filename in listdir(self.folder_path)
                )



if __name__ == '__main__':

        PATH = path.join('..', 'resources', 'mri')
        data_generator = DataGenerator(PATH)
        data_generator.get_batch(2)
