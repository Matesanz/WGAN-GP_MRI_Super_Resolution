from time import strftime, localtime
from os import path, listdir
from numpy import asarray


def save_models(d, g, d_of_g, folder_name):

        """
        Saves generator, discriminator and gan model into folder
        :param d: keras model, discriminator
        :param g: keras model, generator
        :param d_of_g: keras model, D of G
        :param folder_name: str, destiny path
        """

        time = strftime("%d-%b-%H%M", localtime())
        # Save Models
        generator_name = 'generator_model_' + time + '.h5'
        discriminator_name = 'discriminator_model_' + time + '.h5'
        generator_path = path.join('weights', folder_name, generator_name)
        discriminator_path = path.join('weights', folder_name, discriminator_name)
        g.save(generator_path)
        d.save(discriminator_path)

        if d_of_g is not None:
                gan_name = 'gan_model_' + time + '.h5'
                gan_path = path.join('weights', folder_name, gan_name)
                d_of_g.save(gan_path)



class DataGenerator:

        """
        Python Generator that takes mnc files in folder,
        process them and returns in form of batch.
        """

        def __init__(self, folder_path, preprocess_fn):
                self.folder_path = folder_path
                self.preprocess_fn = preprocess_fn
                self.data_generator = (preprocess_fn(path.join(folder_path, filename)) for filename in listdir(folder_path))
                self.data_length = len(listdir(folder_path))
                self.data_taken = 0

        def get_batch(self, batch_size):
                """
                Return batch of processed mnc files
                :param batch_size: int, number of files to be returned
                :return: batch of 3d numpy arrays
                """
                batch = []
                for _ in range(batch_size):

                        if self.data_taken >= self.data_length: self.reset_generator()
                        sample = next(self.data_generator)
                        batch.append(sample)
                        self.data_taken += 1

                return asarray(batch)

        def reset_generator(self):
                """
                When no more files remain to be used for training
                Restarts Generator
                :return:
                """
                self.data_taken = 0
                self.data_generator = (
                        self.preprocess_fn(
                                path.join(self.folder_path, filename)
                        ) for filename in listdir(self.folder_path)
                )
