from numpy import array, zeros, pad, floor, ceil, expand_dims, save
import nibabel as nib
from os import path, listdir
from scipy.ndimage.interpolation import zoom


def nearest_upper_multiple(number, base):
        """
        Finds closest to number upper multiple of base
        :param number: int, reference number
        :param base: int, multiple of
        :return: int, closest to number upper multiple of base
        """
        return base * ceil(number / base)


def pad_image(img: array, multiple=64):

        """
        Add zeros around image to match multiple of <multiple>
        i.e.:   original image: 181 x 217 x 181
                multiple of: 64
                target_image: 192 x 256 x 192
        :param img: np array size 181*217*217
        :param multiple: int, pad img to be multiple of <multiple>
        :return: np array, original image surrounded by zeros
        """

        # resulting shape of original image after processing
        target_shape = [nearest_upper_multiple(i, multiple) for i in img.shape]  # (182, 256, 182)
        # number of zeros to be added to original image to have target_shape shape
        total_padding = array(target_shape) - array(img.shape)  # total_padding: (11, 39, 11)
        start_padding = floor(total_padding/2)  # start_padding: (5, 19, 5)
        end_padding = ceil(total_padding/2)  # end_padding: (6, 20, 6)
        # Number of zeros to be added at start and end of original image dimensions
        padding = tuple(zip(start_padding.astype(int), end_padding.astype(int)))  # padding: ((5,6), (19, 20), (5, 6))
        padding_shape = array(padding).shape  # padding_shape = (2, 3)
        padding_zeros = zeros(shape=padding_shape, dtype=int)  # zeros around original image
        # Processed Image
        res_image = pad(img, padding, 'constant', constant_values=padding_zeros)
        # Return
        return res_image


def mnc_to_npy(mnc_file):

        """
        Converts mnc file to numpy array
        :param mnc_file: mnc file, brainweb fMRI
        :return: 3D shaped numpy array
        """

        data = mnc_file.get_fdata()
        data = data.astype('float32')
        return data


def process_mnc_files(path_origin, path_dest, process_fn):
        """
        Converts mnc files in folder to .npy files on path dest
        :param path_origin: str, path of mnc files folder
        :param path_dest: str, path of npy files folder
        """

        for filename in listdir(path_origin):

                mnc_path = path.join(path_origin, filename)
                procc_img = process_fn(mnc_path)
                name = path.splitext(filename)[0]
                npy_path = path.join(path_dest, name)
                save(npy_path, procc_img)

def process_mnc(path):

        """
        Pads single mnc file in path
        and converts to be filled into NN
        :param path: str, path of mnc file
        :return: numpy array
        """

        mnc = nib.load(path)
        img = mnc_to_npy(mnc)
        procc_img = pad_image(img)
        procc_img = expand_dims(procc_img, 3)

        return procc_img


def norm_image(img):

        """
        returns numpy array normalized to -1 +1
        :param img:
        :return:
        """

        img -= img.min()
        return (img / img.max()) * 2 - 1


def process_mnc_and_reduce(file_path):

        """
        Normalizes and Reduces mnc file to numpy array
        Array gets ready to be fed into NN
        :param file_path: path of mnc file.
        :return: numpy array
        """

        mnc = nib.load(file_path)
        img = mnc_to_npy(mnc)
        img = pad_image(img)
        img = zoom(img, (1/12, 1/16, 1/12), order=2)
        img = norm_image(img)
        img = expand_dims(img, 3)
        return img
