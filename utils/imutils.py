from numpy import array, ndarray, zeros, pad, floor, ceil, ones, expand_dims, save, asarray, load
from matplotlib.pyplot import imshow, show
import nibabel as nib
from os import path, listdir
from scipy.ndimage import zoom


def nearest_upper_multiple(number, base):
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

        data = mnc_file.get_fdata()
        # data = data.reshape((192, 256, 192, 1))
        data = data.astype('float32')
        return data


def process_mnc_files(path_origin, path_dest):


        for filename in listdir(path_origin):
                mnc_path = path.join(path_origin, filename)
                mnc = nib.load(mnc_path)

                img = mnc_to_npy(mnc)
                procc_img = pad_image(img)
                procc_img = expand_dims(procc_img, 3)
                name = path.splitext(filename)[0]


                npy_path = path.join(path_dest, name)
                save(npy_path, procc_img)

def process_mnc(path):

        mnc = nib.load(path)
        img = mnc_to_npy(mnc)
        procc_img = pad_image(img)
        procc_img = expand_dims(procc_img, 3)

        return procc_img


if __name__ == '__main__':

    # image = ones(shape=(181,217,181)) * 150
    # IMAGE_FILE = 'brain.mnc'
    # IMAGE_PATH = path.join('..', 'resources', 'mri', IMAGE_FILE)
    #
    # img = nib.load(IMAGE_PATH)
    # data = img.get_fdata()
    #
    # a = pad_image(data, 64)
    # a = zoom(a, (1/12, 1/16, 1/12))
    #
    # imshow(a[8])
    # show()

    PATH = path.join('..', 'resources', 'mri')
    DEST_PATH = path.join('..', 'resources', 'data')
    process_mnc_files(PATH, DEST_PATH)
    # DEST_PATH = path.join('..', 'resources', 'data', 'images.npy')
    #
    # img = load(DEST_PATH)
    # print(img)