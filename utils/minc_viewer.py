import nibabel as nib
from os import path
import numpy as np
import matplotlib.pyplot as plt
from utils.imutils import pad_image


class Viewer(object):
        """
        Class to Visualize 3D interactive matplotlib plots
        """

        def __init__(self, data, plane=0):

                """
                Plots 3d numpy array Data

                :param data: 3d Numpy Array
                :param plane: 0, 1, 2, transverse, sagittal, frontal
                """

                fig, ax = plt.subplots(1, 1)
                fig.canvas.mpl_connect('scroll_event', self.onscroll)

                self.ax = ax
                self.ax.set_title('use scroll wheel to navigate images')

                self.X = np.flip(data, 0)
                self.plane = plane

                if self.plane == 0:
                        self.slices, _, _ = data.shape
                elif self.plane == 1:
                        _, self.slices, _ = data.shape
                elif self.plane == 2:
                        _, _, self.slices = data.shape

                self.ind = self.slices // 2

                if self.plane == 0:
                        self.im = self.ax.imshow(self.X[self.ind, :, :], cmap='inferno')
                if self.plane == 1:
                        self.im = self.ax.imshow(self.X[:, self.ind, :], cmap='inferno')
                if self.plane == 2:
                        self.im = self.ax.imshow(self.X[:, :, self.ind], cmap='inferno')

                self.update()
                plt.show()


        def onscroll(self, event):
                # print("%s %s" % (event.button, event.step))
                if event.button == 'up':
                        self.ind = (self.ind + 1) % self.slices
                else:
                        self.ind = (self.ind - 1) % self.slices
                self.update()


        def update(self):
                if self.plane == 0:
                        self.im.set_data(self.X[self.ind, :, :])
                if self.plane == 1:
                        self.im.set_data(self.X[:, self.ind, :])
                if self.plane == 2:
                        self.im.set_data(self.X[:, :, self.ind])

                self.ax.set_ylabel('slice %s' % self.ind)
                self.im.axes.figure.canvas.draw()


if __name__ == '__main__':

        IMAGE_FILE = 'brain.mnc'
        IMAGE_PATH = path.join('..', 'resources', 'mri', IMAGE_FILE)

        img = nib.load(IMAGE_PATH)
        img = img.get_fdata()
        img = pad_image(img)

        tracker = Viewer(img)
