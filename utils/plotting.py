from numpy import load, flip, ceil
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from utils.minc_viewer import Viewer

class TrainingImagePlotter:

        def __init__(self, plot_interval, n_epochs, slices):

                """
                This class builds a graphic that collects generated MRI during training
                MRI image is 3d so it slices MRI into plain images to be plotted
                Result is a GRID of (slices x plot_epoch) images
                :param plot_interval: int, perform a plot every n epochs
                :param n_epochs: int, total number of training epochs
                :param slices: int, number of 2D images to take from 3D MRI
                """

                self.col_n = int(ceil(n_epochs / plot_interval))
                self.slices = slices
                self.plot_interval = plot_interval
                self.fig, self.axes = plt.subplots(
                        self.slices,
                        self.col_n,
                        figsize=(self.col_n, self.slices)
                )

                self.fig.subplots_adjust(top=0.82)
                self.fig.suptitle('Generated MRI Evolution', fontsize=16)

        def plot_epoch(self, e, img):

                """
                PLots slices of MRI into corresponding column
                :param e: int, actual epoch
                :param img: 3D numpy to be plotted
                """

                img_len = len(img[0])  # total length of 3D image
                col = e // self.plot_interval  # idx of actual column
                img = flip(img, 0)  # Originally MRI is upside down, flip.


                # Iterate over rows in figure
                for idx, ax in enumerate(self.axes):

                        # if col_n == 1 then ax == self.axes so no need for iteration
                        if self.col_n > 1: ax = ax[col]
                        # Remove ticks
                        ax.set_xticks([])
                        ax.set_yticks([])
                        # Get 2D image from 3D MRI evenly separated
                        slice_idx = img_len // self.slices * idx
                        slice = img[slice_idx]
                        # Add Side and Upper Titles to GRID
                        if idx is 0: ax.set_title('Epoch {}'.format(e), size=8)
                        if col is 0: ax.set_ylabel('Slice #{}'.format(slice_idx), size=8)
                        # Print image into actual axe
                        ax.imshow(
                                slice,
                                vmin=img.min(),
                                vmax=img.max(),
                                cmap='inferno')

        def save(self, filepath):

                plt.savefig(filepath)


        def show(self):
                """
                Shows Resulting Plot
                """
                plt.show()

class TrainingEvolutionPlotter():

        def __init__(self):

                self.fig = plt.figure(figsize=(10, 5))
                self.fig.suptitle("3D WGAN GP Training Evolution", size=20)
                rows = 3
                cols = 7
                self.fig.subplots_adjust(top=0.82, bottom=0.2)
                self.fig_idx = 0
                grid = plt.GridSpec(rows, cols, wspace=0.1, hspace=0.25)
                sns.set(color_codes=True, style='ticks', palette='colorblind')
                plt.show(block=False)
                pos = {
                        'Transverse': ['Superior', 'Central', 'Inferior'],
                        'Frontal': ['Posterior', 'Central', 'Anterior'],
                        'Sagittal': ['Left', 'Central', 'Right']
                }

                views = list(pos.keys())

                slice_areas = [[]]

                self.axes_img = []
                i = 0
                for r in range(rows):
                        for c in range(cols):
                                if c == 3: break
                                view = views[r]
                                ax = plt.subplot(grid[r, c])
                                ax.set_xticks([])
                                ax.set_yticks([])
                                self.axes_img.append(ax)
                                if c < 3: ax.set_xlabel(pos[view][c], size=8)
                                if c == 0: ax.set_ylabel(view, size=8)
                                if c == 1 and r == 0: ax.set_title('Generated 3D MRI')


                # ax0 = plt.subplot(grid[0, 0])
                # plt.subplot(grid[0, 1])
                # plt.subplot(grid[0, 2])
                # plt.subplot(grid[1, 0])
                # plt.subplot(grid[1, 1])
                # plt.subplot(grid[1, 2])
                # plt.subplot(grid[2, 0])
                # plt.subplot(grid[2, 1])
                # plt.subplot(grid[2, 2])
                self.ax_train = plt.subplot(grid[:, 4:])
                sns.despine(ax=self.ax_train)


        def plot_img(self, img):

                imgs = []
                img = np.flip(img, 0)

                for dimension in range(3):

                        temp = np.rollaxis(img, dimension)
                        img_len = len(temp[0])  # total length of 3D image
                        slice_idxs = [img_len // 4 * (i+1) for i in range(3)]
                        slices = temp[slice_idxs]
                        imgs.extend(slices)

                for ax, img in zip(self.axes_img, imgs):

                        ax.imshow(
                                img,
                                vmin=img.min(),
                                vmax=img.max(),
                                cmap='inferno')

                        # if plane == 0:
                        #         self.im = self.ax.imshow(self.X[self.ind, :, :], cmap='inferno')
                        # if self.plane == 1:
                        #         self.im = self.ax.imshow(self.X[:, self.ind, :], cmap='inferno')
                        # if self.plane == 2:
                        #         self.im = self.ax.imshow(self.X[:, :, self.ind], cmap='inferno')



        def plot_train(self, g_loss, c_loss):

                epochs = np.linspace(0, len(g_loss)+1, len(g_loss))

                self.ax_train.clear()
                self.ax_train = sns.lineplot(epochs, g_loss, ax=self.ax_train, label='Generator Loss', color='g', linewidth=2.0)
                self.ax_train = sns.lineplot(epochs, c_loss, ax=self.ax_train, label='Critic Loss', color='orange',
                                  linewidth=2.0)
                self.ax_train.set_xlabel('Epochs',  size=9)
                self.ax_train.set_ylabel('Loss',  size=9)
                self.ax_train.set_title('GAN loss')
                self.ax_train.legend(loc='upper right', frameon=True)


        def save(self, filepath):

                plt.savefig('D:\\Matesanz\\Imágenes\\training\\plotting\\dc_wgan_low_{}.png'.format(
                        self.fig_idx
                ))

                self.fig_idx += 1

        def show(self):
                """
                Shows Resulting Plot
                """
                plt.show()
                # plt.close(self.fig)


if __name__ == '__main__':

        # generator.load_weights("../models/weights/dc_wgan_low/generator_dc_wgan_low.h5")
        # # generate fake sample to visualize
        # z_control = tf.random.normal((1, 10))  # Vector to feed gen and control training evolution
        # fake = generator(z_control)[0]
        # fake = squeeze(fake, 3)


        plot_interval = 200
        n_epochs = 1001
        slices = 4
        plotter = TrainingImagePlotter(plot_interval, n_epochs, slices)
        fake = load('fake.npy')

        loss = []

        for e in range(n_epochs):

                loss.append(np.random.randint(0, 100))

                if e % plot_interval == 0:

                        plotter.plot_epoch(e, fake)
                        # plotter.plot_train(loss, loss)
                        # plotter.save(None)

        Viewer(fake, 1)
        plotter.show()
