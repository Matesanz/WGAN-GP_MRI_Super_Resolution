from numpy import load, flip, ceil
from matplotlib import pyplot as plt


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

        def show(self):
                """
                Shows Resulting Plot
                """
                plt.show()
                plt.close(self.fig)


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

        for e in range(n_epochs):

                if e % plot_interval == 0:

                        plotter.plot_epoch(e, fake)

        plotter.show()
