import io

from numpy.random import exponential, sample, random, uniform
from numpy import arange, power, polyfit, poly1d, sort, array, float32
import seaborn as sns
import matplotlib.pyplot as plt


def get_quadratic_data_sample(n, add_noise=False, x_min=-1, x_max=1, noise=0.01, pow=2):

        """
        Create random sample of n points on parabolic distribution with some noise
        :param n: int, number of points
        :param add_noise: bool, whether to add noise to y coordinate
        :param x_min: float, min value of sample
        :param x_max: float, max value of sample
        :param noise: float, multiplier of random divergence from trend
        :param pow: int, parabolic power
        :return: numpy array shape (n, 2), x and y coordinates
        """

        ### Sanity Checks ###
        if type(n) is not int or n <= 0: raise Exception("Sample size must be a positive integer")
        if type(pow) is not int or pow % 2 != 0 or pow <= 0: raise Exception("Power must be a positive even integer ")
        if x_min >= x_max: raise Exception("Min value ({}) is higher or equal than max value ({})".format(x_min, x_max))

        x = uniform(x_min, x_max, n)  # Points along x
        mid = (x_max-x_min)/2 + x_min  # Middle point
        y = power((x-mid), pow)  # Raise x

        if add_noise: y += (2 * random(n) - 1) * noise  # Add noise to y

        # combine x and y as points in 2D plane
        coordinates = list(zip(x,y))
        # Return coordinates as numpy array
        return array(coordinates, dtype=float32)


def plot_quadratic_data(data, show=True, pow=2):

        """
        Creates a plt axes with datapoints in data
        :param data: numpy array shape (n, 2), x and y coordinates
        :param show: bool, whether to show the graph or not
        :param pow: int, parable exponent
        :return:
        """

        if type(pow) is not int or pow % 2 != 0 or pow <= 0: raise Exception("Power must be a positive even integer")

        x = data[:, 0]  # unzip x
        y = data[:, 1]  # unzip y
        ax = sns.regplot(x, y, order=pow)  # Plot
        if show: plt.show()  # Show if True
        return ax  # Return plt axes


def save_plot_to_buffer():

        """
        takes all active figure plots and adds them into memory
        its useful and computationally cheaper not save them
        when plotting in Tensorboard
        :return: memory object
        """

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        return buf


if __name__ == '__main__':

        clean_data = get_quadratic_data_sample(100)
        plot = plot_quadratic_data(clean_data)
        # plt.show()
