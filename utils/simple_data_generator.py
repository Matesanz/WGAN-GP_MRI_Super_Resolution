import io

from numpy.random import exponential, sample, random
from numpy import arange, power, polyfit, poly1d, sort, array, float32

import matplotlib.pyplot as plt


def get_quadratic_data_sample(n, x_min=-10, x_max=10, noise=10, pow=2):

        """
        Create random sample of n points on parabolic distribution with some noise
        :param n: int, number of points
        :param x_min: float, min value of sample
        :param x_max: float, max value of sample
        :param noise: float, multiplier of random divergence from trend
        :param pow: int, parabolic power
        :return: two arrays of x and y coordinates with and without added noise
        """

        ### Sanity Checks ###

        if type(n) is not int or n <= 0: raise Exception("Sample size must be a positive integer")
        if type(pow) is not int or pow % 2 != 0 or pow <= 0: raise Exception("Power must be a positive even integer ")
        if x_min >= x_max: raise Exception("Min value ({}) is higher or equal than max value ({})".format(x_min, x_max))


        mid = (x_max-x_min)/2 + x_min  # Middle point
        x = (x_max-x_min) * random(n) + x_min  # Points along x
        x = sort(x)  # Sort x
        y = power((x - mid), pow)  # Raise x

        y_noise = y + (2 * random(n) - 1) * noise # Add noise to y

        noisy_points_array = array(list(zip(x, y_noise)), dtype=float32)
        clean_points_array = array(list(zip(x, y)), dtype=float32)

        return noisy_points_array, clean_points_array


def plot_simple_data(data, plot_type='line', show=False):

        x = data[:, 0]
        y = data[:, 1]
        # plot = None
        if plot_type == 'line': plot = plt.plot(x, y)
        elif plot_type == 'points': plot = plt.plot(x, y, '.')
        else: raise Exception('Arg "plot_type" must be "line" or "points", {} format not supported'.format(plot_type))
        if show: plt.show()
        return plot

def save_plot_to_buffer():

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        return buf






if __name__ == '__main__':

        noisy_data, clean_data = get_quadratic_data_sample(100)


        plot_noisy_data = plot_simple_data(noisy_data, plot_type='points')
        plot_clean_data = plot_simple_data(clean_data, plot_type='line')
        buf = save_plot_to_buffer()
