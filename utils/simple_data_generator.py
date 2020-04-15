from numpy.random import exponential, sample, random
from numpy import arange, power, polyfit, poly1d, sort

import matplotlib.pyplot as plt


def get_quadratic_data_sample(n, begin=-10, end=10, noise=10, pow=2):

        """
        Create random sample of n points on parabolic distribution with some noise
        :param n: int, number of points
        :param begin: float, min value of sample
        :param end: float, max value of sample
        :param noise: float, multiplier of random divergence from trend
        :param pow: int, parabolic power
        :return: array of x and y coordinates and trend
        """

        ### Sanity Checks ###

        if type(n) is not int or n <= 0: raise Exception("Sample size must be a positive integer")
        if type(pow) is not int or pow % 2 != 0 or pow <= 0: raise Exception("Power must be a positive even integer ")
        if begin >= end: raise Exception("Min value ({}) is higher or equal than max value ({})".format(begin, end))


        mid = (end-begin)/2 + begin  # Middle point
        x = (end-begin) * random(n) + begin  # Points along x
        x = sort(x)  # Sort x
        y = power((x - mid), pow)  # Raise x
        trend = polyfit(x, y, 2)  # Calculates quadratic line with no noise
        trend = poly1d(trend)
        y += (2 * random(n) - 1) * noise # Add noise to y

        return x, y, trend


if __name__ == '__main__':

        x, y, trend = get_quadratic_data_sample(100)
        plt.plot(x,y, '.')
        plt.plot(x, trend(x))
        plt.show()