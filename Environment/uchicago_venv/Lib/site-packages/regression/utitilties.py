import numpy as np


def MAE(real, prediction):
    difference = abs(real - prediction)
    return sum(difference) / len(difference)


def MSE(real, prediction):
    difference = abs(real - prediction)
    return sum(difference ** 2) / len(difference)


def MSE_from_hypothesis(hypothesis, x, y):
    """
    Mean squared error function

    :param hypothesis: callable: hypothetical prediction function f(x)
    :param x: array: x-values
    :param y: array: y-values
    :return: float: mean squared error of the hypothetical function
    """
    m = len(y)
    return sum(np.array([(hypothesis(x[i]) - y[i]) ** 2 for i in range(m)]).flatten()) / (2 * m)


def NormalEquation(X: np.matrix, y: np.matrix):
    """
    One-step optimisation using Normal Equation

    :param X: matrix: x-values
    :param y: matrix: y-values
    :return: matrix: optimised coefficients for x-values
    """
    X = np.c_[np.ones(len(X)), X]
    return np.ndarray.flatten(np.array((np.linalg.inv(X.transpose().dot(X)).dot(X.transpose())).dot(y)))
