from optimisation_algorithms.GradientDescent import BatchGradientDescent
from .PolynomialFunctions import Polynomial, MultivariateFunction
from .utitilties import NormalEquation
import numpy as np


class Regressor:
    """
    Parent class for different kinds of regressors

    Attributes
    ----------
    d: int
        Number of coefficients,
        either the degree of polynomial or the dimension of the input matrix in Multivariate Regression
    polynomial_function: type
        One of the classes from PolynomialFunctions.py module, that defines the estimator
    axis: int
        Axis on which to apply the function, predefined for each particular subclass

    Methods
    -------
    get_weights(self)
        Returns the coefficients of the function after fit

    set_weights(self, params: numpy.array)
        Sets coefficients to the function

    fit(self, X, y, max_iterations: int = 100000, alpha: float = 0.02, tol: float = 10 ** (-20),
            randomize: bool = False)
        Fit the function to the given data

    def predict(self, X: numpy.array)
        Predict using the estimator

    def score(self)
        Return the coefficient of determination of the prediction

    get_hypothetical_equation(self)
        Returns hypothetical equation to estimate the data

    :return:
    """

    def __init__(self, d: int, polynomial_function, axis: int):
        """
        Parameters
        ---------
        d: int
            Number of coefficients,
            either the degree of polynomial or the dimension of the input matrix in Multivariate Regression
        polynomial_function: type
            One of the classes from PolynomialFunctions.py module, that defines the estimator
        axis: int
            Axis on which to apply the function, predefined for each particular subclass
        """
        self.d = d + 1
        self._X: np.array = None
        self._y: np.array = None
        self.__coefficients: np.array = None
        self.__PolynomialFunction = polynomial_function
        self.__axis = axis

    def get_weights(self) -> np.array:
        """
        Returns the coefficients of the function after fit
        :return: np.array: coefficients
        """
        return self.__coefficients

    def set_weights(self, params: np.array):
        """
        Sets coefficients to the function
        :param params: coefficients of the function
        :return: self: object
        """
        self.__coefficients = params
        return self

    def _MSE_gradient(self, coefficients) -> np.array:
        """
        Gradient of the Mean Squared Error function, predefined for each particular subclass

        Parameters
        ---------
        coefficients: numpy.array
            coefficients of the function

        Return
        ------
        :return: numpy.array
            Gradient of the Mean Squared Error function
        """

    def fit(self, X, y, max_iterations: int = 100000, alpha: float = 0.02, tol: float = 10 ** (-20),
            randomize: bool = False):
        """
        Fit the function to the given data

        Parameters
        ---------
        X: {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data
        y: array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values
        max_iterations: int
            Maximal number of iterations
        alpha: float
            Descent change rate
        tol: float
            Tolerance number, maximal number to be considered as 0
        randomize: bool
            If True, randomizes change rate each step

        Return
        ------
        :return: self: object
            Fitted Estimator
        """
        self._X, self._y = X, y
        bgd = BatchGradientDescent(self._MSE_gradient, self.d)
        return self.set_weights(bgd.optimize(max_iterations, alpha, tol, randomize))

    def predict(self, X: np.array):
        """
        Predict using the estimator

        Parameters
        ---------
        X: {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data

        Return
        ------
        :return: numpy.array of shape (n_samples,)
            Predicted values
        """
        return np.apply_along_axis(self.__PolynomialFunction(self.get_weights()).eval, self.__axis, X)

    def score(self, X, y):
        """
        Return the coefficient of determination of the prediction

        Parameters
        ---------
        X: {array-like, sparse matrix} of shape (n_samples, n_features)
            Test samples
        y: array-like of shape (n_samples,) or (n_samples, n_targets)
            True values for X

        Return
        ------
        :return: float: coefficient of determination of the prediction
        """
        y = np.ndarray.flatten(y)
        u = sum((y - self.predict(X)) ** 2)
        v = sum((y - sum(y) / len(y)) ** 2)
        return 1 - u / v

    def get_hypothetical_equation(self):
        """
        Returns hypothetical equation to estimate the data

        :return: object: estimator function object
        """
        return self.__PolynomialFunction(self.get_weights())


class PolynomialRegressor(Regressor):
    """
       Attributes
       ----------
       d: int
           Number of coefficients,
           either the degree of polynomial or the dimension of the input matrix in Multivariate Regression

       Methods
       -------
       get_weights(self)
           Returns the coefficients of the function after fit

       set_weights(self, params: numpy.array)
           Sets coefficients to the function

       _MSE_gradient(self, coefficients)
           Gradient of the Mean Squared Error function

       fit(self, X, y, max_iterations: int = 100000, alpha: float = 0.02, tol: float = 10 ** (-20),
               randomize: bool = False)
           Fit the function to the given data

       def predict(self, X: numpy.array)
           Predict using the linear model

       def score(self)
           Returns the coefficient of determination of the prediction

       get_hypothetical_equation(self)
           Returns hypothetical equation to estimate the data

       :return:
       """

    def __init__(self, d: int):
        super().__init__(d, Polynomial, 0)

    def _MSE_gradient(self, coefficients) -> np.array:
        m = len(self._y)
        hypothesis = Polynomial(coefficients).eval
        return \
            np.array([sum([(hypothesis(self._X[i]) - self._y[i]) * self._X[i] ** j for i in range(m)]) / m
                      for j in range(self.d)])


class LinearRegressor(PolynomialRegressor):
    """
    Attributes
    ----------
    d: int
        Number of coefficients,
        either the degree of polynomial or the dimension of the input matrix in Multivariate Regression

    Methods
    -------
    get_weights(self)
        Returns the coefficients of the function after fit

    set_weights(self, params: numpy.array)
        Sets coefficients to the function

    _MSE_gradient(self, coefficients)
        Gradient of the Mean Squared Error function

    fit(self, X, y, max_iterations: int = 100000, alpha: float = 0.02, tol: float = 10 ** (-20),
            randomize: bool = False)
        Fit the function to the given data

    def predict(self, X: numpy.array)
        Predict using the linear model

    def score(self)
        Returns the coefficient of determination of the prediction

    get_hypothetical_equation(self)
        Returns hypothetical equation to estimate the data

    :return:
    """

    def __init__(self):
        super().__init__(1)

    def fit(self, X: np.array, y: np.array, **kwargs):
        """
        Fit the function to the given data

        Parameters
        ---------
        X: {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data
        y: array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values

        Return
        ------
        :return: self: object
            Fitted Estimator
        """
        self._X, self._y = X, y
        return self.set_weights(NormalEquation(X, y))


class MultivariateRegressor(Regressor):
    """
    Attributes
    ----------
    d: int
        Number of coefficients,
        either the degree of polynomial or the dimension of the input matrix in Multivariate Regression

    Methods
    -------
    get_weights(self)
        Returns the coefficients of the function after fit

    set_weights(self, params: numpy.array)
        Sets coefficients to the function

    _MSE_gradient(self, coefficients)
        Gradient of the Mean Squared Error function

    fit(self, X, y, max_iterations: int = 100000, alpha: float = 0.02, tol: float = 10 ** (-20),
            randomize: bool = False)
        Fit the function to the given data

    def predict(self, X: numpy.array)
        Predict using the linear model

    def score(self)
        Returns the coefficient of determination of the prediction

    get_hypothetical_equation(self)
        Returns hypothetical equation to estimate the data

    :return:
    """

    def __init__(self, d: int):
        super().__init__(d, MultivariateFunction, 1)

    def _MSE_gradient(self, coefficients) -> np.array:
        m = len(self._y)
        hypothesis = MultivariateFunction(coefficients).eval

        return np.ndarray.flatten(np.array(
            [sum([(hypothesis(self._X[i]) - self._y[i]) for i in range(m)])] +
            [sum([(hypothesis(self._X[i]) - self._y[i]) * self._X[i][j - 1] for i in range(m)])
             for j in range(1, self.d)]) / m)

    def fit(self, X: np.array, y: np.array, max_iterations: int = 100000, alpha: float = 0.02, tol: float = 10 ** (-20),
            randomize: bool = False):
        if self.d < 10 ** 6:
            try:
                self._X, self._y = X, y
                return self.set_weights(NormalEquation(X, y))
            except np.linalg.LinAlgError:
                pass
        return super().fit(X, y, max_iterations, alpha, tol, randomize)
