class Monomial:
    """
    Represents a monomial function: f(x) = k*x^n

    Attributes
    ----------
    coefficient: float
        Coefficient k
    power: int
        Power n

    Methods
    -------
    eval(self, x: float)
        Returns the value of f(x)

    derivative(self)
        Returns the derivative monomial

    :return:
    """
    def __init__(self, coefficient: float, power: int):
        self.coefficient = coefficient
        self.power = power

    def __str__(self):
        if self.coefficient == 0:
            return '0'
        if self.power == 0:
            return str(self.coefficient)

        coefficient = '' if self.coefficient == 1 else str(self.coefficient)
        power = '' if self.power == 1 else '^' + str(self.power)

        return coefficient + 'x' + power

    def eval(self, x: float):
        return self.coefficient * x ** self.power

    def derivative(self):
        return Monomial(self.coefficient * self.power, self.power - 1)


class Polynomial:
    """
    Represents a polynomial function: f(x) = sum(k*x^n) for n in (0, k)

    Attributes
    ----------
    coefficients: iterable
        Coefficients of monomials of each of the powers respectively

    Methods
    -------
    eval(self, x: float)
        Returns the value of f(x)

    derivative(self)
        Returns the derivative polynomial

    :return:
    """
    def __init__(self, coefficients):
        self.coefficients = coefficients

    def __str__(self):
        n = len(self.coefficients)
        return ' + '.join([str(Monomial(self.coefficients[i], i)) for i in range(n)[::-1] if self.coefficients[i] != 0])

    def eval(self, x: float):
        n = len(self.coefficients)
        return sum([Monomial(self.coefficients[i], i).eval(x) for i in range(n)])

    def derivative(self):
        n = len(self.coefficients)
        return Polynomial([self.coefficients[i] * i for i in range(n)][1:])


class LinearFunction(Polynomial):
    """
    Represents a linear as a particular case of polynomial functions: f(x) = a*x + b

    Attributes
    ----------
    a: float
        Coefficient of x
    b: float
        Free member of polynomial

    Methods
    -------
    eval(self, x: float)
        Returns the value of f(x)

    derivative(self)
        Returns the derivative of the function

    :return:
    """
    def __init__(self, a: float, b: float):
        super().__init__([b, a])


class MultivariateFunction:
    """
    Represents a multivariate linear function: f(x) = sum(k*x^n) for n in (0, k)

    Attributes
    ----------
    coefficients: iterable
        Coefficients of monomials of each of the powers respectively

    Methods
    -------
    eval(self, x: float)
        Returns the value of f(x)

    gradient(self)
        Returns the gradient of the function

    :return:
    """
    def __init__(self, coefficients):
        self.coefficients = coefficients

    def __str__(self):
        n = len(self.coefficients)
        return str(self.coefficients[0]) + ' + ' + ' + '.join([f'{self.coefficients[i]} * x_{i}' for i in range(n)[1:]])

    def eval(self, x):
        n = len(self.coefficients)
        return self.coefficients[0] + sum([self.coefficients[i] * x[i-1] for i in range(n)[1:]])

    def gradient(self):
        return self.coefficients[1:]
