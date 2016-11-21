""" Module exploiting Python's functional programming features to demonstrate
    that the Fourier series is actually just a vector projection.
"""

from math import sin, cos, pi
from scipy.integrate import quad

SQRT_PI = pi ** .5

def quad_integrator(func, lower_bound, upper_bound):
    start, end = quad(func, lower_bound, upper_bound)
    return end - start
    

def fourierInnerProduct(func1, func2, integrator = quad_integrator):
    """ Inner product for the Fourier series.
        Integral from -pi to pi of ((func1(t) * func2(t))dt) / pi

        Arguments:
        func1, func2:   Functions upon which the product will be computed.
        integrator:     Function that accepts a function, a lowerbound, and
                        an upper bound and returns the integral of that function
                        from it's lower bound to it's upper bound (one number scalar).
                        integrator(func: function, lower_bound: float, upper_bound: float)->float
                        
    """

    # integrate the product of func1 and func2 over -pi to pi and return
    return integrator(lambda x: func1(x) * func2(x), -pi, pi)/pi

    # integrate the product of func1 and func2 over -pi to pi
    start, end = integrator(lambda x: func1(x) * func2(x), -pi, pi)

    return (end - start)/pi


def project(function, basis, innerProductDef):
    """ Project 'function' onto the vector space with basis 'basis'.

        Arguments:
        function: function to be projected onto 'basis'.
        basis: list of functions defining the basis of the vector space.
        innerProductDef: definition of an inner product. Must accept two
                         functions and should return
                         scalar values according to the definition of the
                         inner product, found here:
                         https://en.wikipedia.org/wiki/Inner_product_space#Definition
    """
    
    # initialize lists for coefficients and projections because I have an
    # irrational fear of append()
    projection = [None for vector in basis]
    coeffs = [None for vector in basis]

    # iterate through the basis functions
    for index, vector in enumerate(basis):
        
        # get the scaling factors for the projections of each basis function
        coeffs[index] = (innerProductDef(function, vector)/
                         innerProductDef(vector, vector))

        # create a new list of lambdas containing the projected basis functions
        projection[index] = (lambda t, index = index:
                             coeffs[index] * basis[index](t) )

    # return a function that evaluates each lambda and sums their values
    # as well as coeffs because sometimes you need that I guess.
    return (lambda arg: sum([vec(arg) for vec in projection])), coeffs


def fourier(function, order = 10, projection = project, innerProductDef = fourierInnerProduct):
    """ Returns a function representing the fourier series for the input.
        function:   function for which the fourier series will be calculated.
        order:      order of the fourier series to be calculated.
                    ex) order = 10 calculates terms up to cos(10t), and sin(10t)
    """
    
    # create a list of basis functions for the vector space of sines and cosines.
    # The first basis vector/function is the constant function, sqrt(pi)
    basis = [lambda x: SQRT_PI] + 2 * order * [0]

    # iterate through each pair of sinusoids
    for index in xrange(1, order + 1):
        # assign the next two basis vectors  sin(n*x) and cos(n*x) up
        # through n = order
        basis[2*index - 1] = (lambda x, index=index: sin(index * x))
        basis[2*index] = (lambda x, index=index:  cos(index * x))

    # calculate the projection of 'function' onto the generated basis
    projFunc, coeffs = projection(function, basis, innerProductDef)

    # return the projected function and the coefficients of the series
    return projFunc, coeffs

        
    
if __name__ == '__main__':
    from numpy import linspace
    from math import exp, sqrt
    import matplotlib.pyplot as plt

    # define your function here.  may be a lambda such as:
    func = lambda x: sqrt(abs(x)**(.5*x))
    # Other options:
    # square pulse 
    # func = lambda x: 2 * (-.5 < x < .5)
    # exponential
    # func = exp

    # get the series and coefficients
    # note that fs is a *continuous function* not just an array of discrete values
    fs, coeffs = fourier(func, order = 10)

    # resolution of the plot
    res = 300
    
    # x values to plot 
    args = linspace(-pi, pi, res)

    # plot it
    fig = plt.figure(1)
    plt.plot(args, [fs(arg) for arg in args])
    plt.plot(args, [func(arg) for arg in args])
    plt.show()
