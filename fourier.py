from math import sin, cos, sqrt
from numpy import array, pi, linspace
from scipy.integrate import quad 


def fourierInnerProduct(func1, func2):
    """ Inner product for the Fourier series.

        func1, func2: functions upon which the product will be computed.
    """
    # integrate the product of func1 and func2 over -pi to pi
    start, end = quad(lambda x: func1(x) * func2(x), -pi, pi)

    # return 
    return (end - start)/pi

def project(function, basis, innerProductDef):
    """ Project 'function' onto the vector space with basis 'basis'.

        function: function to be projected onto 'basis'.
        basis: list of functions defining the basis of the vector space.
        innerProductDef: definition of an inner product. Must accept two
                         functions and a resolution argument.  Should return
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


def fourier(function, order = 10):
    """ Returns a function representing the fourier series for the input.

        function:   function for which the fourier series will be calculated.
        order:      order of the fourier series to be calculated.
                    ex) order = 10 calculates terms up to cos(10t), and sin(10t)
    """

    # create a list of basis functions for the vector space of sines and cosines.
    # The first basis vector/function is the constant function, sqrt(pi)
    basis = [lambda x: sqrt(pi)] + 2 * order * [0]

    # iterate through each pair of sinusoids
    for index in xrange(1, order + 1):
        # assign the next two basis vectors  sin(n*x) and cos(n*x) up
        # through n = order
        basis[2*index - 1] = (lambda x, index=index: sin(index * x))
        basis[2*index] = (lambda x, index=index:  cos(index * x))

    # calculate the projection of 'function' onto the generated basis
    projFunc, coeffs = project(function, basis, fourierInnerProduct)

    # return the projected function and the coefficients of the series
    return projFunc, coeffs

        
    
if __name__ == '__main__':
    from numpy import exp
    import matplotlib.pyplot as plt

    # define your function here.  may be a lambda such as:
    # ex) f(x) = x^x
    # func = lambda x: x**x
    #
    # square pulse 
    # func = lambda x: 2 * (-.5 < x < .5)
    func = exp

    # resolution of the plot
    res = 100

    # get the series and coefficients
    fs, coeffs= fourier(func, order = 10)

    # x values to plot 
    args = linspace(-pi, pi, res)

    # plot it
    fig = plt.figure(1)
    plt.plot(args, [fs(arg) for arg in args])
    plt.plot(args, func(args))
    plt.show()
    
