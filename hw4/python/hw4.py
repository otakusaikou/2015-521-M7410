#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from sympy import symbols, diff, expand, solve, sqrt, pprint, Matrix


# In the case that x1 and x2 are dependent
def case1():
    a, x1, x2 = symbols('a x1 x2')              # Define symbols
    b = 3 * (a - 1)                             # Relation between a and b
    y = a * x1 + b * x2 + 5                     # Given function y
    # Error of x1 and x2
    Sigx1 = 0.8
    Sigx2 = 1.5

    print "In the case that x1 and x2 are dependent:\n"
    Sigx1x2 = Sigx1 * Sigx2     # Correlation coefficient equals to 1
    SigmaXX = np.matrix([[Sigx1**2, Sigx1x2],   # Define covariance
                         [Sigx1x2, Sigx2**2]])  # matrix of x1 and x2
    print "SigmaXX =\n"
    pprint(Matrix(SigmaXX))
    print

    # Jacobian matrix of y function with respect to x1 and x2
    Jyx = np.matrix([diff(y, x1), diff(y, x2)])

    SigmaYY = Jyx * SigmaXX * Jyx.T   # Compute covariance matrix of y function

    # Compute solution in witch the SigmaYY is minimum
    a = round(solve(diff(expand(SigmaYY[0, 0]), a))[0], 8)
    b = round(b.evalf(subs={'a': a}), 8)

    # Compute sigma y
    SigmaY = sqrt(SigmaYY[0, 0].evalf(subs={'a': a, 'b': b}))

    print "a = %.4f\nb = %.4f" % (a, b)
    print "SigmaY = %.4f" % float(SigmaY)


# In the case that x1 and x2 are independent
def case2():
    a, x1, x2 = symbols('a x1 x2')              # Define symbols
    b = 3 * (a - 1)                             # Relation between a and b
    y = a * x1 + b * x2 + 5                     # Given function y
    # Error of x1 and x2
    Sigx1 = 0.8
    Sigx2 = 1.5

    print "In the case that x1 and x2 are independent:\n"
    Sigx1x2 = 0     # Correlation coefficient equals to zero
    SigmaXX = np.matrix([[Sigx1**2, Sigx1x2],   # Define covariance
                         [Sigx1x2, Sigx2**2]])  # matrix of x1 and x2
    print "SigmaXX =\n"
    pprint(Matrix(SigmaXX))
    print

    # Jacobian matrix of y function with respect to x1 and x2
    Jyx = np.matrix([diff(y, x1), diff(y, x2)])

    SigmaYY = Jyx * SigmaXX * Jyx.T   # Compute covariance matrix of y function

    # Compute solution in witch the SigmaYY is minimum
    a = round(solve(diff(expand(SigmaYY[0, 0]), a))[0], 8)
    b = round(b.evalf(subs={'a': a}), 8)

    # Compute sigma y
    SigmaY = sqrt(SigmaYY[0, 0].evalf(subs={'a': a, 'b': b}))

    print "a = %.4f\nb = %.4f" % (a, b)
    print "SigmaY = %.4f" % float(SigmaY)


def main():
    case1()
    print '-' * 50
    case2()


if __name__ == '__main__':
    main()
