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


# In the case that x1 and x2 are partial dependent
def case3():
    a, c, x1, x2 = symbols('a c x1 x2')              # Define symbols
    b = 3 * (a - 1)                             # Relation between a and b
    y = a * x1 + b * x2 + 5                     # Given function y
    # Error of x1 and x2
    Sigx1 = 0.8
    Sigx2 = 1.5

    print "In the case that x1 and x2 are partial dependent:\n"
    Sigx1x2 = c     # Correlation coefficient equals to an unknown parameter c
    SigmaXX = np.matrix([[Sigx1**2, Sigx1x2],   # Define covariance
                         [Sigx1x2, Sigx2**2]])  # matrix of x1 and x2

    # Jacobian matrix of y function with respect to x1 and x2
    Jyx = np.matrix([diff(y, x1), diff(y, x2)])

    SigmaYY = Jyx * SigmaXX * Jyx.T   # Compute covariance matrix of y function

    # Compute solution in witch the SigmaYY is minimum
    dc = diff(expand(SigmaYY[0, 0]), c)     # Derivative SigmaYY  with respect
                                            # to c
    a1, a2 = solve(dc)                      # Get two solutions of a

    # Compute results with a = 0
    print "Parameter a has two solutions [%d, %d]\nIf a = %d:" % (a1, a2, a1)

    # Get solution of c with a = 0
    c1 = solve(diff(expand(SigmaYY[0, 0]), a), c)[0].evalf(subs={'a': a1})
    b1 = round(b.evalf(subs={'a': a1}), 8)

    # Compute sigma y with a = 0
    SigmaY1 = sqrt(SigmaYY[0, 0].evalf(subs={'a': a1, 'b': b1, 'c': c1}))
    print "SigmaXX =\n"
    pprint(Matrix(SigmaXX).evalf(subs={'c': round(c1, 4)}))
    print
    print "a = %.4f\nb = %.4f\nc = %.4f" % (a1, b1, c1)
    print "SigmaY = %.4f\n" % float(SigmaY1)

    # Compute results with a = 1
    print "If a = %d:" % a2

    # Get solution of c with a = 1
    c2 = solve(diff(expand(SigmaYY[0, 0]), a), c)[0].evalf(subs={'a': a2})
    b2 = round(b.evalf(subs={'a': a2}), 8)

    # Compute sigma y with a = 1
    SigmaY2 = sqrt(SigmaYY[0, 0].evalf(subs={'a': a2, 'b': b2, 'c': c2}))
    print "SigmaXX =\n"
    pprint(Matrix(SigmaXX).evalf(subs={'c': round(c2, 4)}))
    print
    print "a = %.4f\nb = %.4f\nc = %.4f" % (a2, b2, c2)
    print "SigmaY = %.4f" % float(SigmaY2)


def main():
    case1()
    print '-' * 50
    case2()
    print '-' * 50
    case3()


if __name__ == '__main__':
    main()
