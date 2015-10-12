#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from math import sqrt
from sympy import symbols, diff
from fractions import gcd


# Get greatest common divisor of a list of number
def getGCD(l):
    # Check if any float contained in list. If it so, return 1.0
    if False in map(lambda x: x == int(x), l):
        return 1.0

    return reduce(lambda x, y: gcd(x, y), l)


def matrixApproach(B, P, f, l):
    N = B.T * P * B                         # Compute Normal matrix
    W = -B.T * P * f                        # Compute W matrix
    X = N.I * W                             # Compute the unknown parameters
    V = B * X + f                           # Compute residual vector
    print "Residual : "
    print "V = \t", str(V.round(4)).replace("\n", "\n\t")
    print

    L = l + V               # Compute corrected observation
    print "Correct observation : "
    print "L = \t", str(L.round(4)).replace("\n", "\n\t")
    print

    # Compute error of unit weight
    sigma0 = sqrt((V.T * P * V) / (B.shape[0] - B.shape[1]))
    print "Error of unit weight : %.4f" % sigma0

    return X, V


def longHandApproach(B, P, f, l):
    Ha, Hb, Hc = symbols('Ha Hb Hc')        # Define symbols
    x = np.matrix([Ha, Hb, Hc]).T
    V = B * x + f                           # List observation equations
    phi = (V.T * P * V)[0, 0]               # Compute mean square error

    # Differentiate phi with respect to Ha, Hb and Hc
    # Take their corefficients as matrix form
    D = np.array([
        map(
            lambda i: diff(phi, Ha).as_coefficients_dict()[i[0, 0]],
            np.vstack([x, 1])),
        map(
            lambda i: diff(phi, Hb).as_coefficients_dict()[i[0, 0]],
            np.vstack([x, 1])),
        map(
            lambda i: diff(phi, Hc).as_coefficients_dict()[i[0, 0]],
            np.vstack([x, 1]))]).astype(float)

    D = np.round(D, 8)
    D = np.matrix(
        map(
            lambda i: D[i] / getGCD(D[i]),
            [0, 1, 2]))                     # Simplify every equations

    Left, Right = D[:, :3], -D[:, 3:]
    X = Left.I * Right                      # Solve unknowns

    V = B * X + f                           # Compute residual vector
    print "Residual : "
    print "V = \t", str(V.round(4)).replace("\n", "\n\t")
    print

    # Compute corrected observation
    L = l + V
    print "Correct observation : "
    print "L = \t", str(L.round(4)).replace("\n", "\n\t")
    print

    # Compute error of unit weight
    sigma0 = sqrt((V.T * P * V) / (B.shape[0] - B.shape[1]))
    print "Error of unit weight : %.4f" % sigma0

    return X, V


def main():
    B = np.matrix([
        [-1, 0, 0],
        [0, 0, 1],
        [1, 0, -1],
        [0, 1, 0],
        [0, -1, 1],
        [-1, 1, 0],
        [0, 0, -1]])                        # Define coefficient matrix

    s0 = 3.0
    s = np.array([.5, .1, .1, .3, .2, .3, .1])
    P = np.diag(s0**2/s**2)                 # Define sigma0 and weight matrix

    f = np.matrix([
        [103.8],
        [-107.4],
        [3.7],
        [-104.6],
        [-2.8],
        [-1.1],
        [107.4]])                           # Define f matrix

    l = np.matrix([
        [1.2],
        [2.4],
        [-3.7],
        [-.4],
        [2.8],
        [1.1],
        [-2.4]])                            # Define vector of obserbation

    # Solve problem with matrix approach
    print "*" * 10 + "Result of matrix approach" + "*" * 10 + "\n"
    Xm, Vm = matrixApproach(B, P, f, l)
    print

    print "*" * 10 + "Result of long-hand approach" + "*" * 10 + "\n"
    # Solve problem with long-hand approach
    Xl, Vl = longHandApproach(B, P, f, l)
    print


if __name__ == '__main__':
    main()
