#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from math import sqrt


def main():
    # Define coefficient matrix
    B = np.matrix([
        [-1, 0, 0],
        [0, 0, 1],
        [1, 0, -1],
        [0, 1, 0],
        [0, -1, 1],
        [-1, 1, 0],
        [0, 0, -1]])

    # Define sigma0 and weight matrix
    s0 = 3.0
    s = np.array([.5, .1, .1, .3, .2, .3, .1])
    P = np.diag(s0**2/s**2)

    # Define f matrix
    f = np.matrix([
        [103.8],
        [-107.4],
        [3.7],
        [-104.6],
        [-2.8],
        [-1.1],
        [107.4]])

    # Define vector of obserbation
    l = np.matrix([
        [1.2],
        [2.4],
        [-3.7],
        [-.4],
        [2.8],
        [1.1],
        [-2.4]])

    # Compute Normal matrix
    N = B.T * P * B

    # Compute W matrix
    W = -B.T * P * f

    # Compute the unknown parameters
    X = N.I * W

    # Compute residual vector
    V = B * X + f
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

if __name__ == '__main__':
    main()
