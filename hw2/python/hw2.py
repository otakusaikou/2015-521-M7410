#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from math import sqrt


def main():
    # Define coefficient matrix
    A = np.matrix([
        [1, 0, 0, 1, 0, -1, 0],
        [0, -1, 0, 1, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 1],
        [1, 1, 1, 0, 0, 0, 0]])

    # Define sigma0 and weight matrix
    s0 = 3.0
    W = np.matrix([
        [s0**2/.5**2, 0, 0, 0, 0, 0, 0],
        [0, s0**2/.1**2, 0, 0, 0, 0, 0],
        [0, 0, s0**2/.1**2, 0, 0, 0, 0],
        [0, 0, 0, s0**2/.3**2, 0, 0, 0],
        [0, 0, 0, 0, s0**2/.2**2, 0, 0],
        [0, 0, 0, 0, 0, s0**2/.3**2, 0],
        [0, 0, 0, 0, 0, 0, s0**2/.1**2]])

    # Define f matrix
    f = np.matrix([
        [.3],
        [0],
        [0],
        [.1]])

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
    N = A * W.I * A.T

    # Compute K matrix
    K = N.I * f

    # Compute residual vector
    V = W.I * A.T * K
    print "Residual : "
    print "V = \t", str(V.round(4)).replace("\n", "\n\t")
    print

    # Compute corrected observation
    L = l + V
    print "Correct observation : "
    print "L = \t", str(L.round(4)).replace("\n", "\n\t")
    print

    # Compute error of unit weight
    sigma0 = sqrt((V.T * W * V) / A.shape[0])
    print "Error of unit weight : %.4f" % sigma0

if __name__ == '__main__':
    main()
