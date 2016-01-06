#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, Matrix, lambdify


def getEqn(xp, ls, Xs):
    """Return observation equations with first observables"""
    F = Matrix(map(lambda i: ls[i] - Xs[0] * xp[i] - Xs[1], range(len(ls))))

    return F


def getCoeffConst(xp, l, ls, W, lx, X0, Xs):
    """Compute coefficient matrix B and constants matrix f"""
    l0 = np.matrix(l).T          # Transform the observables to matrix form

    # Get observation equations
    F = getEqn(xp, ls, Xs)

    # Compute jacobian matrix
    JFx = F.jacobian(Xs)

    # Create function objects for cofficient matrix B and constants matrix
    FuncB = lambdify(tuple(np.append(Xs, ls)), JFx, modules="sympy")
    FuncF = lambdify(tuple(np.append(Xs, ls)), F, modules="sympy")

    # Combine two arrays of initial values together
    val0 = np.array(np.concatenate((X0, l0)).T).flatten()

    # Substitute values for symbols
    B = np.matrix(FuncB(*val0)).astype(np.double)
    f = -np.matrix(FuncF(*val0)).astype(np.double)

    return B, f


def addObs(xp, l, ls, W, lx, X0, Xs, Wxx, M0, taw0):
    """Update normal matrix and constants matrix"""
    # Get coefficient matrix and constants matrix
    B, f = getCoeffConst(xp, l, ls, W, lx, X0, Xs)

    # Update normal matrix and constants matrix
    Q = W.I
    M1 = (M0.I * (np.identity(len(Xs)) - B.T * (Q+B*M0.I*B.T).I * B * M0.I)).I
    taw1 = taw0 + B.T * W * f

    # Solve unknown parameters and update unknown parameter observables
    X = M1.I * taw1
    X0 = X0 + X

    return M1, taw1


def subObs(xp, l, ls, W, lx, X0, Xs, Wxx, M0, taw0):
    """Update normal matrix and constants matrix"""
    # Get coefficient matrix and constants matrix
    B, f = getCoeffConst(xp, l, ls, W, lx, X0, Xs)

    # Update normal matrix and constants matrix
    Q = W.I
    Q = W.I
    M1 = (M0.I * (np.identity(len(Xs)) + B.T * (Q-B*M0.I*B.T).I * B * M0.I)).I
    taw1 = taw0 - B.T * W * f

    # Solve unknown parameters and update unknown parameter observables
    X = M1.I * taw1
    X0 = X0 + X

    return M1, taw1


def unifiedseq(xp, l, ls, W, lx, X0, Xs, Wxx):
    """Compute solution with unified sequential LSQ method"""
    # Compute normal matrix and constants matrix with only unknown parameters
    # observables
    fx = X0 - lx
    M = Wxx
    taw = -Wxx * fx

    M1, taw1 = addObs(xp, l, ls, W, lx, X0, Xs, Wxx, M, taw)

    return M1, taw1


def unified(xp, l, ls, W, lx, X0, Xs, Wxx):
    """Compute solution with batch unified LSQ method"""
    # Get coefficient matrix and constants matrix
    B, f = getCoeffConst(xp, l, ls, W, lx, X0, Xs)

    # Solve unknown parameters
    fx = X0 - lx
    M = B.T * W * B + Wxx
    taw = B.T * W * f - Wxx * fx

    return M, taw


def drawLines(xp, l, slope, intercept, annotation=False):
    """Draw lines with given slope and intercept"""
    # Create figure
    fig = plt.figure(0, figsize=(12, 9), dpi=70)
    ax = fig.add_subplot(111)

    # Set equal axis and extent of x y axis
    ax.axis("equal")
    ax.axis([0, 4, 0, 30])
    ax.grid()  # Enable grid
    ax.spines["left"].set_position("zero")
    ax.spines["bottom"].set_position("zero")

    # Set x y label and title
    plt.title("Line variation plot", size=20)
    plt.xlabel("X", size=15)
    plt.ylabel("Y", size=15)

    # Draw lines
    x = np.linspace(-5, 5, 5)
    xp = np.delete(xp, -1)
    l = np.delete(l, -1)
    for i in range(len(slope)):
        ax.plot(x, intercept[i] + x * slope[i])
        ax.plot(xp, l, "ro")

    # Add annotation if the flag is true
    if annotation:
        for i in range(len(slope)):
            ax.annotate(
                "$1%s: y=%.6fx+%.6f$" % (chr(97+i), slope[i], intercept[i]),
                xy=(xp[-1], xp[-1] * slope[i] + intercept[i]),
                xytext=(40, 20), textcoords="offset points", va="center",
                size=15,
                arrowprops=dict(arrowstyle="->"))

    plt.show()


def main():
    # Define input values
    xp = np.array([1.0, 2.0, 3.0, 4.0, 3.0])

    # Use y coordinate of measure point as observables
    l = np.array([10.01, 12.98, 16.05, 18.99, 16.02])

    # Define unknown parameter observables
    lx = np.matrix([[0], [0]])

    # Define symbol arrays for unknown parameters and observables
    Xs = np.array((symbols("m k")))
    ls = np.array(list(symbols("ya yb yc yd yc2")))

    # Get initial value of unknown parameters
    X0 = np.matrix(np.zeros(len(Xs))).T

    # Define weight vector
    w = np.ones(len(ls)) * 10**-2       # For observables
    wxx = np.ones(len(Xs)) * 10**1      # For unknown parameters
    sig0 = 10**-2       # Define a priori error

    # Compute weight matrix for observables and unknown parameters
    W = np.matrix(np.diag(sig0**2 / w**2))
    Wxx = np.matrix(np.diag(sig0**2 / wxx**2))

    # Create two list to store slope and intercept values
    slopeList = []
    interceptList = []

    # For the case with only one observable point A
    M1, taw1 = unifiedseq(xp[:1], l[:1], ls[:1], W[:1, :1], lx, X0, Xs, Wxx)
    slope, intercept = tuple(np.array(M1.I * taw1).flatten())
    slopeList.append(slope)
    interceptList.append(intercept)
    print "Only point A:\n  Slope m = %.6f, Y-intercept k = %.6f\n" \
        % (slope, intercept)

    # Add new observable point B
    M2, taw2 = addObs(
        xp[1:2], l[1:2], ls[1:2], W[1:2, 1:2], lx, X0, Xs, Wxx, M1, taw1)
    slope, intercept = tuple(np.array(M2.I * taw2).flatten())
    slopeList.append(slope)
    interceptList.append(intercept)
    print "Points A, B:\n  Slope m = %.6f, Y-intercept k = %.6f\n" \
        % (slope, intercept)

    # Add new observable point C
    M3, taw3 = addObs(
        xp[2:3], l[2:3], ls[2:3], W[2:3, 2:3], lx, X0, Xs, Wxx, M2, taw2)
    slope, intercept = tuple(np.array(M3.I * taw3).flatten())
    slopeList.append(slope)
    interceptList.append(intercept)
    print "Points A, B, C:\n  Slope m = %.6f, Y-intercept k = %.6f\n" \
        % (slope, intercept)

    # Add new observable point D
    M4, taw4 = addObs(
        xp[3:4], l[3:4], ls[3:4], W[3:4, 3:4], lx, X0, Xs, Wxx, M3, taw3)
    slope, intercept = tuple(np.array(M4.I * taw4).flatten())
    slopeList.append(slope)
    interceptList.append(intercept)
    print "Points A, B, C, D:\n  Slope m = %.6f, Y-intercept k = %.6f\n" \
        % (slope, intercept)

    # Draw line variation plot
    drawLines(xp, l, np.array(slopeList), np.array(interceptList))

    # Remove point C and add new point C'
    M5, taw5 = subObs(
        xp[2:3], l[2:3], ls[2:3], W[2:3, 2:3], lx, X0, Xs, Wxx, M4, taw4)
    M6, taw6 = addObs(
        xp[4:5], l[4:5], ls[4:5], W[4:5, 4:5], lx, X0, Xs, Wxx, M5, taw5)
    slope, intercept = tuple(np.array(M6.I * taw6).flatten())
    slopeList.append(slope)
    interceptList.append(intercept)
    print "Points A, B, D, C':\n  Slope m = %.6f, Y-intercept k = %.6f\n" \
        % (slope, intercept)

    # Check the result with batch unified LSQ method
    # Remove point C
    xp, l, ls, w = map(lambda a: np.delete(a, 2), [xp, l, ls, w])
    W = np.matrix(np.diag(sig0**2 / w**2))

    M7, taw7 = unified(xp, l, ls, W, lx, X0, Xs, Wxx)
    print "Points A, B, D, C':\n  Slope m = %.6f, Y-intercept k = %.6f\n" \
        % tuple(M7.I * taw7)

    return 0

if __name__ == "__main__":
    main()
