#!/usr/bin/env python
# -*- coding: utf-8 -*-
from sympy import symbols, Matrix, lambdify, S
import numpy as np
import sys


LANG = sys.stdout.encoding          # Get system language code
np.set_printoptions(suppress=True)  # Disable scientific notation for numpy


def getEqn(Xp, Yp, xs, ls):
    """List observation equations"""
    F = Matrix([
        [-((xs[6] - Xp[0])**2 + (xs[7] - Yp[0])**2)**.5 + ls[0]],
        [-((xs[4] - Xp[0])**2 + (xs[5] - Yp[0])**2)**.5 + ls[1]],
        [-((xs[4] - xs[2])**2 + (xs[5] - xs[3])**2)**.5 + ls[2]],
        [-((xs[6] - xs[2])**2 + (xs[7] - xs[3])**2)**.5 + ls[3]],
        [-((xs[2] - xs[0])**2 + (xs[3] - xs[1])**2)**.5 + ls[4]]])

    return F


def getConstraint(xs, lcs):
    """List constraint equations"""
    # Point B, D and E lie on a straight line
    FC = Matrix([
        [-((xs[5]-xs[1]) * (xs[6]-xs[0])
         - (xs[7]-xs[1]) * (xs[4]-xs[0])) + lcs[0]]])

    return FC


def getWcc(FuncC, x0, Wxx):
    """Compute weight matrix of constraints"""
    # Covariance matrix of unknown parameters
    Qxx = Wxx.I

    JFCx = np.matrix(FuncC(*x0)).astype(np.double)
    Wcc = (JFCx * Qxx * JFCx.T).I

    return Wcc


def unified(Xp, Yp, l, W, Wxx, xs, ls):
    """ Compute solution with unified LSQ method for case which has no other
    information"""
    print ("*" * 25 + "  Start  " + "*" * 25)

    # Known information, coordinates of point D and E are inherited from hw7
    lx = np.matrix(np.array(zip(Xp[1:], Yp[1:])).flatten()).T

    # Use known information as initial values
    x0 = np.copy(lx)
    l0 = np.matrix(l).T

    # Get observation equations
    F = getEqn(Xp, Yp, xs, ls)

    # Compute jacobian matrices
    JFx = F.jacobian(xs)
    JFl = F.jacobian(ls)

    # Create function objects for cofficient matrix A, B and f matrix
    FuncA = lambdify(tuple(ls), JFl, modules='sympy')
    FuncB = lambdify(tuple(xs), JFx, modules='sympy')
    FuncF = lambdify(tuple(np.append(xs, ls)), F, modules='sympy')

    l = np.matrix(l).T                          # Observations with matrix form
    X = np.ones(1)                              # Initial value for iteration
    counter = 1                                 # Loop counter

    # Iteration process
    while abs(X.sum()) > 10**-15:
        # Combine two arrays of initial values together
        val0 = np.array(np.concatenate((x0, l0)).T).flatten()

        # Substitute values for symbols
        A = np.matrix(FuncA(*np.array(l0).flatten())).astype(np.double)
        B = np.matrix(FuncB(*np.array(x0).flatten())).astype(np.double)
        F0 = np.matrix(FuncF(*val0)).astype(np.double)
        f = -F0 - A * (l - l0)
        fx = x0 - lx

        # Solve unknown parameters
        Q = W.I
        Qe = A * Q * A.T
        We = Qe.I
        N = (B.T * We * B)                 # Compute normal matrix
        t = (B.T * We * f)                 # Compute t matrix
        X = (N + Wxx).I * (t - Wxx * fx)   # Compute the unknown parameters
        V = A.T * We * (f - B * X)         # Compute residual vector

        # Update initial values
        X = X.astype(np.double)
        x0 += X
        l0 = l + V

        # Output messages for iteration process
        # print "Iteration count: %d" % counter, u"|ΔX| = %.6f" % abs(X.sum())
        counter += 1    # Update Loop counter

    # Compute residual vector
    V = A.T * We * (f - B * X)

    # Compute error of unit weight
    s0 = ((V.T * V)[0, 0] / (len(l0)))**0.5

    # Compute other information
    QXX = (N + Wxx).I
    SigmaXX = s0**2 * QXX
    param_std = np.sqrt(np.diag(SigmaXX))

    # Output results
    print "Point coordinates:"
    print ("%-8s"+" %-9s"*4) % ("Point", "X", "Y", "SD-X", "SD-Y")
    print ("%-8s"+" %-9.6f"*4) % (
        "Point B", x0[0], x0[1], param_std[0], param_std[1])
    print ("%-8s"+" %-9.6f"*4) % (
        "Point C", x0[2], x0[3], param_std[2], param_std[3])
    print ("%-8s"+" %-9.6f"*4) % (
        "Point D", x0[4], x0[5], param_std[4], param_std[5])
    print ("%-8s"+" %-9.6f"*4) % (
        "Point E", x0[6], x0[7], param_std[6], param_std[7])

    print "\nMeasured distances:"
    print ("%-8s"+" %-9s"*2) % ("Distance", "value", "res")
    for i in range(len(l0)):
        print ("%-8s"+" %-9.6f"*2) % ("s%d" % (i+1), l0[i], V[i, 0])

    print "\nSlope difference between BD and BE: %.6f" \
        % abs((x0[5]-x0[1]) / (x0[4]-x0[0]) - (x0[7]-x0[1]) / (x0[6]-x0[0]))

    print "\nCofactor matrix:"
    print u"QΔΔ =".encode(LANG)
    print QXX
    print ("*" * 26 + "  End  " + "*" * 26)

    print "\nStandard error of unit weight : %.4f" % s0
    print "Degree of freedom: %d" % (len(l0))


def unifiedC(Xp, Yp, l, W, Wxx, xs, ls, lcs):
    """ Compute solution for case which which has constraint conditions with
    unified LSQ method for case which has no other information"""
    print ("*" * 25 + "  Start  " + "*" * 25)

    # Known information, coordinates of point D and E are inherited from hw7
    lx = np.matrix(np.array(zip(Xp[1:], Yp[1:])).flatten()).T

    # Use known information as initial values
    x0 = np.copy(lx)
    l0 = np.matrix(l).T

    # Get observation equations
    F = getEqn(Xp, Yp, xs, ls)

    # Get constraint equations
    FC = getConstraint(xs, lcs)

    # Compute jacobian matrices
    JFx = F.jacobian(xs)
    JFl = F.jacobian(ls)
    JFCx = FC.jacobian(xs)
    JFClc = FC.jacobian(lcs)

    # Create function objects for cofficient matrix A, B, C, f and g matrix
    FuncA = lambdify(tuple(ls), JFl, modules='sympy')
    FuncB = lambdify(tuple(xs), JFx, modules='sympy')
    FuncF = lambdify(tuple(np.append(xs, ls)), F, modules='sympy')
    FuncAc = lambdify(tuple(xs), JFClc, modules='sympy')
    FuncC = lambdify(tuple(xs), JFCx, modules='sympy')
    FuncG = lambdify(tuple(np.append(xs, lcs)), FC, modules='sympy')

    # Compute weight matrix of constraints
    Wcc = getWcc(FuncC, x0, Wxx)

    # Treat the constraints as observations and compute its value
    lc = -np.matrix(FuncG(*tuple(np.append(x0, 0))))
    lc0 = np.copy(lc)

    l = np.matrix(l).T                          # Observations with matrix form
    X = np.ones(1)                              # Initial value for iteration
    counter = 1                                 # Loop counter

    # Iteration process
    while abs(X.sum()) > 10**-15:
        # Combine two arrays of initial values together
        val0 = np.array(np.concatenate((x0, l0)).T).flatten()
        valc0 = np.array(np.concatenate((x0, lc0)).T).flatten()

        # Substitute values for symbols
        A = np.matrix(FuncA(*np.array(l0).flatten())).astype(np.double)
        B = np.matrix(FuncB(*np.array(x0).flatten())).astype(np.double)
        F0 = np.matrix(FuncF(*val0)).astype(np.double)
        f = -F0 - A * (l - l0)
        Ac = np.matrix(FuncAc(*np.array(x0).flatten())).astype(np.double)
        C = np.matrix(FuncC(*np.array(x0).flatten())).astype(np.double)
        FC0 = np.matrix(FuncG(*valc0)).astype(np.double)
        fc = -FC0 - Ac * (lc - lc0)
        fx = x0 - lx

        # Solve unknown parameters
        Q = W.I
        Qcc = Wcc.I
        Qe = A * Q * A.T
        Qec = Ac * Qcc * Ac.T
        We = Qe.I
        Wec = Qec.I

        # Compute normal matrices
        N = (B.T * We * B)
        Nc = (C.T * Wec * C)

        # Compute t matrices
        t = (B.T * We * f)
        tc = C.T * Wec * fc

        # Compute the unknown parameters
        X = (N + Nc + Wxx).I * (t + tc - Wxx * fx)

        # Compute residual vectors
        V = A.T * We * (f - B * X)
        Vc = Ac.T * Wec * (fc - C * X)

        # Update initial values
        X = X.astype(np.double)
        x0 += X
        l0 = l + V
        lc0 = lc + Vc

        # Output messages for iteration process
        # print "Iteration count: %d" % counter, u"|ΔX| = %.6f" % abs(X.sum())
        counter += 1    # Update Loop counter

    # Compute residual vectors
    V = A.T * We * (f - B * X)

    # Compute error of unit weight
    s0 = np.double(((V.T * V)[0, 0] / (len(l0) + len(lc)))**0.5)

    # Compute other information
    QXX = (N + Nc + Wxx).I
    SigmaXX = s0**2 * QXX
    param_std = np.sqrt(np.diag(SigmaXX))

    # Output results
    print "Point coordinates:"
    print ("%-8s"+" %-9s"*4) % ("Point", "X", "Y", "SD-X", "SD-Y")
    print ("%-8s"+" %-9.6f"*4) % (
        "Point B", x0[0], x0[1], param_std[0], param_std[1])
    print ("%-8s"+" %-9.6f"*4) % (
        "Point C", x0[2], x0[3], param_std[2], param_std[3])
    print ("%-8s"+" %-9.6f"*4) % (
        "Point D", x0[4], x0[5], param_std[4], param_std[5])
    print ("%-8s"+" %-9.6f"*4) % (
        "Point E", x0[6], x0[7], param_std[6], param_std[7])

    print "\nMeasured distances:"
    print ("%-8s"+" %-9s"*2) % ("Distance", "value", "res")
    for i in range(len(l0)):
        print ("%-8s"+" %-9.6f"*2) % ("s%d" % (i+1), l0[i], V[i, 0])

    print "\nConstraint observations:"
    print ("%-12s"+" %-12s"*2) % ("Constraint", "value", "res")
    print ("%-12s"+" %-12.8f"*2) % (
        "c1", lc0[0, 0], Vc[0, 0])

    print "\nSlope difference between BD and BE: %.6f" \
        % abs((x0[5]-x0[1]) / (x0[4]-x0[0]) - (x0[7]-x0[1]) / (x0[6]-x0[0]))

    print "\nCofactor matrix:"
    print u"QΔΔ =".encode(LANG)
    print QXX
    print ("*" * 26 + "  End  " + "*" * 26)

    print "\nStandard error of unit weight : %.4f" % s0
    print "Degree of freedom: %d" % (len(l0) + len(lc))


def main():
    # Define input values
    # Information of point D and E are inherited from hw7
    Xp = np.array([0, 4, 8, 7.974993, 11.884163])
    Yp = np.array([4, 0, 0, 3.955293, 7.845088])
    l = np.array([12.41, 8.06, 3.87, 8.83, 4.0])

    # Define symbol arrays for unknown parameters and observations
    # Point B, C, D and E are known point having error
    xs = np.array((symbols("Xb Yb Xc Yc Xd Yd Xe Ye")))

    # Add new observation s5, s5 can be calculated from point B and C
    ls = np.array(list(symbols("s1 s2 s3 s4 s5")))

    # Define weight vector
    # Error of s5 can be calculated from error of point B and C
    w = np.array([.02, .02, .02, .02, .014142])
    wxx = np.array([.01, .01, .01, .01, .140239, .139979, .17039, .186464])

    sig0 = .02          # Define a priori error
    # Define weight matrix for observations and unknown parameters
    W = np.matrix(np.diag(sig0**2 / w**2))
    Wxx = np.matrix(np.diag(sig0**2 / wxx**2))

    # For case which has no other information
    print "For case which has no other information..."
    unified(Xp, Yp, l, W, Wxx, xs, ls)

    # Define symbol of constraint
    lcs = np.array([S("c1")])

    # For case in which point B, D and E lie on a straight line
    # Without pseudo unknown parameters
    print "\nFor case in which point B, D and E lie on a straight line..."
    unifiedC(Xp, Yp, l, W, Wxx, xs, ls, lcs)

    return 0


if __name__ == '__main__':
    main()
