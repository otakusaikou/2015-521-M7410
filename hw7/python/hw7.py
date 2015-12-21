#!/usr/bin/env python
# -*- coding: utf-8 -*-
from sympy import symbols, Matrix, lambdify, solve, S
import numpy as np
import sys


LANG = sys.stdout.encoding          # Get system language code
np.set_printoptions(suppress=True)  # Disable scientific notation for numpy


def getPseudoParam(c=1):
    """Return pseudo unknown parameters by given case flag"""
    xs2 = None
    if c == 1:      # Parameter a stands for slope of line BDE
        xs2 = np.array(([S("a")]))

    elif c == 2:    # Parameter b stands for length of BD and DE
        xs2 = np.array((symbols("a b")))

    return xs2


def getInit(Xp, Yp, l, xs):
    """Compute initial values for unknown parameters"""
    # Compute Xd, Yd
    Xd = Xp[2]
    Yd = Yp[2] + Yp[0]

    # Compute Xe, Ye
    Ye = Xp[2] - Xp[1] + xs[1] - 8
    eq = (xs[1] - 8)**2 + Ye**2 - l[3]**2
    Xe = solve(eq)[-1]
    Ye = Ye.subs(xs[1], Xe)

    return (np.matrix([Xd, Yd, Xe, Ye]).T).astype(np.double)


def getPseudoInit(Xp, Yp, x0, c=1):
    """Compute initial values for pseudo unknown parameters"""
    # Point B, D and E lie on a straight line
    if c == 1:
        a = ((x0[1]-Yp[1]) / (x0[0]-Xp[1]))[0, 0]
        return (np.matrix([a]).T).astype(np.double)

    # Point B, D and E lie on a straight line and point D
    # is the midpoint of BE
    elif c == 2:
        a = ((x0[1]-Yp[1]) / (x0[0]-Xp[1]))[0, 0]
        b = (np.sqrt((x0[0]-Xp[1])**2 + (x0[1]-Yp[1])**2))[0, 0]
        return (np.matrix([a, b]).T).astype(np.double)


def getEqn(Xp, Yp, xs, ls):
    """List observation equations"""
    F = Matrix([
        [-((xs[2] - Xp[0])**2 + (xs[3] - Yp[0])**2)**.5 + ls[0]],
        [-((xs[0] - Xp[0])**2 + (xs[1] - Yp[0])**2)**.5 + ls[1]],
        [-((xs[0] - Xp[2])**2 + (xs[1] - Yp[2])**2)**.5 + ls[2]],
        [-((xs[2] - Xp[2])**2 + (xs[3] - Yp[2])**2)**.5 + ls[3]]])

    return F


def getConstraint(Xp, Yp, xs, c=1):
    """List constraint equations"""
    FC = None
    if c == 1:       # Point B, D and E lie on a straight line
        FC = Matrix([
            [(xs[1]-Yp[1]) * (xs[2]-Xp[1]) - (xs[3]-Yp[1]) * (xs[0]-Xp[1])]])

    elif c == 2:     # Point B, D and E lie on a straight line and point D
        FC = Matrix([   # is the midpoint of BE
            [(xs[1]-Yp[1]) * (xs[2]-Xp[1]) - (xs[3]-Yp[1]) * (xs[0]-Xp[1])],
            [xs[0] - ((xs[2] + Xp[1])/2)]])

    return FC


def getPseudoConstraint(Xp, Yp, xs, xs2, c=1):
    """List pseudo constraint equations"""
    # Point B, D and E lie on a straight line
    if c == 1:
        FD = Matrix([
            [xs2[0] * (xs[0]-Xp[1]) - xs[1]],
            [xs2[0] * (xs[2]-Xp[1]) - xs[3]]])

    # Point B, D and E lie on a straight line and point D is the midpoint of BE
    elif c == 2:
        FD = Matrix([
            [xs2[0] * (xs[0]-Xp[1]) - xs[1]],
            [xs2[0] * (xs[2]-Xp[1]) - xs[3]],
            [((xs[0]-Xp[1])**2 + (xs[1]-Yp[1])**2)**.5 - xs2[1]],
            [((xs[2]-xs[0])**2 + (xs[3]-xs[1])**2)**.5 - xs2[1]]])

    return FD


def general(Xp, Yp, l, w, xs, ls):
    """Compute solution for case which has no other information"""
    print ("*" * 25 + "  Start  " + "*" * 25)
    # Compute initial values of unknown parameters
    x0 = getInit(Xp, Yp, l, xs)
    l0 = np.matrix(l).T         # Use result of observations as initial values

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
    lc = 1                                      # Loop counter

    # Iteration process
    while abs(X.sum()) > 10**-15:
        # Combine two arrays of initial values together
        val0 = np.array(np.concatenate((x0, l0)).T).flatten()

        # Substitute values for symbols
        A = np.matrix(FuncA(*np.array(l0).flatten())).astype(np.double)
        B = np.matrix(FuncB(*np.array(x0).flatten())).astype(np.double)
        F0 = np.matrix(FuncF(*val0)).astype(np.double)
        f = -F0 - A * (l - l0)

        # Solve unknown parameters
        sig0 = 1.0                              # Define a priori error
        W = np.matrix(np.diag(sig0**2 / w**2))  # Define weight matrix
        Q = W.I
        Qe = A * Q * A.T
        We = Qe.I
        N = (B.T * We * B)                  # Compute normal matrix
        t = (B.T * We * f)                  # Compute t matrix
        X = N.I * t                         # Compute the unknown parameters
        V = A.T * We * (f - B * X)          # Compute residual vector

        # Update initial values
        x0 += X
        l0 = l + V

        # Output messages for iteration process
        # print "Iteration count: %d" % lc, u"|ΔX| = %.6f" % abs(X.sum())
        lc += 1         # Update Loop counter

    # Compute other information
    QXX = N.I

    # Output results
    print "Point coordinates:"
    print ("%-8s"+" %-9s"*2) % ("Point", "X", "Y")
    print ("%-8s"+" %-9.6f"*2) % (
        "Point D", x0[0], x0[1])
    print ("%-8s"+" %-9.6f"*2) % (
        "Point E", x0[2], x0[3])

    print "\nMeasured distances:"
    print ("%-8s"+" %-9s"*2) % ("Distance", "s", "res")
    for i in range(len(l0)):
        print ("%-8s"+" %-9.6f"*2) % ("s%d" % (i+1), l0[i], 0)

    print "\nCofactor matrix:"
    print u"QΔΔ =".encode(LANG)
    print QXX
    print ("*" * 26 + "  End  " + "*" * 26)


def generalC(Xp, Yp, l, w, xs, ls, c=1):
    """Compute solution for case which which has constraint conditions"""
    print ("*" * 25 + "  Start  " + "*" * 25)

    x0 = getInit(Xp, Yp, l, xs)
    l0 = np.matrix(l).T         # Use result of observations as initial values

    # Get observation equations
    F = getEqn(Xp, Yp, xs, ls)

    # Get constraint equations
    FC = getConstraint(Xp, Yp, xs, c)

    # Compute jacobian matrices
    JFx = F.jacobian(xs)
    JFl = F.jacobian(ls)
    JFCx = FC.jacobian(xs)

    # Create function objects for cofficient matrix A, B, C, f and g matrix
    FuncA = lambdify(tuple(ls), JFl, modules='sympy')
    FuncB = lambdify(tuple(xs), JFx, modules='sympy')
    FuncF = lambdify(tuple(np.append(xs, ls)), F, modules='sympy')
    FuncC = lambdify(tuple(xs), JFCx, modules='sympy')
    FuncG = lambdify(tuple(xs), FC, modules='sympy')

    l = np.matrix(l).T                          # Observations with matrix form
    X = np.ones(1)                              # Initial value for iteration
    lc = 1                                      # Loop counter

    # Iteration process
    while abs(X.sum()) > 10**-14:
        # Combine two arrays of initial values together
        val0 = np.array(np.concatenate((x0, l0)).T).flatten()

        # Substitute values for symbols
        A = np.matrix(FuncA(*np.array(l0).flatten())).astype(np.double)
        B = np.matrix(FuncB(*np.array(x0).flatten())).astype(np.double)
        F0 = np.matrix(FuncF(*val0)).astype(np.double)
        f = -F0 - A * (l - l0)
        C = np.matrix(FuncC(*np.array(x0).flatten())).astype(np.double)
        g = -np.matrix(FuncG(*np.array(x0).flatten())).astype(np.double)

        # Solve unknown parameters
        sig0 = 1.0                              # Define a priori error
        W = np.matrix(np.diag(sig0**2 / w**2))  # Define weight matrix
        Q = W.I
        Qe = A * Q * A.T
        We = Qe.I
        N = (B.T * We * B)                  # Compute normal matrix
        t = (B.T * We * f)
        M = C * N.I * C.T

        # Compute the unknown parameters
        X = N.I * t + N.I * C.T * M.I * (g - C * N.I * t)
        V = W.I * A.T * We * (f - B * X)          # Compute residual vector

        # Update initial values
        x0 += X
        l0 = l + V

        # Output messages for iteration process
        # print "Iteration count: %d" % lc, u"|ΔX| = %.6f" % abs(X.sum())
        lc += 1         # Update Loop counter

    # Compute residual vector
    V = A.T * We * (f - B * X)

    # Compute error of unit weight
    s0 = ((V.T * W * V)[0, 0] / (B.shape[0] - (B.shape[1] - C.shape[0])))**0.5

    # Compute other information
    QXX = (np.identity(N.shape[0]) - N.I * C.T * M.I * C) * N.I
    SigmaXX = s0**2 * QXX
    SigmaVV = (s0**2 * A.T * We * (np.identity(V.shape[0]) - B * N.I * B.T * We
               + B * N.I * C.T * M.I * C * N.I * B.T * We) * A)
    param_std = np.sqrt(np.diag(SigmaXX))
    obs_std = np.sqrt(np.diag(SigmaVV))

    # Output results
    print "Point coordinates:"
    print ("%-8s"+" %-9s"*4) % ("Point", "X", "Y", "SD-X", "SD-Y")
    print ("%-8s"+" %-9.6f"*4) % (
        "Point D", x0[0], x0[1], param_std[0], param_std[1])
    print ("%-8s"+" %-9.6f"*4) % (
        "Point E", x0[2], x0[3], param_std[2], param_std[3])

    print "\nMeasured distances:"
    print ("%-8s"+" %-9s"*3) % ("Distance", "s", "SD", "res")
    for i in range(len(l0)):
        print ("%-8s"+" %-9.6f"*3) % (
            "s%d" % (i+1), l0[i], obs_std[i], V[i, 0])

    print "\nCofactor matrix:"
    print u"QΔΔ =".encode(LANG)
    print QXX
    print ("*" * 26 + "  End  " + "*" * 26)


def generalC2(Xp, Yp, l, w, xs, ls, xs2, c=1):
    """Compute solution for case which which has constraint conditions and"""
    """pseudo unknown parameters"""
    print ("*" * 25 + "  Start  " + "*" * 25)
    x0 = getInit(Xp, Yp, l, xs)
    l0 = np.matrix(l).T         # Use result of observations as initial values
    x20 = getPseudoInit(Xp, Yp, x0, c)

    # Get observation equations
    F = getEqn(Xp, Yp, xs, ls)

    # Get pseudo constraint equations
    FD = getPseudoConstraint(Xp, Yp, xs, xs2, c)

    # Compute jacobian matrices
    JFx = F.jacobian(xs)
    JFl = F.jacobian(ls)
    JFDx = FD.jacobian(xs)
    JFDx2 = FD.jacobian(xs2)

    # Create function objects for cofficient matrix A, B, C, f and g matrix
    FuncA = lambdify(tuple(ls), JFl, modules='sympy')
    FuncB = lambdify(tuple(xs), JFx, modules='sympy')
    FuncF = lambdify(tuple(np.append(xs, ls)), F, modules='sympy')
    FuncD1 = lambdify(tuple(np.append(xs, xs2)), JFDx, modules='sympy')
    FuncD2 = lambdify(tuple(xs), JFDx2, modules='sympy')
    FuncH = lambdify(tuple(np.append(xs, xs2)), FD, modules='sympy')

    l = np.matrix(l).T                          # Observations with matrix form
    X = np.ones(1)                              # Initial value for iteration
    lc = 1                                      # Loop counter

    # Iteration process
    while abs(X.sum()) > 10**-14:
        # Combine two arrays of initial values together
        val0 = np.array(np.concatenate((x0, l0)).T).flatten()
        val20 = np.array(np.concatenate((x0, x20)).T).flatten()

        # Substitute values for symbols
        A = np.matrix(FuncA(*np.array(l0).flatten())).astype(np.double)
        B = np.matrix(FuncB(*np.array(x0).flatten())).astype(np.double)
        F0 = np.matrix(FuncF(*val0)).astype(np.double)
        f = -F0 - A * (l - l0)
        D1 = np.matrix(FuncD1(*np.array(val20).flatten())).astype(np.double)
        D2 = np.matrix(FuncD2(*np.array(x0).flatten())).astype(np.double)
        h = -np.matrix(FuncH(*val20)).astype(np.double)

        # Solve unknown parameters
        sig0 = 1.0                              # Define a priori error
        W = np.matrix(np.diag(sig0**2 / w**2))  # Define weight matrix
        Q = W.I
        Qe = A * Q * A.T
        We = Qe.I
        N = (B.T * We * B)                  # Compute normal matrix
        t = (B.T * We * f)

        P = D1 * N.I * D1.T
        R = D2.T * P.I * D2

        # Compute the unknown parameters
        X2 = R.I * D2.T * P.I * (h - D1 * N.I * t)
        X = N.I * (t + D1.T * P.I * (h - D1 * N.I * t - D2 * X2))
        V = W.I * A.T * We * (f - B * X)          # Compute residual vector

        # Update initial values
        x0 += X
        x20 += X2
        l0 = l + V

        # Output messages for iteration process
        # print "Iteration count: %d" % lc, u"|ΔX| = %.6f" % abs(X.sum())
        lc += 1         # Update Loop counter

    # Compute residual vector
    V = A.T * We * (f - B * X)

    # Compute error of unit weight
    s0 = ((V.T * W * V)[0, 0] / (B.shape[0] - (B.shape[1] - D2.shape[1])))**0.5

    # Compute other information
    QXX = N.I * (
        np.identity(N.shape[0]) - D1.T * P.I * D1 * N.I
        + D1.T * P.I * D2 * R.I * D2.T * P.I * D1 * N.I)
    QXX2 = R.I
    SigmaXX = s0**2 * QXX
    SigmaXX2 = s0**2 * QXX2
    param_std = np.sqrt(np.diag(SigmaXX))
    param2_std = np.sqrt(np.diag(SigmaXX2))

    # Output results
    print "Point coordinates:"
    print ("%-8s"+" %-9s"*4) % ("Point", "X", "Y", "SD-X", "SD-Y")
    print ("%-8s"+" %-9.6f"*4) % (
        "Point D", x0[0], x0[1], param_std[0], param_std[1])
    print ("%-8s"+" %-9.6f"*4) % (
        "Point E", x0[2], x0[3], param_std[2], param_std[3])

    print "\nPseudo parameters:"
    print ("%-8s"+" %-9s"*2) % ("Param.", "value", "SD")
    for i in range(len(xs2)):
        print ("%-8s"+" %-9.6f"*2) % (
            chr(97+i), x20[i], param2_std[i])

    print "\nMeasured distances:"
    print ("%-8s"+" %-9s"*2) % ("Distance", "s", "res")
    for i in range(len(l0)):
        print ("%-8s"+" %-9.6f"*2) % ("s%d" % (i+1), l0[i], V[i, 0])

    print "\nCofactor matrix:"
    print u"QΔΔ =".encode(LANG)
    print QXX
    print u"QΔΔ' =".encode(LANG)
    print QXX2
    print ("*" * 26 + "  End  " + "*" * 26)


def main():
    # Define input values
    Xp = np.array([0, 4, 8])
    Yp = np.array([4, 0, 0])
    l = np.array([12.41, 8.06, 3.87, 8.83])

    # Define weight vectors
    w = np.array(np.ones(len(l)))

    # Define symbol arrays for unknown parameters and observations
    xs = np.array((symbols("Xd Yd Xe Ye")))
    ls = np.array(list(symbols("s1 s2 s3 s4")))

    # For case which has no other information
    print "For case which has no other information..."
    general(Xp, Yp, l, w, xs, ls)

    # For case in which point B, D and E lie on a straight line
    # Without pseudo unknown parameters
    print "\nFor case in which point B, D and E lie on a straight line..."
    generalC(Xp, Yp, l, w, xs, ls, 1)

    # Assume pseudo unknown parameters a
    print "\nFor case in which point B, D and E lie on a straight line.\n" \
          "Assume pseudo unknown parameters..."

    # Define pseudo unknown parameters
    xs2 = getPseudoParam(1)
    generalC2(Xp, Yp, l, w, xs, ls, xs2, 1)

    # For case in which point B, D and E lie on a straight line
    # and point D is the midpoint of BE
    # Without pseudo unknown parameters
    print "\nFor case in which point B, D and E lie on a straight line\n" \
          "and point D is the midpoint of BE..."
    generalC(Xp, Yp, l, w, xs, ls, 2)

    # Assume pseudo unknown parameters a and b
    print "\nFor case in which point B, D and E lie on a straight line\n" \
          "and point D is the midpoint of BE. Assume pseudo unknown\n" \
          "parameters..."

    # Define pseudo unknown parameters
    xs2 = getPseudoParam(2)
    generalC2(Xp, Yp, l, w, xs, ls, xs2, 2)

    return 0


if __name__ == '__main__':
    main()
