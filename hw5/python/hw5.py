#!/usr/bin/env python
# -*- coding: utf-8 -*-
from sympy import symbols, diff, sin, cos
from math import hypot, atan2
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import sys


LANG = sys.stdout.encoding          # Get system language code
SHOWFIG = True                     # A flag for debugging
np.set_printoptions(suppress=True)  # Disable scientific notation for numpy


def getInit(p, q, P, Q):
    seq = range(0, len(p))  # Create index
    seq.append(seq.pop(0))
    index = zip(range(0, len(p)), seq)

    # Convert input parameters to arrays
    p, q, P, Q = tuple(map(lambda x: np.array(x), [p, q, P, Q]))

    # Get mean value of scale factrs as initial parameter of Sigma
    Sigma = map(
        lambda i: hypot(
            p[i[1]] - p[i[0]],
            q[i[1]] - q[i[0]]) / hypot(
            P[i[1]] - P[i[0]],
            Q[i[1]] - Q[i[0]]),
        index)
    Sigma0 = sum(Sigma) / len(p)

    # Get mean rotate angle as initial parameter of theta
    theta = map(
        lambda i: atan2(
            Q[i[1]] - Q[i[0]],
            P[i[1]] - P[i[0]]) - atan2(
            q[i[1]] - q[i[0]],
            p[i[1]] - p[i[0]]),
        index)
    theta0 = sum(theta) / len(p)

    # Compute initial horizontal and vertical translation
    tp0 = (p - Sigma0 * (P * cos(theta0) + Q * sin(theta0))).mean()
    tq0 = (q - Sigma0 * (P * -sin(theta0) + Q * cos(theta0))).mean()

    return Sigma0, theta0, tp0, tq0


def getPQ(P, Q, Sigma, theta, tp, tq):
    # Define nonlinear transformation model
    p = Sigma * P * cos(theta) + Sigma * Q * sin(theta) + tp
    q = -Sigma * P * sin(theta) + Sigma * Q * cos(theta) + tq
    return p, q


def getPQ2(P, Q, a, b, tp, tq):
    # Define linear transformation model
    p = a * P + b * Q + tp
    q = -b * P + a * Q + tq
    return p, q


def writeMatrix(fout, mat, dig):
    # Get number of row and column
    nrow, ncol = mat.shape

    # Write out matrix values
    for r in range(nrow):
        for c in range(ncol):
            fout.write(("%." + str(dig) + "f  ") % mat[r, c])
        fout.write("\n")
    fout.write("\n")


def nonlinearApproach(p, q, P, Q, W, s):
    # Define symbols
    try:
        ps, qs, Ps, Qs, Sigmas, tps, tqs, thetas, dSigmas, dthetas, dtps, \
            dtqs = symbols(u"p q P Q σ tp tq θ, Δσ, Δθ, Δtp, Δtq".encode(LANG))
    except:
        ps, qs, Ps, Qs, Sigmas, tps, tqs, thetas, dSigmas, dthetas, dtps, \
            dtqs = symbols("p q P Q σ tp tq θ, Δσ, Δθ, Δtp, Δtq")

    # Create equations object for conformal transformation
    equP, equQ = getPQ(Ps, Qs, Sigmas, thetas, tps, tqs)

    # Define initial resolution values, space list and loop counter
    res_old = 1000
    res_new = 10**-2
    lc = 1

    # Define space lists to record intermediate information
    res_list = [res_new]
    dX_list = []

    # Compute initial values of unknown parameters
    Sigma0, theta0, tp0, tq0 = getInit(p, q, P, Q)

    # Iteration process
    while abs(res_new - res_old) > 10**-18:
        # Linearize nonlinear model
        linP = (
            equP + diff(equP, Sigmas) * dSigmas + diff(equP, thetas) *
            dthetas + diff(equP, tps) * dtps + diff(equP, tqs) * dtqs -
            ps).subs([
                (Sigmas, Sigma0),
                (thetas, theta0),
                (tps, tp0),
                (tqs, tq0)])
        linQ = (
            equQ + diff(equQ, Sigmas) * dSigmas + diff(equQ, thetas) *
            dthetas + diff(equQ, tps) * dtps + diff(equQ, tqs) * dtqs -
            qs).subs([
                (Sigmas, Sigma0),
                (thetas, theta0),
                (tps, tp0),
                (tqs, tq0)])

        # Substitute symbols with observation and given values
        eqsP = map(lambda a, b, c: linP.subs([
            (ps, a), (Ps, b), (Qs, c)]), p, P, Q)
        eqsQ = map(lambda a, b, c: linQ.subs([
            (qs, a), (Ps, b), (Qs, c)]), q, P, Q)

        # Compute f matrix for constant term
        f = -np.matrix(
            map(lambda x: x.subs([
                (dSigmas, 0), (dthetas, 0), (dtps, 0), (dtqs, 0)]),
                eqsP + eqsQ)).T

        # Compute coefficient matrix
        B = np.matrix([
            map(lambda d: diff(e, d), [dSigmas, dthetas, dtps, dtqs])
            for e in eqsP + eqsQ]).astype(np.double)

        N = B.T * W * B                     # Compute normal matrix
        t = B.T * W * f                     # Compute t matrix
        X = N.I * t                         # Compute the unknown parameters
        V = f - B * X                       # Compute residual vector

        # Update residual values
        res_old = res_new
        res_new = (V.T * W * V)[0, 0]
        res_list.append(res_new)

        # Update initial values
        Sigma0 += X[0, 0]
        theta0 += X[1, 0]
        tp0 += X[2, 0]
        tq0 += X[3, 0]
        dX_list.append(np.array(X).astype(np.double))

        # Output results
        print ("*" * 10 + "  Iteration count: %d  " + "*" * 10) % lc
        try:
            print u"Δσ: \t%.18f\nΔθ: \t%.18f\nΔtp: \t%.18f\nΔtq: \t%.18f\n"\
                .encode(LANG) % tuple((np.array(X).T)[0])
            print u"σ: \t%.18f\nθ: \t%.18f\ntp: \t%.18f\ntq: \t%.18f\n"\
                .encode(LANG) % (Sigma0, theta0, tp0, tq0)
            print "V.T * P * V = \t\t%.18f" % res_new
            print u"Δ(V.T * P * V) = \t%.18f\n".encode(LANG)\
                % abs(res_new - res_old)
        except:
            print "Δσ: \t%.18f\nΔθ: \t%.18f\nΔtp: \t%.18f\nΔtq: \t%.18f\n"\
                % tuple((np.array(X).T)[0])
            print "σ: \t%.18f\nθ: \t%.18f\ntp: \t%.18f\ntq: \t%.18f\n"\
                % (Sigma0, theta0, tp0, tq0)
            print "V.T * P * V = \t\t%.18f" % res_new
            print "Δ(V.T * P * V) = \t%.18f\n"\
                % abs(res_new - res_old)

        print "*" * 17 + "  End  " + "*" * 18 + "\n"

        # Update loop counter
        lc += 1

    # Compute error of unit weight
    s0 = (res_new / (B.shape[0] - B.shape[1]))**0.5
    print "Error of unit weight : %.4f\n" % s0

    # Compute other informations
    SigmaXX = s**2 * N.I
    SigmaVV = s**2 * (W.I - B * N.I * B.T)
    Sigmallhat = s**2 * B * N.I * B.T

    # Write out sigma matrix results
    fout = open("SigmaMat.txt", "w")
    try:
        fout.write(u"∑ΔΔ = \n".encode(LANG))
        writeMatrix(fout, SigmaXX, 10)

        fout.write(u"∑VV = \n".encode(LANG))
        writeMatrix(fout, SigmaVV, 10)

        fout.write(u"∑llhat = \n".encode(LANG))
        writeMatrix(fout, Sigmallhat, 10)
    except:
        fout.write("∑ΔΔ = \n")
        writeMatrix(fout, SigmaXX, 10)

        fout.write("∑VV = \n")
        writeMatrix(fout, SigmaVV, 10)

        fout.write("∑ll = \n")
        writeMatrix(fout, Sigmallhat, 10)
    fout.close()
    print "Covariance matrices have been written to file: 'SigmaMat.txt'..."

    return res_list, np.array(dX_list)


def linearApproach(p, q, P, Q, W, s):
    # Define symbols
    ps, qs, Ps, Qs, a, b, tp, tq = symbols("p q P Q a b tp tq")

    # Create equations object for conformal transformation
    equP, equQ = getPQ2(Ps, Qs, a, b, tp, tq)
    equP -= ps
    equQ -= qs

    # Substitute symbols with observation and given values
    equP = map(lambda a, b, c: equP.subs([
        (ps, a), (Ps, b), (Qs, c)]), p, P, Q)
    equQ = map(lambda a, b, c: equQ.subs([
        (qs, a), (Ps, b), (Qs, c)]), q, P, Q)

    # Compute f matrix for constant term
    f = -np.matrix(
        map(lambda x: x.subs([
            (a, 0), (b, 0), (tp, 0), (tq, 0)]),
            equP + equQ)).T

    # Compute coefficient matrix
    B = np.matrix([
        map(lambda d: diff(e, d), [a, b, tp, tq])
        for e in equP + equQ]).astype(np.double)

    N = B.T * W * B                     # Compute normal matrix
    t = B.T * W * f                     # Compute t matrix
    X = N.I * t                         # Compute the unknown parameters
    V = f - B * X                       # Compute residual vector
    res = (V.T * W * V)[0, 0]           # Compute residual square

    # Output results
    print "a: \t%.18f\nb: \t%.18f\ntp: \t%.18f\ntq: \t%.18f\n"\
        % tuple((np.array(X).T)[0])
    print "V.T * P * V = \t\t%.18f\n" % res

    # Compute error of unit weight
    s0 = (res / (B.shape[0] - B.shape[1]))**0.5
    print "Error of unit weight : %.4f\n" % s0

    # Compute other informations
    SigmaXX = s**2 * N.I
    SigmaVV = s**2 * (W.I - B * N.I * B.T)
    Sigmallhat = s**2 * B * N.I * B.T

    # Write out sigma matrix results
    fout = open("SigmaMat2.txt", "w")
    try:
        fout.write(u"∑ΔΔ = \n".encode(LANG))
        writeMatrix(fout, SigmaXX, 10)

        fout.write(u"∑VV = \n".encode(LANG))
        writeMatrix(fout, SigmaVV, 10)

        fout.write(u"∑llhat = \n".encode(LANG))
        writeMatrix(fout, Sigmallhat, 10)
    except:
        fout.write("∑ΔΔ = \n")
        writeMatrix(fout, SigmaXX, 10)

        fout.write("∑VV = \n")
        writeMatrix(fout, SigmaVV, 10)

        fout.write("∑ll = \n")
        writeMatrix(fout, Sigmallhat, 10)
    fout.close()
    print "Covariance matrices have been written to file: 'SigmaMat2.txt'..."


def drawFunctionPlot(
        data, title, ylabel, fig, pos, xylim, offset=None, show_plt=False):
    # Create figure
    fig = plt.figure(fig, figsize=(12, 9), dpi=80)
    ax = fig.add_subplot(pos)

    # Enable grid line
    plt.grid()

    # Disable scientific notation of z axis
    ax.get_yaxis().get_major_formatter().set_useOffset(False)

    # Set x and y axis range
    ax.axis(xylim)

    # Set y axis interval and range
    if offset:
        ax.yaxis.set_major_locator(ticker.MultipleLocator(offset))

    # Set title and labels
    plt.title(title, size=12)
    ax.set_xlabel("Iteration times", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    # Plot all input data
    plt.plot(range(8 - len(data), 8), data, 'bo')
    plt.plot(range(8 - len(data), 8), data, 'b-')

    # Adjust subplot layout
    fig.tight_layout()

    # Show plot
    if show_plt is True:
        plt.show()


def main():
    # Define input values
    p = np.array([16.6791, 47.6718, 72.4188, 8.4674, 15.7592, -24.3569])
    q = np.array([16.1734, 58.7223, 20.8377, 103.4796, -15.7964, 2.3997])
    P = np.array([10, 23, 60, -30, 21, -23])
    Q = np.array([20, 71, 45, 98, -10, -8])

    # Draw points
    # drawPt(p, q, P, Q)

    # Define weight matrix
    s = 0.01     # Define a priori error
    W = np.matrix(np.diag(s**2 / 0.01**2 * np.ones(12)))

    # Solve problem with nonlinear approach
    print "Solve problem with nonlinear approach..."
    res_list, dX_arr = nonlinearApproach(p, q, P, Q, W, s)

    # Draw delta X as functions of iteration number
    dX_arr.reshape(len(dX_arr), dX_arr.shape[1])
    dSigma = dX_arr[:, 0]
    dtheta = dX_arr[:, 1]
    dtp = dX_arr[:, 2]
    dtq = dX_arr[:, 3]

    drawFunctionPlot(
        dSigma,
        "Relationship between\nvariation of scale and iteration times",
        "Variation of scale",
        "0", 221, [0, 8, -1 * 10**-5, 9 * 10**-5], 2 * 10**-5)
    drawFunctionPlot(
        dtheta,
        "Relationship between\nvariation of rotate angle and iteration times",
        "Variation of rotate angle (rad)",
        "0", 222, [0, 8, -1 * 10**-5, 9 * 10**-5], 2 * 10**-5)
    drawFunctionPlot(
        dtp,
        "Relationship between\nvariation of horizontal shift and iteration "
        "times",
        "Variation of horizontal shift (m)",
        "0", 223, [0, 8, -3.5 * 10**-3, 1.5 * 10**-3], 10**-3)
    drawFunctionPlot(
        dtq,
        "Relationship between\nvariation of vertical shift and iteration "
        "times",
        "Variation of vertical shift (m)",
        "0", 224, [0, 8, -3.5 * 10**-3, 1.5 * 10**-3], 10**-3)

    # Draw delta residuals as functions of iteration number
    # New residual values divided by the old one
    div = [res_list[i + 1] / res_list[i] for i in range(len(res_list) - 1)]

    drawFunctionPlot(
        res_list,
        "Relationship between\nvariation of residual and iteration times",
        "Variation of residual",
        "1", 211, [-1, 8, 0, 0.012], 0.002)
    drawFunctionPlot(
        div,
        "Relationship between\nvariation of division of residuals and "
        "iteration times",
        "Variation of division of residuals",
        "1", 212, [0, 8, 0, 1.2], 0.2, SHOWFIG)

    # Solve problem with linear approach
    print "\nSolve problem with linear approach..."
    linearApproach(p, q, P, Q, W, s)


if __name__ == '__main__':
    main()
