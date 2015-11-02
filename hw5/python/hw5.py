#!/usr/bin/env python
# -*- coding: utf-8 -*-
from sympy import symbols, diff, sin, cos
from math import hypot, atan2
import matplotlib.pyplot as plt
import numpy as np
import sys


LANG = sys.stdout.encoding    # Get system language code


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

    # Compute initial hoizontal and vertical translation
    tp0 = (p - Sigma0 * (P * cos(theta0) + Q * sin(theta0))).mean()
    tq0 = (q - Sigma0 * (P * -sin(theta0) + Q * cos(theta0))).mean()

    return Sigma0, theta0, tp0, tq0


def getPQ(P, Q, Sigma, theta, tp, tq):
    # Define transformation model
    p = Sigma * P * cos(theta) + Sigma * Q * sin(theta) + tp
    q = -Sigma * P * sin(theta) + Sigma * Q * cos(theta) + tq
    return p, q


def drawPt(p, q, P, Q):
    # Create figure
    fig = plt.figure(0)
    fig.add_subplot(111)

    # Enable grid line
    plt.grid()

    # Plot all points and line
    plt.plot(p[:3], q[:3], 'bo')
    plt.plot(P[:3], Q[:3], 'ro')

    # Show plot
    plt.show()


def main():
    # Define input values
    p = np.array([16.6791, 47.6718, 72.4188, 8.4674, 15.7592, -24.3569])
    q = np.array([16.1734, 58.7223, 20.8377, 103.4796, -15.7964, 2.3997])
    P = np.array([10, 23, 60, -30, 21, -23])
    Q = np.array([20, 71, 45, 98, -10, -8])

    # Define weight matrix
    W = np.diag(np.ones(12))

    # Define symbols
    try:
        ps, qs, Ps, Qs, Sigmas, tps, tqs, thetas, dSigmas, dthetas, dtps, dtqs \
            = symbols(u"p q P Q σ tp tq θ, Δσ, Δθ, Δtp, Δtq".encode(LANG))
    except:
        ps, qs, Ps, Qs, Sigmas, tps, tqs, thetas, dSigmas, dthetas, dtps, dtqs \
            = symbols("p q P Q σ tp tq θ, Δσ, Δθ, Δtp, Δtq")

    # Create equations object for conformal transformation
    equP, equQ = getPQ(Ps, Qs, Sigmas, thetas, tps, tqs)

    # Define initial resolution values, and loop counter
    res_old = 1000
    res_new = 0
    lc = 0

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
        f = -np.matrix(map(lambda x: x.subs([
            (dSigmas, 0), (dthetas, 0), (dtps, 0), (dtqs, 0)]),
            eqsP + eqsQ)).T

        # Compute coefficient matrix
        B = np.matrix([
            map(lambda d: diff(e, d), [dSigmas, dthetas, dtps, dtqs])
            for e in eqsP + eqsQ])

        X = (B.T * W * B).I * (B.T * W * f)  # Compute the unknown parameters
        V = f - B * X                        # Compute residual vector

        # Update residual values
        res_old = res_new
        res_new = V.T * W * V

        # Update initial values
        Sigma0 += X[0, 0]
        theta0 += X[1, 0]
        tp0 += X[2, 0]
        tq0 += X[3, 0]

        # Update loop counter
        lc += 1

        # Output results
        print ("*" * 10 + "  Iteration count: %d  " + "*" * 10) % lc
        try:
            print u"Δσ: \t%.12f\nΔθ: \t%.12f\nΔtp: \t%.12f\nΔtq: \t%.12f\n"\
                .encode(LANG) % tuple((np.array(X).T)[0])
            print u"σ: \t%.12f\nθ: \t%.12f\ntp: \t%.12f\ntq: \t%.12f\n"\
                .encode(LANG) % (Sigma0, theta0, tp0, tq0)
            print "V.T * P * V = \t\t%.12f" % res_new
            print u"Δ(V.T * P * V) = \t%.12f\n".encode(LANG)\
                % (res_old - res_new)
        except:
            print "Δσ: \t%.12f\nΔθ: \t%.12f\nΔtp: \t%.12f\nΔtq: \t%.12f\n"\
                % tuple((np.array(X).T)[0])
            print "σ: \t%.12f\nθ: \t%.12f\ntp: \t%.12f\ntq: \t%.12f\n"\
                % (Sigma0, theta0, tp0, tq0)
            print "V.T * P * V = \t\t%.12f" % res_new
            print "Δ(V.T * P * V) = \t%.12f\n"\
                % (res_old - res_new)

        print "*" * 17 + "  End  " + "*" * 18 + "\n"


if __name__ == '__main__':
    main()
