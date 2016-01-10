#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import cv2


def getCoeffConst(RGB1, RGB2):
    """Compute coefficient matrix B and constants matrix f"""
    r1, g1, b1 = np.hsplit(RGB1, 3)

    # Form coefficient matrix B and constants matrix f
    Br = np.array(
        map(lambda (r, g, b): [r, g, b, 1] + [0 for i in range(8)], RGB2))
    Bg = np.array(
        map(lambda (r, g, b): [0, 0, 0, 0, r, g, b, 1, 0, 0, 0, 0], RGB2))
    Bb = np.array(
        map(lambda (r, g, b): [0 for i in range(8)] + [r, g, b, 1], RGB2))
    B = np.matrix(np.concatenate((Br, Bg, Bb))).astype(np.double)
    f = np.matrix(np.concatenate((r1, g1, b1))).astype(np.double)

    return B, f


def addObs(RGB1, RGB2, N0I, t0):
    """Update normal matrix and constants matrix"""
    # Get coefficient matrix and constants matrix
    B, f = getCoeffConst(RGB1, RGB2)

    # Update normal matrix and constants matrix
    Q = np.identity(len(RGB1) * 3)
    N1I = (N0I * (np.identity(N0I.shape[0]) - B.T * (Q+B*N0I*B.T).I * B * N0I))
    t1 = t0 + B.T * f
    X = N1I * t1          # Solve unknown parameters

    # Output results
    print "\nTransformation parameters:"
    print ("%9s"+" %9s") % ("Parameter", "Value")
    for i in range(12):
        print "%-10s %8.4f" % (chr(97 + i), X[i, 0])

    return N1I, t1


def get12Par(Lcol, Lrow, Rcol, Rrow, trainImgRGB, queryImgRGB, end):
    """Compute transformation parameters"""
    # Extract color values from image, the parameter "end" means the end index
    # of observables
    RGB1 = np.array(map(
        lambda (col, row): trainImgRGB[row, col],
        np.dstack((Lcol[0:end], Lrow[0:end]))[0]))
    RGB2 = np.array(map(
        lambda (col, row): queryImgRGB[row, col],
        np.dstack((Rcol[0:end], Rrow[0:end]))[0]))

    # Get coefficient matrix and constants matrix
    B, f = getCoeffConst(RGB1, RGB2)

    N = B.T * B
    t = B.T * f
    X = N.I * t     # Solve unknown parameters
    V = B * X - f   # Comptue resudials

    # Compute error of unit weight
    df = B.shape[0] - B.shape[1]        # Degree of fredom
    s0 = np.sqrt((V.T*V)[0, 0] / df)

    # Compute other informations
    SigmaXX = s0**2 * N.I
    param_std = np.sqrt(np.diag(SigmaXX))

    # Output results
    print "\nTransformation parameters:"
    print ("%9s"+" %9s"*2) % ("Parameter", "Value", "Std.")
    for i in range(12):
        print "%-10s %8.4f %9.4f" % (chr(97 + i), X[i, 0], param_std[i])
    print "Standard error of unit weight : %.4f" % s0
    print "Degree of freedom: %d" % df

    return N.I, t, np.array(X.flatten())[0]


def transImg12par(Img, param):
    """Use given twelve parameters to transform the image color"""
    a, b, c, d, e, f, g, h, i, j, k, l = param
    R, G, B = cv2.split(Img)    # Get R, G, B

    # Transform image color
    R2 = (a * R + b * G + c * B + d)
    G2 = (e * R + f * G + g * B + h)
    B2 = (i * R + j * G + k * B + l)

    # Assign values which is greater than the maxmium value of uint8 type data
    # to 255, and cast color arrays to uint8 type
    R2, G2, B2 = map(
        lambda x: np.clip(x, 0, 255).astype(np.uint8), [R2, G2, B2])

    # Merge three bands and return the array
    Img2 = cv2.merge([B2, G2, R2])

    return Img2


def main():
    # Define input/output image names and conjugate point
    trainImgName = "01.jpg"
    queryImgName = "02.jpg"
    pointFile = "result_1046.txt"
    outputImgName = "output.jpg"

    # Read image and convert them from BGR to RGB
    trainImgRGB = cv2.cvtColor(cv2.imread(trainImgName), cv2.COLOR_BGR2RGB)
    queryImgRGB = cv2.cvtColor(cv2.imread(queryImgName), cv2.COLOR_BGR2RGB)

    # Create a copy of input image
    queryImgRGB2 = queryImgRGB.copy()

    # Read conjugate point coordinates from file
    data = np.genfromtxt(
        pointFile, delimiter=" ", names=["Lcol", "Lrow", "Rcol", "Rrow"])
    data.sort(order=["Rcol", "Rrow", "Lcol", "Lrow"], axis=0)

    Lcol = data["Lcol"]
    Lrow = data["Lrow"]
    Rcol = data["Rcol"]
    Rrow = data["Rrow"]

    # Transform the image sequentially
    offset = 60     # The offset between every transformation
    start = 0       # Start column
    sIdx = 0        # Start index of observables

    # The first transformation
    NI, t, X = get12Par(
        Lcol, Lrow, Rcol, Rrow, trainImgRGB, queryImgRGB, offset)
    queryImgRGB2[:, start:Rcol[offset - 1]] = transImg12par(
        queryImgRGB[:, start:Rcol[offset - 1]], X)

    # Update the start column and index
    start = Rcol[offset - 1]
    sIdx = offset

    # Update observables and transform the image sequentially
    for i in range(offset, len(Lcol) - offset, offset):
        # Extract color values from image
        RGB1 = np.array(map(
            lambda (col, row): trainImgRGB[row, col],
            np.dstack((Lcol[i:i + offset], Lrow[i:i + offset]))[0]))
        RGB2 = np.array(map(
            lambda (col, row): queryImgRGB[row, col],
            np.dstack((Rcol[i:i + offset], Rrow[i:i + offset]))[0]))

        # Update coefficient matrix and constants matrix
        NI, t = addObs(RGB1, RGB2, NI, t)
        X = np.array((NI * t).flatten())[0]
        queryImgRGB2[:, start:Rcol[i + offset]] = transImg12par(
            queryImgRGB[:, start:Rcol[i + offset]], X)

        # Update the start column and index
        start = Rcol[i + offset - 1]
        sIdx = i + offset

    # Transform the left part of image
    RGB1 = np.array(map(
        lambda (col, row): trainImgRGB[row, col],
        np.dstack((Lcol[sIdx:], Lrow[sIdx:]))[0]))
    RGB2 = np.array(map(
        lambda (col, row): queryImgRGB[row, col],
        np.dstack((Rcol[sIdx:], Rrow[sIdx:]))[0]))
    NI, t = addObs(RGB1, RGB2, NI, t)
    X = np.array((NI * t).flatten())[0]
    queryImgRGB2[:, start:] = transImg12par(queryImgRGB[:, start:], X)

    # Output the image
    cv2.imwrite(outputImgName, queryImgRGB2)

    return 0


if __name__ == '__main__':
    main()
