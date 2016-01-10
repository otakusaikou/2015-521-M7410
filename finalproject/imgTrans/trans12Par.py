#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import cv2


def get12Par(RGB1, RGB2):
    """Compute transformation parameters"""
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

    N = B.T * B
    t = B.T * f
    X = N.I * t     # Solve unknown parameters
    V = B * X - f   # Comptue resudials

    # Compute error of unit weight
    s0 = np.sqrt((V.T*V)[0, 0] / (B.shape[0]-B.shape[1]))

    # Compute other informations
    SigmaXX = s0**2 * N.I
    # SigmaVV = s0**2 * (Sigmall - B * N.I * B.T)
    # Sigmallhat = s0**2 * (Sigmall - SigmaVV)
    param_std = np.sqrt(np.diag(SigmaXX))

    # Output results
    print "\nTransformation parameters:"
    print ("%9s"+" %9s"*2) % ("Parameter", "Value", "Std.")
    for i in range(12):
        print "%-10s %8.4f %9.4f" % (chr(97 + i), X[i, 0], param_std[i])
    print "Standard error of unit weight : %.4f" % s0
    print "Degree of freedom: %d" % (B.shape[0] - B.shape[1])

    return np.array(X.flatten())[0]


def transImg12par(Img, param, outputImgName):
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

    # Merge three bands and save the image
    Img2 = cv2.merge([B2, G2, R2])
    cv2.imwrite(outputImgName, Img2)


def main():
    # Define input/output image names and conjugate point
    trainImgName = "01.jpg"
    queryImgName = "02.jpg"
    pointFile = "result_1046.txt"
    outputImgName = "output.jpg"

    # Read image and convert them from BGR to RGB
    trainImgRGB = cv2.cvtColor(cv2.imread(trainImgName), cv2.COLOR_BGR2RGB)
    queryImgRGB = cv2.cvtColor(cv2.imread(queryImgName), cv2.COLOR_BGR2RGB)

    # Read conjugate point coordinates from file
    fin = open(pointFile)
    data = np.array(
        map(lambda l: l.split(), fin.readlines())).astype(np.double)
    fin.close()
    Lcol, Lrow, Rcol, Rrow = np.hsplit(data, 4)

    # Extract color values from image
    RGB1 = np.array(map(
        lambda l: trainImgRGB[l[0, 1], l[0, 0]],
        np.dstack((Lcol, Lrow))))
    RGB2 = np.array(map(
        lambda l: queryImgRGB[l[0, 1], l[0, 0]],
        np.dstack((Rcol, Rrow))))

    # Compute transformation parameters
    X = get12Par(RGB1, RGB2)

    # Transform image color
    transImg12par(queryImgRGB, X, outputImgName)

    return 0


if __name__ == '__main__':
    main()
