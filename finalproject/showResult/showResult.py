#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import cv2
import matplotlib.pyplot as plt


def showResult(
        trainImg, queryImg, outputImg, RGB1, RGB2, RGB3, startPt, lineLength):
    """Display color transformation result of two images"""
    # Get r, g, b information
    R1, G1, B1 = np.hsplit(RGB1, 3)     # For reference image
    R2, G2, B2 = np.hsplit(RGB2, 3)     # For input image
    R3, G3, B3 = np.hsplit(RGB3, 3)     # For corrected image

    # Create figure
    fig = plt.figure("Result", figsize=(12, 9), dpi=80)

    # Show the image before corrected
    ax1 = fig.add_subplot(331)
    plt.title("Input image", size=15)
    ax1.axis("off")
    ax1.imshow(queryImg)

    # Show reference image
    ax2 = fig.add_subplot(332)
    plt.title("Reference image", size=15)
    ax2.axis([0, trainImg.shape[1], 0, trainImg.shape[0]])
    ax2.axis("off")
    ax2.invert_yaxis()
    ax2.imshow(trainImg)
    ax2.plot(
        range(startPt[1], startPt[1] + lineLength),
        np.ones(lineLength) * startPt[0], "r-")         # Draw a red line

    # Show the image after corrected
    ax3 = fig.add_subplot(333)
    plt.title("Image after correction", size=15)
    ax3.axis("off")
    ax3.imshow(outputImg)

    # Plot r, g, b values as function of column
    # For r, g, b variation plot of input image
    col = range(startPt[1], startPt[1] + lineLength)
    ax4 = fig.add_subplot(334)
    plt.title("Intensity variation plot\n for input image", size=15)
    ax4.set_xlabel("Position (pixel)", size=15)
    ax4.set_ylabel("Intensity", size=15)
    ax4.axis([col[0], col[0] + lineLength, 0, 255])
    ax4.grid()

    ax4.plot(col, R2, "r-")
    ax4.plot(col, G2, "g-")
    ax4.plot(col, B2, "b-")

    # For r, g, b variation plot of reference image
    ax5 = fig.add_subplot(335)
    plt.title("Intensity variation plot\n for reference image", size=15)
    ax5.set_xlabel("Position (pixel)", size=15)
    ax5.set_ylabel("Intensity", size=15)
    ax5.axis([col[0], col[0] + lineLength, 0, 255])
    ax5.grid()

    ax5.plot(col, R1, "r-")
    ax5.plot(col, G1, "g-")
    ax5.plot(col, B1, "b-")

    # For r, g, b variation plot of corrected image
    ax6 = fig.add_subplot(336)
    plt.title("Intensity variation plot\n for corrected image", size=15)
    ax6.set_xlabel("Position (pixel)", size=15)
    ax6.set_ylabel("Intensity", size=15)
    ax6.axis([col[0], col[0] + lineLength, 0, 255])
    ax6.grid()

    ax6.plot(col, R3, "r-")
    ax6.plot(col, G3, "g-")
    ax6.plot(col, B3, "b-")

    # For red profile
    ax7 = fig.add_subplot(337)
    plt.title("Red profile", size=15)
    ax7.set_xlabel("Position (pixel)", size=15)
    ax7.set_ylabel("Intensity", size=15)
    ax7.axis([col[0], col[0] + lineLength, 0, 255])
    ax7.grid()

    ax7.plot(col, R2, "r-")
    ax7.plot(col, R1, "k-")
    ax7.plot(col, R3, "g-")

    # For green profile
    ax8 = fig.add_subplot(338)
    plt.title("Green profile", size=15)
    ax8.set_xlabel("Position (pixel)", size=15)
    ax8.set_ylabel("Intensity", size=15)
    ax8.axis([col[0], col[0] + lineLength, 0, 255])
    ax8.grid()
    plt.ylim([0, 255])

    ax8.plot(col, G2, "r-")
    ax8.plot(col, G1, "k-")
    ax8.plot(col, G3, "g-")

    # For blue profile
    ax9 = fig.add_subplot(339)
    plt.title("Blue profile", size=15)
    ax9.set_xlabel("Position (pixel)", size=15)
    ax9.set_ylabel("Intensity", size=15)
    ax9.axis([col[0], col[0] + lineLength, 0, 255])
    ax9.grid()
    plt.ylim([0, 255])

    l1, = ax9.plot(col, B2, "r-", label="Input image")
    l2, = ax9.plot(col, B1, "k-", label="Reference image")
    l3, = ax9.plot(col, B3, "g-", label="Corrected image")
    plt.legend(handles=[l1, l2, l3], loc=4, fontsize="small")

    fig.tight_layout()
    plt.show()


def main():
    # Define input/output image names and conjugate point
    trainImgName = "01.jpg"                 # Reference image
    queryImgName = "02.jpg"                 # Input image
    outputImgName = "output.jpg"       # The corrected image

    # Define starting point and length of sample line
    startPt = [1045, 1950]
    lineLength = 300

    # Read image and convert them from BGR to RGB
    trainImgRGB = cv2.cvtColor(cv2.imread(trainImgName), cv2.COLOR_BGR2RGB)
    queryImgRGB = cv2.cvtColor(cv2.imread(queryImgName), cv2.COLOR_BGR2RGB)
    outputImgRGB = cv2.cvtColor(cv2.imread(outputImgName), cv2.COLOR_BGR2RGB)

    # Extract color values from image
    RGB1 = trainImgRGB[startPt[0], startPt[1]:startPt[1] + lineLength]
    RGB2 = queryImgRGB[startPt[0], startPt[1]:startPt[1] + lineLength]
    RGB3 = outputImgRGB[startPt[0], startPt[1]:startPt[1] + lineLength]

    showResult(
        trainImgRGB, queryImgRGB, outputImgRGB,
        RGB1, RGB2, RGB3, startPt, lineLength)

    return 0

if __name__ == '__main__':
    main()
