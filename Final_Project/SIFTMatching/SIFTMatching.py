#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import cv2


def warpTwoImages(leftImg, rightImg, M):
    """Warp img2 to img1 with homograph M
    Reference : http://goo.gl/l7Vv3l"""
    # Get the extent of two image
    h1, w1 = leftImg.shape[:2]  # Height and width
    h2, w2 = rightImg.shape[:2]

    # Get four corners
    pts1 = np.float32([[0, 0],  [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    pts2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)

    # Transform the four corners of right image to left image
    pts2_ = cv2.perspectiveTransform(pts2, M)

    # Find new extent after warpping two images
    pts = np.concatenate((pts1, pts2_), axis=0)
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    t = [-xmin, -ymin]

    # Translate matrix
    Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])

    result = cv2.warpPerspective(
        leftImg, Ht.dot(M), (xmax - xmin, ymax - ymin))
    result[t[1]:h1 + t[1], t[0]:w1 + t[0]] = rightImg
    return result


def match(fileName1, fileName2, threshold, show=False, transform=False):
    """SIFT matching with opencv
    Reference : http://goo.gl/70Tk8G"""
    # Read image
    leftImg = cv2.imread(fileName1)
    rightImg = cv2.imread(fileName2)

    # Convert image from bgr to rgb
    leftImgRGB = cv2.cvtColor(leftImg, cv2.COLOR_BGR2RGB)
    rightImgRGB = cv2.cvtColor(rightImg, cv2.COLOR_BGR2RGB)

    # Convert image to gray scale
    leftGray = cv2.cvtColor(leftImg, cv2.COLOR_BGR2GRAY)
    rightGray = cv2.cvtColor(rightImg, cv2.COLOR_BGR2GRAY)

    # Create sift detector object
    sift = cv2.xfeatures2d.SIFT_create()

    # Find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(leftGray, None)
    kp2, des2 = sift.detectAndCompute(rightGray, None)

    # Create Brute-Force matching object with default parameters
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < threshold * n.distance:
            good.append(m)

    # Get coordinates of matching points
    leftPts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    rightPts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # Apply ransac algorithm and homography model to find the inliers
    M, mask = cv2.findHomography(leftPts, rightPts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    # Transform left image with the homography matrix and save the result
    if transform:
        fname1, ext1 = fileName1.split(".")
        fname2, ext2 = fileName2.split(".")
        cv2.imwrite(
            "_".join([fname1, fname2]) + "." + ext1,
            warpTwoImages(leftImg, rightImg, M))

    # Get unuque matching points
    data = np.concatenate((
        leftPts[mask == 1], rightPts[mask == 1]), axis=1)
    uniqueData = np.unique(np.array(
        map(lambda (lx, ly, rx, ry): "%d %d %d %d\n" % (lx, ly, rx, ry),
            data)))

    # Write out results
    fout = open("result_%d.txt" % len(uniqueData), "w")
    for i in range(len(uniqueData)):
        fout.write("%s" % uniqueData[i])
    fout.close()

    print "Number of matching points: %d" % uniqueData.shape[0]

    if show:
        # Draw matching points with green line
        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor = None,
                           matchesMask=matchesMask,
                           flags = 2)

        matchImg = cv2.drawMatches(
            leftImgRGB, kp1, rightImgRGB, kp2, good, None, **draw_params)

        plt.imshow(matchImg, "gray")
        plt.show()


def main():
    match("01.jpg", "02.jpg", 0.8, show=True, transform=True)

    return 0

if __name__ == '__main__':
    main()
