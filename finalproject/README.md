Adjustment and Analysis of Spatial Information Final Project
==========

##Description
This project contains the following programs:  
- **SIFTMatching:** Detect conjugate points with SIFT and brute force matching methods. The RANSAC algorithm is applied to extract inliers with homography model.
  Finally the homography matrix is used to warp two images together.
- **imgTrans:** Transform color of input image with information from reference image. Four models are implemented, including 3, 6, 9 and 12 parameter transformation.
- **showResult:** Display color transformation result of two images.
- **sequentialTrans:** Repeat the concept of the program "imgTrans", but solve transformation parameters sequentially and transform partial image areas with corresponding transformation parameters. Only 12 parameter model is implemented, including regular and sequential least square methods.

Sample photos are in the folder "photos".  
The folder "result" contains only part of the result.
