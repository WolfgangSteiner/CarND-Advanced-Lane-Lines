## Advanced Lane Finding

### Introduction


### 1. Camera Calibration
The code for the calculation of the distortion matrix of the camera can be found in ```camera_calibration.py```. When loading this module, it checks if camera calibration data is present in ```camera_calibration.pickle```. If the calibration has not yet been calculated, the calibration images in the ```camera_cal``` directory are analyzed to derive the ```camera_matrix```
and the ```distortion_coefficients```.

This code can be found in lines 9-41. It first uses ```cv2.findChessboardCorners``` to detect
chessboard corners in the calibration images and ```cv2.calibrateCamera``` to
calculate the distortion coefficients and the camera matrix.
The calibration data is then saved as a pickle for future use.
An example of an undistorted image can be seen below:

![](output_images/camera_calibration.png)

### 2. Perspective Transform:
In order to calculate the transformation matrix for the perspective transform
I first measured the coordinates of four corner points of a straight lane.
I chose the following coordinates:

| Source        | Destination   |
|:-------------:|:-------------:|
| 595, 450      | 440, 0        |
| 690, 450      | 840, 0        |
| 216, 720      | 440, 720      |
| 1115, 720     | 840, 720      |

The code for the perspective transform can be found in ```perspective_transform.py```. Here, the source and destination points are
used to calculate the transformation matrix M with the function ```cv2.getPerspectiveTransform```. The undistorted input image is then
transformed into a bird's eye view by a call to ```cv2.warpPerspective```. Some example images for the perspective transform are shown below:

![](output_images/perspective_transform.png)

### 3. Computation of a Binary Mask to Identify Lane Line Pixels:
Pleas refer to the following block diagram of the processing pipeline for
the different steps taken to derive a binary image for the lane line detection. All of the images shown in the block diagram can also be found in the ```output_images``` directory.

![](fig/pipeline.png)

As described above, the input image is first undistorted and transformed into a bird's eye view. In order to improve performance of the system the image may then be scaled down (not shown). The scaling factor (a power of two) can be chosen freely from the command line of ```detect_lane.py```. Down-scaling
factors between 1 and 8 have successfully tested and a factor of 4 is used
as a speed/quality compromise in the result videos linked below.

The code of the following steps is contained in the class ```YUVPipline``` in the module ```FilterPipeline.py```. The bird's eye image is transformed into the YUV color space and the channels Y, U, V are contrast enhanced by histogram equalization. Equalization allows for more robust detection of the
white and yellow lanes for different lighting conditions.

In order to create a mask for the white lane lines, the following masks are combined by a logical AND function:
* Thresholding and dilation of the equalized Y channel.
* Thresholding and dilation of the absolute difference of the raw U and V channels.
* Thresholding of the magnitude of the gradient of the non-equalized Y channel.

The thresholded color sub-mask are dilated by a 3x3 kernel in order to achieve a stronger response when combined with the gradient magnitude.

A similar procedure is followed for creating a yellow mask:
* Thresholding and dilation of the equalized Y channel.
* Thresholding and dilation of the equalized U channel.
* Thresholding and dilation of the equalized V channel.
* Thresholding of the magnitude of the gradient of the non-equalized V channel.

The result of these operations are then combined by a bitwise OR into the resulting binary input for lane line detection.


### 4. Lane Line Detection and Polynomial Fitting
The lane line detection and polynomial fitting is handeled by the class ```LaneLine``` defined in ```LaneLine.py```. The class ```LaneDetector``` defined in ```LaneDetector.py``` holds two instances of the ```LaneLine``` class for the left and right lane lines. Each instance has an anchor point that is set to the lower left/right destination coordinates of the perspective transform. Lane detection and fitting is coordinated by the method ```LaneLine.fit_laen_line```.

The first detection is achieved using a sliding window algorithm (method ```LaneLine.perform_sliding_window``` called from ```LaneLine.detect_with_sliding_window```): Starting from the anchor point,  a histogram of a window centered around the anchor point and extending upward to the center of the input frame is computed. The x-coordinate of the highest peak of this histogram is then selected as a starting point for the sliding window algorithm.

All the non-zero pixels of the binary input mask that are contained in the current window are added into a growing list of coordinates. The window is successively moved upwards and is moved sideways by the distance between its center and the computed mean of the contained pixels. This allows the window to follow the curvature of the lane line.

![](fig/sliding_window.png)

After a pass of the window through the image, the collection of encountered positive pixel coordinates is used to fit a second order polynomial (function ```lanemath.fit_quadratic``` called from method ```LaneLine.detect_with_sliding_window```). Every time a fit for the lane line is found and it does not deviate too much from the current best fit (determined by ```LaneLine.is_current_fit_good```) the best fit is updated. This is done by use of a simple low-pass filtering algorithm (```LaneLine.do_update_polynomial```).

When the next frame is processed and a previous "best fit" is present, the sliding window algorithm is bypassed in favor for direct polynomial fitting. This has two advantages: direct fitting is significantly faster and it increases the resilience of the algorithm as the image is searched selectively in the vicinity where the line was present in the last frame.

This algorithm uses the current best fit to select pixels inside of a certain margin ```LaneLine.fit_margin``` around the interpolated coordinates of the
current best fit.

![](fig/direct_fitting.png)
