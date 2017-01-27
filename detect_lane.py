import cv2
import Common
import numpy as np
from calibrate_camera import undistort_image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.signal import find_peaks_cwt
from LaneLine import LaneLine
from ImageProcessing import perspective_transform, inv_perspective_transform
from ImageThresholding import binarize_img, normalize_img, grad_x, grad_y, mag_grad, dir_grad
from Color import color
from Drawing import *


def draw_histogram(img, line):
    if not line.histogram is None:
        h,w = img.shape[0:2]
        cv2.polylines(img, [line.histogram], isClosed=False, color=(127,255,127))
        for p in line.peaks:
            draw_marker(img, p)


def draw_lane_points(img, line):
    if line.lane_points is not None:
        for p in line.lane_points:
            draw_pixel(img, p, color=color.pink)


def mask_as_image(m):
    return np.stack((m,m,m),axis=2).astype(np.uint8) * 255


def draw_lane_line(img,lane_line):
    h,w = img.shape[0:2]
    pts = []
    for y in np.arange(0,h+1,h/16):
        x = np.polyval(lane_line.current_fit,y)
        pts.append((x,y))
    coords = np.array(pts, np.int32)
    cv2.polylines(img, [coords], isClosed=False, color=(0,0,255), thickness=4)


def fill_lane_area(img, left, right):
    pts = []
    h,w = img.shape[0:2]
    if len(left.lane_points):
        pts.append((left.lane_points[0][0],h))
        for p in left.lane_points:
            pts.append(p)

    if len(right.lane_points):
        for p in reversed(right.lane_points):
            pts.append(p)
        pts.append((right.lane_points[0][0],h))

    if len(pts):
        cv2.fillPoly(img, np.array([pts],np.int32), (0,255,0))



def detect_lane(orig_img, enhanced_warped_img, warped_img, M_inv):
    left, right = extract_lane_lines(enhanced_warped_img)
    h,w = enhanced_warped_img.shape[0:2]
    composite_img = np.zeros((h,w,3), np.uint8)

    fill_lane_area(composite_img, left, right)

    if len(left.current_fit):
        draw_lane_line(composite_img, left)

    if len(right.current_fit):
        draw_lane_line(composite_img, right)


    transformed_composite_img = inv_perspective_transform(composite_img, M_inv)
    annotated_img = cv2.addWeighted(orig_img, 1, transformed_composite_img, 0.3, 0)
    warped_annotated_img = cv2.addWeighted(warped_img, 1, composite_img, 0.3, 0)


    annotated_enhanced_warped_img = mask_as_image(enhanced_warped_img)
    draw_lane_points(annotated_enhanced_warped_img, left)
    draw_lane_points(annotated_enhanced_warped_img, right)
    draw_histogram(annotated_enhanced_warped_img, left)
    draw_histogram(annotated_enhanced_warped_img, right)

    return annotated_img, warped_annotated_img, annotated_enhanced_warped_img


def enhance_lane_lines(img):
    img = Common.rgb2hls(img)
    gradx = grad_x(img, 37, 255, 25)
    grady = grad_y(img, 13, 255, 31)
    mag = mag_grad(img, 31, 255, 31)
    dir = dir_grad(img, -0.52, 1.17, 31)
    combined = np.zeros_like(dir)
    combined = np.logical_and(gradx,grady).astype(np.uint8)
    #combined[((gradx == 1) & (grady == 1)) | ((mag == 1) & (dir == 1))] = 1
    return combined


def extract_lane_lines(img):
    left_line = LaneLine.ExtractLeft(img)
    right_line = LaneLine.ExtractRight(img)
    return left_line, right_line


# import Common
# img = mpimg.imread('test.png')
# f, ax = plt.subplots(2,3)
# hls_img = Common.rgb2hls(img)
# ax[0][0].imshow(img[:,:,0],cmap="gray")
# ax[0][1].imshow(img[:,:,1],cmap="gray")
# ax[0][2].imshow(img[:,:,2],cmap="gray")
# ax[1][0].imshow(hls_img[:,:,0],cmap="gray")
# ax[1][1].imshow(hls_img[:,:,1],cmap="gray")
# ax[1][2].imshow(hls_img[:,:,2],cmap="gray")



#
#



#
# left = sliding_window(combined, 42)
# right = sliding_window(combined, 218)
# plt.imshow(combined)
# plt.plot(left[:,0], left[:,1], 'ro')
# plt.plot(right[:,0], right[:,1], 'go')
#
#
#
# def calc_radius(array):
#     z = fit_quadratic


if __name__ == '__main__':
    img = Common.load_image('test_images/test1.jpg')
    detect_lane(img)
