import cv2
import numpy as np
import glob
from Common import *
import matplotlib.pyplot as plt
import pickle
import os


def calibrate_camera():
    object_points = []
    image_points = []
    for f in glob.glob("camera_cal/calibration*.jpg"):
        print("Processing {}".format(f))
        img = load_img(f)
        img = rgb2gray(img)

        pattern_found, corners = cv2.findChessboardCorners(img, (9,6))
        if pattern_found:
            op = np.zeros((6*9,3), np.float32)
            op[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
            object_points.append(op)
            image_points.append(corners)

    camera_matrix = np.zeros((3,3))
    distortion_coefficients = np.zeros(5)
    cv2.calibrateCamera(object_points, image_points, img.shape[0:2], camera_matrix, distortion_coefficients)
    return camera_matrix, distortion_coefficients


if os.path.exists("camera_calibration.pickle"):
    with open("camera_calibration.pickle", "rb") as f:
        camera_matrix = pickle.load(f)
        distortion_coefficients = pickle.load(f)
else:
    camera_matrix, distortion_coefficients = calibrate_camera()
    with open("camera_calibration.pickle", "wb") as f:
        pickle.dump(camera_matrix, f)
        pickle.dump(distortion_coefficients, f)


def undistort_image(img):
    return cv2.undistort(img, camera_matrix, distortion_coefficients)


if __name__ == '__main__':
    img = load_img("camera_cal/calibration1.jpg")
    img = undistort_image(img)
