import cv2
from ImageThresholding import *


def enhance_image(yuv_image):
    y,u,v = split_channels(yuv_image)
    u_minus_v = abs_diff_channels(u,v)
    y_eq,u_eq,v_eq = equalize_channel(y,u,v)

    uv1 = binarize_img(u_eq, 31 * 8, 255)
    uv2 = binarize_img(v_eq, 0, 1 * 8)
    uv3 = cv2.bitwise_and(uv1, uv2)

    uv4 = binarize_img(y_eq, 255-1, 255)
    uv5 = binarize_img(u_minus_v, 0, 16)
    uv6 = cv2.bitwise_and(uv4,uv5)

    return cv2.bitwise_or(uv3, uv6)
