import cv2
import numpy as np


def perspective_transform(img,dst_margin_rel=11.0/32.0):
    h,w = img.shape[0:2]

    #y1  = int(450 * factor)
    #x11 = int(595 * factor)
    #x12 = int(681 * factor)

    y1  = 460
    x11 = 580
    x12 = 700

    #y1  = int(470 * factor)
    #x11 = int(565 * factor)
    #x12 = int(717 * factor)

    y2  = 720
    x21 = 216
    x22 = 1092

    dst_width = w
    dst_height = h
    dst_margin = dst_width * dst_margin_rel

    src = np.array([[x11,y1], [x12,y1], [x22,y2], [x21,y2]], np.float32)
    dst = np.array([[dst_margin,0], [dst_width-dst_margin,0], [dst_width-dst_margin,dst_height], [dst_margin,dst_height]], np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, (dst_width,dst_height), flags=cv2.INTER_LINEAR)
    return warped, M_inv


def inv_perspective_transform(img, M_inv):
    h,w = img.shape[0:2]
    return cv2.warpPerspective(img, M_inv, (w,h), flags=cv2.INTER_LINEAR)


def scale_img(img, factor):
    return cv2.resize(img,None,fx=factor,fy=factor, interpolation=cv2.INTER_LINEAR)
