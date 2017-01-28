import cv2
import numpy as np


def perspective_transform(img,scale=1,dst_margin_rel=11.0/32.0):
    h,w = img.shape[0:2]
    factor = w / 1280

    #y1  = int(450 * factor)
    #x11 = int(595 * factor)
    #x12 = int(681 * factor)

    y1  = int(460 * factor)
    x11 = int(580 * factor)
    x12 = int(700 * factor)

    #y1  = int(470 * factor)
    #x11 = int(565 * factor)
    #x12 = int(717 * factor)

    y2  = int(720 * factor)
    x21 = int(216 * factor)
    x22 = int(1092 * factor)
    dst_width = int(w  / scale)
    dst_height = int(h  / scale)
    dst_margin = int(dst_width * dst_margin_rel)
    src = np.array([[x11,y1], [x12,y1], [x22,y2], [x21,y2]], np.float32)
    dst = np.array([[dst_margin,0], [dst_width-dst_margin,0], [dst_width-dst_margin,dst_height], [dst_margin,dst_height]], np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, (dst_width,dst_height), flags=cv2.INTER_LINEAR)
    return warped, M_inv


def inv_perspective_transform(img, M_inv,scale=1):
    h,w = img.shape[0:2]
    return cv2.warpPerspective(img, M_inv, (w*scale,h*scale), flags=cv2.INTER_LINEAR)
