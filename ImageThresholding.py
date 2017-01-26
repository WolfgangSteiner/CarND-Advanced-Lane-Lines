import numpy as np
import cv2


def binarize_img(img, min_thres, max_thres):
    result = np.zeros_like(img)
    result[(img >= min_thres) & (img <= max_thres)] = 1
    return result.astype(np.uint8)


def bgr2hls(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HLS)


def hls2bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_HLS2BGR)


def split_hls(img):
    return split_channels(bgr2hls(img))


def combine_hls(h,l,s):
    return hls2bgr(combine_channels(h,l,s))


def bgr2yuv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2YUV)


def yuv2bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_YUV2BGR)


def split_yuv(img):
    return split_channels(bgr2yuv(img))


def combine_hls(h,l,s):
    return hls2bgr(np.stack((h,l,s),axis=2))


def split_channels(img):
    return img[:,:,0], img[:,:,1], img[:,:,2]


def combine_channels(a,b,c):
    return np.stack((a,b,c),axis=2)


def expand_channel(c):
    return np.stack((c,c,c),axis=2).astype(np.uint8)


def equalize_channel(*args):
    results = []
    for c in args:
        results.append(cv2.equalizeHist(c))

    if len(results) == 1:
        return results[0]
    else:
        return results


def hls_mask(img, min_h, max_h, min_l, max_l, min_s, max_s):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    return cv2.inRange(hls, np.array([min_h,min_l,min_s]), np.array([max_h,max_l,max_s]))


def mask_hls(img, min_h, max_h, min_l, max_l, min_s, max_s):
    mask = hls_mask(img, min_h, max_h, min_l, max_l, min_s, max_s)
    return apply_mask(img, hls_mask)


def apply_mask(img, mask):
    return cv2.bitwise_and(img,img,mask=mask)


def equalize(img):
    h,l,s = split_hls(img)
    l = cv2.equalizeHist(l)
    return combine_hls(h,l,s)


def normalize_img(img):
    return img/np.max(img)


def grad_x(img, min_thres, max_thres, ksize=3, ch=1):
    sobel = cv2.Sobel(img[:,:,ch], cv2.CV_64F, 1, 0, ksize=ksize)
    sobel = np.absolute(sobel)
    sobel = np.uint8(255*sobel/np.max(sobel))
    return binarize_img(sobel, min_thres, max_thres)


def grad_y(img, min_thres, max_thres, ksize=3, ch=1):
    sobel = cv2.Sobel(img[:,:,ch], cv2.CV_64F, 0, 1, ksize=ksize)
    sobel = np.absolute(sobel)
    sobel = np.uint8(255*sobel/np.max(sobel))
    return binarize_img(sobel, min_thres, max_thres)


def mag_grad(img, min_thres, max_thres, ksize=3, ch=1):
    gray = img[:,:,ch]
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    mag = np.sqrt(np.square(sobelx) + np.square(sobely))
    scaled_mag = np.uint8(255*mag/np.max(mag))
    return binarize_img(scaled_mag, min_thres, max_thres)


def dir_grad(img, min_thres, max_thres, ksize=3, ch=1):
    img_ch = img[:,:,ch]
    sobelx = cv2.Sobel(img_ch, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(img_ch, cv2.CV_64F, 0, 1, ksize=ksize)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    dir = np.absolute(np.arctan2(sobely,sobelx))
    return binarize_img(dir, min_thres * np.pi, max_thres * np.pi)


def and_images(img_a, img_b):
    return np.logical_and(img_a, img_b).astype(np.uint8)


def enhance_white_yellow(img, min_l=116, min_s=80):
    yello = hls_mask(img, 13, 24, min_l, 255, min_s, 255)
    white = hls_mask(img, 0, 180, 192, 255, 0, 255)
    mask = cv2.bitwise_or(yello, white)
    return apply_mask(img, mask)
