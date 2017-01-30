import numpy as np
import numpy.polynomial.polynomial as poly
from scipy.signal import find_peaks_cwt


def rel_change(a,b):
    return np.abs(a - b) / np.max(np.abs(0.5 *(a+b)), 1e-6)


def abs_change(a,b):
    return np.abs(a - b)


def calc_radius(a):
    eps = 1e-5

    if a is None:
        return None
    elif len(a) < 3:
        return 1 / eps
    else:
        A = a[2]
        B = a[1]
        R = np.power(1 + B**2, 1.5) / np.maximum(1e-5, np.abs(A))
        return R

def fit_quadratic(lane_x, lane_y):
    return poly.polyfit(lane_y,lane_x,2)


def find_peak(histogram):
    w = len(histogram)

    if w == 0 or histogram.max() == 0:
        return None

    peaks = find_peaks_cwt(histogram, np.arange(w,4*w))

    if len(peaks) == 0:
        return None

    peak_idx = histogram[peaks].argmax()
    return peaks[peak_idx]


def clip_to_range(a, l1, l2):
    return min(max(a,l1), l2)
