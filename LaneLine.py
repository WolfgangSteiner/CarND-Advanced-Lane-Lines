import numpy as np
from scipy.signal import find_peaks_cwt
from Drawing import *
import numpy.polynomial.polynomial as poly

# Define a class to receive the characteristics of each line detection
class LaneLine(object):
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients averaged over the last n iterations in meters
        self.best_fit_meters = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line points
        self.lane_points = None

        self.histogram = None

        self.peaks = None

        # conversion factors for poly coefficients from pixels to meters
        self.pconv = np.ones(3)


    def initialize(self, frame_size, x_anchor, xm_per_px, ym_per_px):
        self.frame_size = frame_size
        self.x_anchor = x_anchor
        self.pconv[0] = xm_per_px
        self.pconv[1] = xm_per_px / ym_per_px
        self.pconv[2] = xm_per_px / ym_per_px**2


    def has_good_fit(self):
        return self.best_fit is not None and len(self.best_fit) == 3


    def interpolate_line_points(self, h):
        y = np.arange(0,h+1,h/16)
        x = poly.polyval(y,self.best_fit)
        return np.stack((x,h - y), axis=1).astype(np.int32)


    def detect(self, img, start_x):
        if not start_x is None:
            lane_x,lane_y = self.sliding_window(img, start_x, self.best_fit)
            self.lane_points = np.stack((lane_x,lane_y),axis=1)

            if len(self.lane_points) > 3:
                self.current_fit = LaneLine.fit_quadratic(lane_x, lane_y)
                #print(self.current_fit)

        else:
            self.lane_points = []
            self.current_fit = None
        #print(line.lane_points)


    def fit_lane_line(self,img):
        h,w = img.shape
        self.peaks = []

        if self.has_good_fit():
            nonzero = img.nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])

            margin = w // 16
            x_poly = poly.polyval(h - nonzeroy, self.best_fit)
            left_edge = x_poly - margin // 2
            right_edge = x_poly + margin // 2

            ids = ((nonzerox >= left_edge) & (nonzerox <= right_edge)).nonzero()[0]
            lane_x = nonzerox[ids]
            lane_y = h - nonzeroy[ids]
            self.current_fit = LaneLine.fit_quadratic(lane_x, lane_y)
            self.lane_points = np.stack((lane_x,lane_y),axis=1)

            self.peaks = []
            self.histogram = None
        else:
            x = self.x_anchor
            x1 = int(x - w/8)
            x2 = int(x + w/8)
            histogram = np.sum(img[int(img.shape[0]/2):,x1:x2], axis=0)
            start_x = LaneLine.find_peak(histogram)

            if start_x != None:
                self.start_x = start_x + x1
                x_coords = np.arange(x1,x2)
                y_coords = h - histogram
                self.histogram = np.stack((x_coords,y_coords),axis=1).astype(np.int32)
                self.peaks = [np.array((start_x+x1,y_coords[start_x]), np.int32)]

            if self.start_x != None:
                self.detect(img, self.start_x)

        self.update_polynomial()


    def calc_radius(self):
        if self.has_good_fit():
            A = self.best_fit[2] * self.pconv[2]
            B = self.best_fit[1] * self.pconv[1]
            R = pow(1 + B**2, 1.5) / max(1e-5, abs(A))
            return R
        else:
            return NULL


    def calc_distance_from_center(self):
        if self.has_good_fit():
            d_in_px = self.best_fit[0] - self.frame_size[1] / 2
            d_in_m = d_in_px * self.pconv[0]
            #print(d_in_m)
            return d_in_m
        else:
            return None


    def update_polynomial(self):
        a = 0.5
        b = 1.0 - a
        if self.current_fit is None:
            return
        elif not self.has_good_fit():
            self.best_fit = self.current_fit
        else:
            for i in range(len(self.best_fit)):
                self.best_fit[i] = a * self.current_fit[i] + b * self.best_fit[i]


    @staticmethod
    def fit_quadratic(lane_x, lane_y):
        return poly.polyfit(lane_y,lane_x,2)


    @staticmethod
    def find_peak(histogram):
        w = len(histogram)

        if w == 0 or histogram.max() == 0:
            return None

        peaks = find_peaks_cwt(histogram, np.arange(w,4*w))

        if len(peaks) == 0:
            return None

        peak_idx = histogram[peaks].argmax()
        return peaks[peak_idx]


    def draw_histogram(self, img):
        if not self.histogram is None:
            h,w = img.shape[0:2]
            cv2.polylines(img, [self.histogram], isClosed=False, color=color.light_green)
            for p in self.peaks:
                draw_marker(img, p)


    def draw_lane_points(self,img):
        h,w = img.shape[0:2]
        if self.lane_points is not None:
            for (x,y) in self.lane_points:
                draw_pixel(img, (x,h-y-1), color=color.pink)


    def sliding_window(self, img, start_x, best_fit):
        h,w = img.shape

        delta_y = h // 16
        delta_x = w // 16

        lane_x = []
        lane_y = []

        y = h
        x = start_x

        while y > 0:
            y1 = y - delta_y
            y2 = y
            x1 = int(max(0, x - delta_x / 2))
            x2 = int(min(w, x1 + delta_x))
            window = img[y1:y2,x1:x2]
            nonzero = window.nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])

            if len(nonzerox > 50):
                x = np.int(nonzerox.mean() + x1)

            lane_x.append(nonzerox + x1)
            lane_y.append(nonzeroy + y1)

            y -= delta_y

        return np.concatenate(lane_x), h - np.concatenate(lane_y)
