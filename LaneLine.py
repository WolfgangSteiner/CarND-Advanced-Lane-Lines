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


    def has_good_fit(self):
        return self.best_fit is not None and len(self.best_fit) == 3


    def calc_interpolated_line_points(self, h):
        pts = []
        if self.has_good_fit():
            for y in np.arange(0,h+1,h/16):
                x = poly.polyval(y,self.best_fit)
                pts.append((x,y))

        return np.array(pts, np.int32)


    def detect(self, img, start_x):
        if not start_x is None:
            self.lane_points = LaneLine.sliding_window(img, start_x, self.best_fit)

            if len(self.lane_points) > 3:
                self.current_fit = LaneLine.fit_quadratic(self.lane_points)

        else:
            self.lane_points = []
            self.current_fit = None
        #print(line.lane_points)

        self.update_polynomial()


    def detect_left(self,img):
        h,w = img.shape
        x = (w * 11) // 32
        return self.detect_at(img, x)


    def detect_right(self,img):
        h,w = img.shape
        x = w - (w * 11) // 32
        return self.detect_at(img, x)


    def detect_at(self,img, x):
        self.start_x = None

        if False and self.has_good_fit():
            self.start_x = self.best_fit[0]
            self.peaks = [start_x]
            self.histogram = None
        else:
            self.peaks = []
            h,w = img.shape
            x1 = int(x - w/16)
            x2 = int(x + w/16)
            histogram = np.sum(img[int(img.shape[0]/2):,x1:x2], axis=0)

            start_x = LaneLine.find_peak(histogram)

            if start_x != None:
                self.start_x = start_x + x1
                x_coords = np.arange(x1,x2)
                y_coords = h - histogram
                self.histogram = np.stack((x_coords,y_coords),axis=1).astype(np.int32)
                self.peaks = [np.array((start_x+x1,y_coords[start_x]), np.int32)]


        if self.start_x == None and self.has_good_fit():
            #print(self.best_fit)
            self.start_x = self.best_fit[2]

        if self.start_x != None:
            self.detect(img, self.start_x)



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
    def fit_quadratic(array):
        return poly.polyfit(array[:,1],array[:,0],2)


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
        if self.lane_points is not None:
            for p in self.lane_points:
                draw_pixel(img, p, color=color.pink)


    @staticmethod
    def sliding_window(img, start_x, best_fit):
        h,w = img.shape
        delta_y = h // 16

        if best_fit is not None:
            delta_x = w // 8
        else:
            delta_x = w // 8

        result = []
        y = h - delta_y
        x = start_x
        last_x = None
        last_y = None
        last_dx = None

        dx = 0
        ddx = 0
        #result.append((start_x, h-1))
        while y >= 0:
#            if best_fit is not None:
#                x = np.polyval(best_fit, y)
            x1 = int(max(0,x - delta_x / 2))
            x2 = x1 + delta_x
            histogram = np.sum(img[y:y+delta_y, x1:x2], axis=0)
            peak = LaneLine.find_peak(histogram)
            if peak is not None:
                x = peak + x1
                result.append((x,y))

                if last_x is not None and last_y is not None:
                    dx = (last_x - x) / (last_y - y)

                if last_dx is not None:
                    ddx = (last_dx - dx) /  (last_y - y)

                if last_x is not None:
                    last_dx = dx

                last_x = x
                last_y = y

            #elif dx is not None:
        #        x += dx
        #        dx += ddx

            y -= delta_y

        return np.array(result)
