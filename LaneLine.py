import numpy as np
from scipy.signal import find_peaks_cwt
from Drawing import *

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


    @staticmethod
    def Extract(img, start_x):
        line = LaneLine()
        if not start_x is None:
            lane_points = LaneLine.sliding_window(img, start_x)
            line.lane_points = lane_points

            if len(line.lane_points) > 2:
                line.current_fit = LaneLine.fit_quadratic(lane_points)

        else:
            line.lane_points = []
        #print(line.lane_points)
        return line


    @staticmethod
    def ExtractLeft(img):
        h,w = img.shape
        return LaneLine.ExtractAt(img, w * 0.25)


    @staticmethod
    def ExtractRight(img):
        h,w = img.shape
        return LaneLine.ExtractAt(img, w * 0.75)


    @staticmethod
    def ExtractAt(img, x):
        h,w = img.shape
        x1 = int(x - w/8)
        x2 = int(x + w/8)
        histogram = np.sum(img[int(img.shape[0]/2):,x1:x2], axis=0)
        #peaks = np.array(find_peaks_cwt(histogram, np.arange(w/16,w/8)))
        start_x = histogram.argmax() + x1
        peaks = [start_x]

        line = LaneLine.Extract(img, start_x)

        x_coords = np.arange(x1,x2)
        y_coords = h - histogram
        line.histogram = np.stack((x_coords,y_coords),axis=1).astype(np.int32)

        line.peaks = []
        for px in peaks:
            line.peaks.append(np.array((px,y_coords[px-x1]), np.int32))

        return line



    @staticmethod
    def fit_quadratic(array):
        return np.poly1d(np.polyfit(array[:,1],array[:,0],2))


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

    @staticmethod
    def draw_histogram(img,histogram, x1, y1):
        for i in range(len(histogram)):
            l = 16 * histogram[i]
            draw_pixel(img, (x1+i,y1), color=l)


    @staticmethod
    def sliding_window(img, start_x):
        h,w = img.shape
        delta_y = h // 16
        delta_x = w // 8
        result = []
        y = h - delta_y
        x = start_x
        last_x = None
        last_y = None
        last_dx = None

        dx = 0
        ddx = 0
        while y >= 0:
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

            y -= delta_y / 2

        return np.array(result)
