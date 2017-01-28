from LaneLine import LaneLine
from Drawing import *
from Color import color
from ImageProcessing import *
from ImageThresholding import *

class LaneDetector(object):
    def __init__(self, pipeline):
        self.left_lane_line = LaneLine()
        self.right_lane_line = LaneLine()
        self.scale = 4
        self.pipeline = pipeline


    def process(self, frame):
        self.warped_frame, self.M_inv = perspective_transform(frame,self.scale)
        self.detection_input = self.pipeline.process(self.warped_frame)

        self.left_lane_line.detect_left(self.detection_input)
        self.right_lane_line.detect_right(self.detection_input)


    def draw_lane_points_and_histogram(self, img):
        self.left_lane_line.draw_lane_points(img)
        self.right_lane_line.draw_lane_points(img)
        self.left_lane_line.draw_histogram(img)
        self.right_lane_line.draw_histogram(img)


    def annotate(self, frame):
        composite_img = np.zeros_like(self.warped_frame, np.uint8)

        self.fill_lane_area(composite_img)

        if len(self.left_lane_line.best_fit):
            self.draw_lane_line(composite_img, self.left_lane_line)

        if len(self.right_lane_line.best_fit):
            self.draw_lane_line(composite_img, self.right_lane_line)

        transformed_composite_img = inv_perspective_transform(composite_img, self.M_inv, self.scale)
        annotated_frame = cv2.addWeighted(frame, 1, transformed_composite_img, 0.3, 0)
        warped_annotated_frame = cv2.addWeighted(self.warped_frame, 1, composite_img, 0.3, 0)

        annotated_detection_input = expand_mask(self.detection_input)
        self.draw_lane_points_and_histogram(annotated_detection_input)
        self.draw_lane_points_and_histogram(warped_annotated_frame)

        return annotated_frame, warped_annotated_frame, annotated_detection_input




    def draw_lane_line(self,img,lane_line):
        h,w = img.shape[0:2]
        pts = []
        coords = lane_line.calc_interpolated_line_points(h)
        thickness = max(1, 4 / self.scale)
        cv2.polylines(img, [coords], isClosed=False, color=color.red, thickness=thickness)


    def fill_lane_area(self, img):
        h,w = img.shape[0:2]
        pts = []
        left_pts = self.left_lane_line.calc_interpolated_line_points(h)
        right_pts = self.right_lane_line.calc_interpolated_line_points(h)
        if len(left_pts) and len(right_pts):
            coords = np.stack((left_pts, right_pts[::-1,:]), axis=0).astype(np.int32).reshape((-1,2))
            cv2.fillPoly(img, [coords], color.green)
