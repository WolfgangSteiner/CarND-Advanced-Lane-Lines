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
        self.frame_size = None
        # relative distance of left/right lane line from
        # left/right edge of bird's eye view
        self.dst_margin_rel = 11.0/32.0
        self.dst_margin_abs = None
        self.is_initialized = False
        self.last_lane_width = 3.7
        self.last_distance_from_center = None


    def process(self, frame):
        if not self.is_initialized:
            self.frame_size = np.array(frame.shape[0:2])
            self.input_frame_size = self.frame_size / self.scale
            self.dst_margin_abs = int(self.input_frame_size[1] * self.dst_margin_rel)

            # meters per pixel in y dimension
            self.ym_per_px = 21.0 / self.input_frame_size[0]

            # meters per pixel in x dimension
            self.xm_per_px = 3.7 / (self.input_frame_size[1] - 2.0 * self.dst_margin_abs)

            # anchor point for lane detection
            x_anchor_left = self.dst_margin_abs
            x_anchor_right = self.input_frame_size[1] - self.dst_margin_abs


            self.left_lane_line.initialize(
                self.input_frame_size,
                x_anchor_left,
                self.xm_per_px, self.ym_per_px)

            self.right_lane_line.initialize(
                self.input_frame_size,
                x_anchor_right,
                self.xm_per_px, self.ym_per_px)

            self.is_initialized = True

        self.warped_frame, self.M_inv = perspective_transform(frame,self.scale,self.dst_margin_rel)
        self.detection_input = self.pipeline.process(self.warped_frame)

        self.left_lane_line.fit_lane_line(self.detection_input)
        self.right_lane_line.fit_lane_line(self.detection_input)


    def draw_lane_points_and_histogram(self, img):
        self.left_lane_line.draw_lane_points(img)
        self.right_lane_line.draw_lane_points(img)
        self.left_lane_line.draw_histogram(img)
        self.right_lane_line.draw_histogram(img)


    def annotate(self, frame):
        composite_img = np.zeros_like(self.warped_frame, np.uint8)

        self.fill_lane_area(composite_img)

        if self.left_lane_line.has_good_fit():
            self.draw_lane_line(composite_img, self.left_lane_line)

        if self.right_lane_line.has_good_fit():
            self.draw_lane_line(composite_img, self.right_lane_line)

        transformed_composite_img = inv_perspective_transform(composite_img, self.M_inv, self.scale)
        annotated_frame = cv2.addWeighted(frame, 1, transformed_composite_img, 0.3, 0)
        warped_annotated_frame = cv2.addWeighted(self.warped_frame, 1, composite_img, 0.3, 0)

        annotated_detection_input = expand_mask(self.detection_input)
        self.draw_lane_points_and_histogram(annotated_detection_input)
        self.draw_lane_points_and_histogram(warped_annotated_frame)

        return annotated_frame, warped_annotated_frame, annotated_detection_input


    def calc_radius(self):
        R_left = self.left_lane_line.calc_radius()
        R_right = self.right_lane_line.calc_radius()
        return R_left, R_right


    def calc_distance_from_center(self):
        d_left = self.left_lane_line.calc_distance_from_center()
        d_right = self.right_lane_line.calc_distance_from_center()

        if d_left == None and d_right == None:
            d = self.last_distance_from_center
        elif d_left == None:
            d = self.last_lane_width / 2 - d_right
        elif d_right == None:
            d = self.last_lane_width / 2 + d_left
        else:
            w = d_right - d_left
            d = 0.5 * (w - d_right + d_left)
            self.last_lane_width = w

        self.last_distance_from_center = d
        return d


    def draw_lane_line(self,img,lane_line):
        h,w = img.shape[0:2]
        pts = []
        coords = lane_line.interpolate_line_points(h)
        if coords is not None:
            thickness = max(1, 4 / self.scale)
            cv2.polylines(img, [coords], isClosed=False, color=color.red, thickness=thickness)



    def fill_lane_area(self, img):
        h,w = img.shape[0:2]
        pts = []
        left_pts = self.left_lane_line.interpolate_line_points(h)
        right_pts = self.right_lane_line.interpolate_line_points(h)
        if len(left_pts) and len(right_pts):
            coords = np.stack((left_pts, right_pts[::-1,:]), axis=0).astype(np.int32).reshape((-1,2))
            cv2.fillPoly(img, [coords], color.green)
