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
        self.current_distance_from_center = None
        self.current_lane_width = None


    def process(self, frame):
        if not self.is_initialized:
            self.frame_size = np.array(frame.shape[0:2])
            self.input_frame_size = self.frame_size / self.scale
            h,w = self.input_frame_size

            self.dst_margin_abs = int(w * self.dst_margin_rel)

            # meters per pixel in y dimension
            self.ym_per_px = 21.0 / h

            # meters per pixel in x dimension
            self.xm_per_px = 3.7 / (w - 2.0 * self.dst_margin_abs)

            # anchor point for lane detection
            x_anchor_left = self.dst_margin_abs #+ w // 32
            x_anchor_right = w - self.dst_margin_abs# - w // 32


            self.left_lane_line.initialize(
                self.input_frame_size,
                x_anchor_left,
                self.xm_per_px, self.ym_per_px)

            self.right_lane_line.initialize(
                self.input_frame_size,
                x_anchor_right,
                self.xm_per_px, self.ym_per_px)

            self.is_initialized = True

        self.warped_frame, self.M_inv = perspective_transform(frame,self.dst_margin_rel)
        self.pipeline_input = scale_img(self.warped_frame, 1.0/self.scale)
        self.detection_input = self.pipeline.process(self.pipeline_input)

        self.left_lane_line.fit_lane_line(self.detection_input)
        self.right_lane_line.fit_lane_line(self.detection_input)


    def annotate(self, frame):
        composite_img = np.zeros_like(frame, np.uint8)

        self.fill_lane_area(composite_img)

        if self.left_lane_line.has_good_fit():
            self.draw_lane_line(composite_img, self.left_lane_line)

        if self.right_lane_line.has_good_fit():
            self.draw_lane_line(composite_img, self.right_lane_line)

        transformed_composite_img = inv_perspective_transform(composite_img, self.M_inv)
        annotated_frame = cv2.addWeighted(frame, 1, transformed_composite_img, 0.3, 0)
        warped_annotated_frame = cv2.addWeighted(self.pipeline_input, 1, scale_img(composite_img, 1/self.scale), 0.3, 0)

        annotated_detection_input = expand_mask(self.detection_input)
        annotated_detection_input = self.annotate_lane_lines(annotated_detection_input)
        warped_annotated_frame = self.annotate_lane_lines(warped_annotated_frame)

        return annotated_frame, warped_annotated_frame, annotated_detection_input


    def annotate_lane_lines(self, img):
        self.left_lane_line.draw_lane_points(img)
        self.right_lane_line.draw_lane_points(img)
        self.left_lane_line.draw_histogram(img)
        self.right_lane_line.draw_histogram(img)
        img = self.left_lane_line.annotate_poly_fit(img)
        img = self.right_lane_line.annotate_poly_fit(img)
        return img


    def get_radii(self):
        R_inv_left = self.left_lane_line.inv_radius.value
        R_inv_right = self.right_lane_line.inv_radius.value
        R_left = 1.0 / R_inv_left if R_inv_left is not None else 1.0e5
        R_right = 1.0 / R_inv_right if R_inv_right is not None else 1.0e5
        return R_left, R_right


    def calc_distance_from_center(self):
    #def calc_position_and_lane_width(self):
        d_left = self.left_lane_line.calc_distance_from_center()
        d_right = self.right_lane_line.calc_distance_from_center()

        if d_left == None and d_right == None:
            d = self.last_distance_from_center
            w = 3.7
        elif d_left == None:
            d = self.last_lane_width / 2 - d_right
            w = 3.7
        elif d_right == None:
            d = self.last_lane_width / 2 + d_left
            w = 3.7
        else:
            w = d_right - d_left
            d = w/2 - d_right
            self.last_lane_width = w

        self.last_distance_from_center = d
        if d is None:
            d = 0.0
        return d,w


    def draw_lane_line(self,img,lane_line):
        h,w = img.shape[0:2]
        pts = []
        coords = lane_line.interpolate_line_points(h//self.scale) * self.scale
        if coords is not None:
            thickness = max(1, int(16))
            cv2.polylines(img, [coords], isClosed=False, color=color.red, thickness=thickness)


    def fill_lane_area(self, img):
        h,w = img.shape[0:2]
        pts = []

        left_pts = self.left_lane_line.interpolate_line_points(h//self.scale)
        right_pts = self.right_lane_line.interpolate_line_points(h//self.scale)

        has_left_pts = left_pts is not None and len(left_pts)
        has_right_pts = right_pts is not None and len(right_pts)

        if has_left_pts and has_right_pts:
            coords = np.stack((left_pts, right_pts[::-1,:]), axis=0).astype(np.int32).reshape((-1,2))
            cv2.fillPoly(img, [coords * self.scale], color.green)
