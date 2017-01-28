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


    def process(self, frame):
        self.frame_size = frame.shape[0:2]
        self.dst_margin_abs = int(self.frame_size[1] * self.dst_margin_rel)
        self.warped_frame, self.M_inv = perspective_transform(frame,self.scale,self.dst_margin_rel)
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
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/(self.frame_size) # meters per pixel in x dimension

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        # Now our radius of curvature is in meters
        print(left_curverad, 'm', right_curverad, 'm')
        # Example values: 632.1 m    626.2 m


    def draw_lane_line(self,img,lane_line):
        h,w = img.shape[0:2]
        pts = []
        coords = lane_line.calc_interpolated_line_points(h)
        if coords is not None:
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
