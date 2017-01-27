from LaneLine import LaneLine
from Drawing import *
from Color import color
from ImageProcessing import *
from ImageThresholding import *


class LaneDetector(object):
    def __init__(self):
        self.left_lane_line = LaneLine()
        self.right_lane_line = LaneLine()
        self.scale = 4

    def process(self, frame):
        self.warped_frame, self.M_inv = perspective_transform(frame,self.scale)
        self.warped_frame_hsv = bgr2hsv(self.warped_frame)
        self.h,self.s,self.v = split_channels(self.warped_frame_hsv)
        self.s_eq,self.v_eq = equalize_adapthist_channel(self.s,self.v, clip_limit=0.02, nbins=4096, kernel_size=(15,4))
        self.mag_v = mag_grad(self.v_eq, 32, 255, ksize=9)
        self.mag_s = mag_grad(self.s_eq, 32, 255, ksize=9)
        self.mag = AND(self.mag_v, self.mag_s)
        self.s_mask = NOT(binarize_img(self.s_eq, 32, 175))
        self.v_mask = binarize_img(self.v_eq, 160, 255)
        self.mag_and_s_mask = AND(self.mag_v, binarize_img(self.v_eq, 128, 255),self.s_mask)

        center_angle = 0.5
        delta_angle = 0.5
        alpha1 = center_angle - 0.5*delta_angle
        alpha2 = alpha1 + delta_angle
        self.dir = NOT(dir_grad(self.v_eq, alpha1, alpha2, ksize=3))

        self.mag_and_dir = AND(self.mag, self.dir)
        self.detection_input = AND(self.mag_and_s_mask, self.v_mask)

        self.left_lane_line = LaneLine.ExtractLeft(self.detection_input)
        self.right_lane_line = LaneLine.ExtractRight(self.detection_input)



    def annotate(self, frame):
        composite_img = np.zeros_like(self.warped_frame, np.uint8)

        self.fill_lane_area(composite_img)

        if len(self.left_lane_line.current_fit):
            self.draw_lane_line(composite_img, self.left_lane_line)

        if len(self.right_lane_line.current_fit):
            self.draw_lane_line(composite_img, self.right_lane_line)

        transformed_composite_img = inv_perspective_transform(composite_img, self.M_inv, self.scale)
        print(frame.shape, transformed_composite_img.shape)
        annotated_frame = cv2.addWeighted(frame, 1, transformed_composite_img, 0.3, 0)
        warped_annotated_frame = cv2.addWeighted(self.warped_frame, 1, composite_img, 0.3, 0)

        annotated_detection_input = mask_as_image(self.detection_input)
        self.draw_lane_points(annotated_detection_input, self.left_lane_line)
        self.draw_lane_points(annotated_detection_input, self.right_lane_line)
        self.draw_histogram(annotated_detection_input, self.left_lane_line)
        self.draw_histogram(annotated_detection_input, self.right_lane_line)

        return annotated_frame, warped_annotated_frame, annotated_detection_input


    def draw_histogram(self, img, line):
        if not line.histogram is None:
            h,w = img.shape[0:2]
            cv2.polylines(img, [line.histogram], isClosed=False, color=(127,255,127))
            for p in line.peaks:
                draw_marker(img, p)


    def draw_lane_points(self,img, line):
        if line.lane_points is not None:
            for p in line.lane_points:
                draw_pixel(img, p, color=color.pink)


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
