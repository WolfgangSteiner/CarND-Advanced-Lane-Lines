import cv2
from ImageThresholding import *

class FilterPipeline(object):
    def __init__(self):
        self.intermediates = []

    def process(self,img):
        return img

    def add_intermediate_channel(self, channel, title=None):
        self.intermediates.append((expand_channel(channel),title))

    def add_intermediate_mask(self, mask, title=None):
        self.intermediates.append((expand_mask(mask),title))


class YUVPipeline(FilterPipeline):
    def __init__(self):
        super().__init__()


    def process(self, warped_frame):
        self.intermediates = []
        y,u,v = split_yuv(warped_frame)
        y_eq,u_eq,v_eq = equalize_channel(y,u,v)

        for ch in "y,u,v,y_eq,u_eq,v_eq".split(","):
            self.add_intermediate_channel(eval(ch),ch)


        uv1 = binarize_img(u_eq, 255-16, 255)
        uv2 = binarize_img(v_eq, 0, 1 * 8)
        yellow = cv2.bitwise_and(uv1, uv2)

        uv4 = binarize_img(y_eq, 255-8, 255)
        u_minus_v = abs_diff_channels(u,v)
        uv5 = binarize_img(u_minus_v, 0, 32)
        white = cv2.bitwise_and(uv4,uv5)

        mag_v_eq = mag_grad(v_eq, 32, 255, ksize=31)

        for ch in "yellow,white,u_minus_v,mag_v_eq".split(","):
            self.add_intermediate_mask(eval(ch),ch)

        return AND(OR(white, yellow),mag_v_eq)


class HSVPipeline(FilterPipeline):
    def __init__(self):
        super().__init__()


    def process(self, warped_frame):
        self.intermediates = []
        h,s,v = split_hsv(warped_frame)
        self.add_intermediate_channel(h,"S")
        self.add_intermediate_channel(h,"V")

        s_eq,v_eq = equalize_adapthist_channel(s,v, clip_limit=0.02, nbins=4096, kernel_size=(15,4))
        self.add_intermediate_channel(s_eq,"EQ(S)")
        self.add_intermediate_channel(v_eq,"EQ(V)")


        mag_v = mag_grad(v_eq, 32, 255, ksize=3)
        mag_s = mag_grad(s_eq, 32, 255, ksize=3)
        mag = AND(mag_v, mag_s)
        self.add_intermediate_mask(mag_v,"mag_v")
        self.add_intermediate_mask(mag_s,"mag_s")
        self.add_intermediate_mask(mag,"mag")

        s_mask = NOT(binarize_img(s_eq, 32, 175))
        v_mask = binarize_img(v_eq, 160, 255)
        mag_and_s_mask = AND(mag_v, binarize_img(v_eq, 128, 255),s_mask)
        self.add_intermediate_mask(s_mask,"s_mask")
        self.add_intermediate_mask(v_mask,"v_mask")

        center_angle = 0.5
        delta_angle = 0.5
        alpha1 = center_angle - 0.5*delta_angle
        alpha2 = alpha1 + delta_angle
        dir = NOT(dir_grad(v_eq, alpha1, alpha2, ksize=3))
        mag_and_dir = AND(mag, dir)
        self.add_intermediate_mask(dir,"dir_v")
        self.add_intermediate_mask(mag_and_dir,"dir_v & mag")

        return AND(mag_and_s_mask, v_mask)
