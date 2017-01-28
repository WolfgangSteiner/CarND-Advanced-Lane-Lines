import cv2
from ImageThresholding import *
from MidiControl import MidiManager, MidiControl

class FilterPipeline(MidiManager):
    def __init__(self):
        super().__init__()
        self.intermediates = []

    def process(self,img):
        return img

    def add_intermediate(self, img, title=None):
        self.intermediates.append((img,title))

    def add_intermediate_channel(self, channel, title=None):
        self.intermediates.append((expand_channel(channel),title))

    def add_intermediate_mask(self, mask, title=None):
        self.intermediates.append((expand_mask(mask),title))

    def add_empty_intermediate(self):
        self.intermediates.append((None,None))


class YUVPipeline(FilterPipeline):
    def __init__(self):
        super().__init__()
        self.y_y_min = MidiControl(self, "y_y_min", 70, value=128)
        self.y_u_min = MidiControl(self, "y_u_min", 71, value=100)
        self.y_v_max = MidiControl(self, "y_v_max", 72, value=140)

        self.w_y_min = MidiControl(self, "w_y_min", 73, value=180)
        self.w_uv_max = MidiControl(self, "w_uv_max", 105, value=32)

        self.mag_y_min = MidiControl(self, "mag_y_min", 87, value=16)
        self.mag_y_max = MidiControl(self, "mag_y_max", 111, value=255)
        self.mag_y_ksize = MidiControl(self, "mag_y_ksize", 106, value=3, allowed_values=range(3,33,2))

        self.mag_u_min = MidiControl(self, "mag_u_min", 88, value=0)
        self.mag_u_max = MidiControl(self, "mag_u_max", 112, value=255)
        self.mag_u_ksize = MidiControl(self, "mag_u_ksize", 107, value=5, allowed_values=range(3,33,2))

        self.mag_v_min = MidiControl(self, "mag_v_min", 7, value=0)
        self.mag_v_max = MidiControl(self, "mag_v_max", 116, value=255)
        self.mag_v_ksize = MidiControl(self, "mag_v_ksize", 108, value=5, allowed_values=range(3,33,2))



    def process(self, warped_frame):
        self.intermediates = []
        y,u,v = split_yuv(warped_frame)
        #y_eq,u_eq,v_eq = equalize_adapthist_channel(y,u,v, )
        y_eq,u_eq,v_eq = equalize_adapthist_channel(y,u,v, clip_limit=0.02, nbins=4096, kernel_size=(15,1))

        kernel3 = np.ones((3,3),np.uint8)
        kernel5 = np.ones((5,5),np.uint8)

        #----------- yellow -------------#
        y_y = binarize_img(y_eq, self.y_y_min.value, 255)
        y_u = binarize_img(u_eq, self.y_u_min.value, 255)
        y_v = binarize_img(v_eq, 0, self.y_v_max.value)
        yellow = AND(y_y,y_u,y_v)
        #yellow = cv2.dilate(yellow,kernel5,iterations=1)

        #------------ white -------------#
        w_y = binarize_img(y_eq, self.w_y_min.value, 255)
        u_minus_v = abs_diff_channels(u,v)
        w_uv = binarize_img(u_minus_v, 0, self.w_uv_max.value)
        white = AND(w_y,w_uv)
        #white = cv2.dilate(white,kernel5,iterations=1)

        mag_y_eq = mag_grad(y, self.mag_y_min.value, self.mag_y_max.value, ksize=self.mag_y_ksize.value)
        mag_u_eq = mag_grad(u_eq, self.mag_u_min.value, self.mag_u_max.value, ksize=self.mag_u_ksize.value)
        mag_v_eq = mag_grad(v_eq, self.mag_v_min.value, self.mag_v_max.value, ksize=self.mag_v_ksize.value)

        w_mag_y = AND(white,mag_y_eq)


        for ch in "y,u,v".split(","):
            self.add_intermediate_channel(eval(ch),ch)

        for ch in "y_y,y_u,y_v,yellow".split(","):
            self.add_intermediate_mask(eval(ch),ch)

        self.add_empty_intermediate()

        for ch in "y_eq,u_eq,v_eq,u_minus_v".split(","):
            self.add_intermediate_channel(eval(ch),ch)

        for ch in "w_y,w_uv,white,w_mag_y".split(","):
            self.add_intermediate_mask(eval(ch),ch)

        #self.add_empty_intermediate()

        for ch in "mag_y_eq,mag_u_eq,mag_v_eq".split(","):
            self.add_intermediate_mask(eval(ch),ch)

        return OR(white, yellow)


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
