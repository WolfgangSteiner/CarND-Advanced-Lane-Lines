import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def load_image(img_path):
    return mpimg.imread(img_path)

def save_image(img, img_path):
    cv2.imwrite(img_path, img)

def rgb2gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def rgb2hls(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

def split_channels(img):
    return img[:,:,0], img[:,:,1], img[:,:,2]

def show_image(img):
    f, ax = plt.subplots(1,1)
    ax.imshow(img)
    plt.show()
