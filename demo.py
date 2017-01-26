from moviepy.editor import VideoFileClip
import cv2
import numpy as np
from ImageProcessing import perspective_transform
from calibrate_camera import undistort_image
from ImageThresholding import *
from detect_lane import detect_lane
import argparse
from Color import color
from Drawing import *

def scale_img(img, factor):
    return cv2.resize(img,None,fx=factor,fy=factor, interpolation=cv2.INTER_CUBIC)


def paste_img(target_img, source_img, pos):
    h,w = source_img.shape[0:2]
    target_img[pos[0]:pos[0]+h,pos[1]:pos[1]+w,:] = source_img


def scale_and_paste(target_frame, img, pos, factor=1.0, title=None):
    img = cv2.resize(img,None,fx=factor,fy=factor, interpolation = cv2.INTER_LINEAR)
    abs_pos = (pos[1] * img.shape[0], pos[0]*img.shape[1])
    paste_img(target_frame, img, abs_pos)
    if not title is None:
        put_text(target_frame, title, (abs_pos[1],abs_pos[0]))


def scale_and_paste_mask(target_frame, mask, pos, factor=1.0, title=None):
    scale_and_paste(target_frame, expand_channel(mask)*255, pos, factor,title=title)


def scale_and_paste_channel(target_frame, channel, pos, factor=1.0, title=None):
    scale_and_paste(target_frame, expand_channel(channel), pos, factor, title=title)


def clip_channel(*imgs, min=0, max=255):
    result = []
    for i in imgs:
        result.append(np.clip(i, min, max))

    if len(result) == 1:
        return result[0]
    else:
        return result


def abs_channel(*imgs):
    result = []
    for i in imgs:
        result.append(np.abs(u-v))

    if len(result) == 1:
        return result[0]
    else:
        return result

def abs_diff_channels(a,b):
    return np.abs(a.astype(np.int32) - b.astype(np.int32)).astype(np.uint8)



parser = argparse.ArgumentParser()
#parser.add_argument('model')
#parser.add_argument('test_data')
parser.add_argument('-1', action="store_const", dest="video_file", const="project")
parser.add_argument('-2', action="store_const", dest="video_file", const="challenge")
parser.add_argument('-3', action="store_const", dest="video_file", const="harder_challenge")
parser.add_argument('-t', action="store", dest="t1", default=None, type=float)
args = parser.parse_args()

if args.video_file == None:
    args.video_file = "project"

if args.t1 == None:
    if args.video_file == "project":
        args.t1 = 9.25
    else:
        args.t1 = 0

args.video_file += "_video.mp4"
clip = VideoFileClip(args.video_file)

dir_grad_min = 0.0
dir_grad_delta = 0.5
dir_grad_ksize = 3

def on_dir_grad_min(val):
    global dir_grad_min
    dir_grad_min = val / 1000.0

def on_dir_grad_delta(val):
    global dir_grad_delta
    dir_grad_delta = val / 1000.0

def on_dir_grad_ksize(val):
    global dir_grad_ksize
    dir_grad_ksize = 1 + 2*val

cv2.namedWindow('test')
#cv2.createTrackbar('dir_grad min', 'test', 0, 2000, on_dir_grad_min)
#cv2.createTrackbar('dir_grad delta', 'test', 500, 2000, on_dir_grad_delta)
#cv2.createTrackbar('dir_grad ksize', 'test', 1, 10, on_dir_grad_ksize)
# Do whatever you want with contours
#cv2.imshow('test', frame)

counter = 0
frame_skip = 1
start_frame = 60 * args.t1
for frame in clip.iter_frames():
    counter += 1
    if counter % frame_skip or counter < start_frame:
        continue

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame = undistort_image(frame)
    frame = scale_img(frame, 0.5)

    warped_img, M_inv = perspective_transform(frame)
    warped_img_yuv = bgr2yuv(warped_img)
    b = enhance_white_yellow(warped_img,min_l=64, min_s=64)
#    b_hls = bgr2hls(b)
#    c = mag_grad(b_hls, 1, 255, ksize=3, ch=2)
#    e = cv2.bitwise_and(c,d)

    y,u,v = split_channels(warped_img_yuv)
    y_divisor = np.clip(255 - y, 1, 255)
    u_minus_v = abs_diff_channels(u,v)
    y_eq,u_eq,v_eq = equalize_channel(y,u,v)

    #mg = mag_grad(warped_img_yuv, 1, 255, ksize=3, ch=0)
    #dg = dir_grad(warped_img_yuv, dir_grad_min, dir_grad_min + dir_grad_delta, ksize=dir_grad_ksize, ch=0)
    #dg = cv2.bitwise_and(mg,dg)

#    u = np.divide(u, y_divisor)*255
#    v = np.divide(v, y_divisor)*255
#    u,v = clip_channel(u,v)
    uv1 = binarize_img(u_eq, 30 * 8, 255)
    uv2 = binarize_img(v_eq, 0, 1 * 8)
    uv3 = cv2.bitwise_and(uv1, uv2)

    uv4 = binarize_img(y_eq, 255-8, 255)
    uv5 = binarize_img(u_minus_v, 0, 32)
    uv6 = cv2.bitwise_and(uv4,uv5)

    detection_input = cv2.bitwise_or(uv3, uv6)

    new_frame = np.zeros_like(frame)

    annotated_frame, warped_annotated_frame, annotated_input_img = detect_lane(frame, detection_input, warped_img, M_inv)
    scale_and_paste(new_frame, annotated_frame, (0,0), factor=0.75)
    #scale_and_paste_mask(new_frame, c,(6,4))

    scale_and_paste_mask(new_frame, uv1, (6,4), factor=0.5)
    scale_and_paste_mask(new_frame, uv2, (6,5), factor=0.5)
    scale_and_paste_mask(new_frame, uv3, (6,6), factor=0.5)

    scale_and_paste_mask(new_frame, uv4, (7,4), factor=0.5)
    scale_and_paste_mask(new_frame, uv5, (7,5), factor=0.5)
    scale_and_paste_mask(new_frame, uv6, (7,6), factor=0.5)
    #scale_and_paste_mask(new_frame, dg,(3,5))
    #scale_and_paste_mask(new_frame, cv2.bitwise_and(uv6,dg),(4,5))

    scale_and_paste(new_frame, warped_img, (0,3))
    #scale_and_paste(new_frame, b, (1,3))
    scale_and_paste_mask(new_frame, detection_input, (1,3))
    scale_and_paste(new_frame, annotated_input_img,(2,3))
    scale_and_paste(new_frame, warped_annotated_frame, (3,3))


    for ch, pos, t in zip((y,u,v), range(0,3), ('YUV')):
        scale_and_paste_channel(new_frame,ch,(6,pos), factor=0.5, title=t)

#    scale_and_paste_channel(new_frame, u_minus_v,(4,3),title='u-v')

    for ch, pos, t in zip((y_eq,u_eq,v_eq), range(0,3), ('YUV')):
        scale_and_paste_channel(new_frame,ch,(7,pos), factor=0.5, title="EQ(%s)"%t)


    put_text(new_frame, "%d:%d.%d" % (counter//3600, counter//60,counter%60), (0,0))

    #scale_and_paste_mask(new_frame, e,(2,3))
    new_frame = scale_img(new_frame, 2.0)
    cv2.imshow("test",new_frame)
    cv2.waitKey(1)
