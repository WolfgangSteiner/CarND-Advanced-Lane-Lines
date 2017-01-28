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
import Utils
import time
from LaneDetector import LaneDetector
from FilterPipeline import HSVPipeline,YUVPipeline

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


def plot_intermediates(target_frame, pipeline, scale=1):
    for idx,(img,title) in enumerate(pipeline.intermediates):
        x_pos = idx // 6 + 6
        y_pos = idx % 6
        scale_and_paste(target_frame, img, (x_pos, y_pos), factor=0.125*scale, title=title)


parser = argparse.ArgumentParser()
#parser.add_argument('model')
#parser.add_argument('test_data')
parser.add_argument('-1', action="store_const", dest="video_file", const="project")
parser.add_argument('-2', action="store_const", dest="video_file", const="challenge")
parser.add_argument('-3', action="store_const", dest="video_file", const="harder_challenge")
parser.add_argument('-d', action="store_const", dest="delay", const=0.5, default=0.0)
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


pipeline = YUVPipeline()
detector = LaneDetector(pipeline)

counter = 0
frame_skip = 1
start_frame = 60 * args.t1
scale = 4
for frame in clip.iter_frames():
    counter += 1
    if counter % frame_skip or counter < start_frame:
        continue

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame = undistort_image(frame)
    frame = scale_img(frame, 0.5)
    new_frame = np.zeros_like(frame)

    detector.process(frame)
    annotated_frame, warped_annotated_frame, annotated_input_img = detector.annotate(frame)

    scale_and_paste(new_frame, annotated_frame, (0,0), factor=0.75)
    scale_and_paste(new_frame, detector.warped_frame, (0,3), factor=scale*0.25)
    scale_and_paste_mask(new_frame, detector.detection_input, (1,3), factor=scale*0.25)
    scale_and_paste(new_frame, annotated_input_img,(2,3), factor=scale*0.25)
    scale_and_paste(new_frame, warped_annotated_frame, (3,3), factor=scale*0.25)

    plot_intermediates(new_frame, pipeline, scale=scale)

    put_text(new_frame, "%02d.%d" % (counter//60,counter%60), (0,0))

    new_frame = scale_img(new_frame, 2.0)
    cv2.imshow("test",new_frame)
    key = cv2.waitKey(10)
    if key == ord(' '):
        Utils.save_screenshot(new_frame)

    if args.delay > 0:
        time.sleep(args.delay)
