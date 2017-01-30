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
        x_pos = idx // 8 + 6
        y_pos = idx % 8
        if img is not None:
            scale_and_paste(target_frame, img, (x_pos, y_pos), factor=0.125*scale, title=title)

frame_rate = 25
parser = argparse.ArgumentParser()
parser.add_argument('-1', action="store_const", dest="video_file", const="project")
parser.add_argument('-2', action="store_const", dest="video_file", const="challenge")
parser.add_argument('-3', action="store_const", dest="video_file", const="harder_challenge")
parser.add_argument('-d', action="store_const", dest="delay", const=500, default=10)
parser.add_argument('-dd', action="store_const", dest="delay", const=1000, default=10)
parser.add_argument('-t', action="store", dest="t1", default=None, type=str)
parser.add_argument('--render', action="store_true", dest="render")
args = parser.parse_args()

if args.video_file == None:
    args.video_file = "project"

if args.t1 == None:
    if args.video_file == "project":
        args.t1 = 20 * frame_rate
    else:
        args.t1 = 0
else:
    t_array = args.t1.split(".")
    args.t1 = int(t_array[0]) * frame_rate
    if len(t_array) == 2:
        args.t1 += int(t_array[1])

args.video_file += "_video.mp4"

pipeline = YUVPipeline()
detector = LaneDetector(pipeline)

def process_frame(frame):
    global counter
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame = undistort_image(frame)
    frame = scale_img(frame, 0.5)
    new_frame = np.zeros((frame.shape[0],frame.shape[1]//4*5,3),np.uint8)

    if not args.render:
        pipeline.poll()

    detector.process(frame)
    annotated_frame, warped_annotated_frame, annotated_input_img = detector.annotate(frame)
    R_left, R_right = detector.get_radii()
    d,w = detector.calc_distance_from_center()

    scale_and_paste(new_frame, annotated_frame, (0,0), factor=0.75)
    scale_and_paste(new_frame, detector.warped_frame, (0,3), factor=scale*0.25)
    scale_and_paste(new_frame, annotated_input_img,(1,3), factor=scale*0.25)
    scale_and_paste(new_frame, warped_annotated_frame, (2,3), factor=scale*0.25)

    plot_intermediates(new_frame, pipeline, scale=scale)


    new_frame = scale_img(new_frame, 2.0)

    put_text(new_frame, "%02d.%d" % (counter//frame_rate,counter%frame_rate), (0,0), color=color.black)
    put_text(new_frame, "R1 = %5.2fm  R2 = %5.2fm" % (R_left, R_right), (0,15), color=color.black)

    A_left = detector.left_lane_line.best_fit[2] * detector.left_lane_line.pconv[2]
    A_right = detector.right_lane_line.best_fit[2] * detector.right_lane_line.pconv[2]
    put_text(new_frame, "A1 = %f  A2 = %f" % (A_left, A_right), (0,30), color=color.black)

    B_left = detector.left_lane_line.best_fit[1] * detector.left_lane_line.pconv[1]
    B_right = detector.right_lane_line.best_fit[1] * detector.right_lane_line.pconv[1]
    put_text(new_frame, "B1 = %f  B2 = %f" % (B_left, B_right), (0,45), color=color.black)

    C_left = detector.left_lane_line.best_fit[0] * detector.left_lane_line.pconv[0]
    C_right = detector.right_lane_line.best_fit[0] * detector.right_lane_line.pconv[0]
    put_text(new_frame, "C1 = %f  C2 = %f" % (C_left, C_right), (0,60), color=color.black)

    W = C_right - C_left
    put_text(new_frame, "W = %f" % (W), (0,75), color=color.black)

    pos_text = "to the " + ("left" if d <= 0 else "right")
    put_text(new_frame, "Distance from center: %2.2fm %s"  % (abs(d), pos_text), (0,90), color=color.black)

    counter += 1

    if args.render:
        new_frame = bgr2rgb(new_frame)

    return new_frame


clip = VideoFileClip(args.video_file)
counter = 0
frame_skip = 1
start_frame = args.t1
scale = 4
detector.scale = scale
key_wait = args.delay

if args.render:
    out_file_name = args.video_file.split(".")[0] + "_annotated.mp4"
    annotated_clip = clip.fl_image(process_frame)
    annotated_clip.write_videofile(out_file_name, fps=frame_rate, audio=False)
else:
    for frame in clip.iter_frames():
        if (counter % frame_skip) or (counter < start_frame):
            counter += 1
            continue

        new_frame = process_frame(frame)

        cv2.imshow("test",new_frame)
        key = cv2.waitKey(key_wait)
        if key == ord('.'):
            Utils.save_screenshot(new_frame)
        elif key == ord('+'):
            key_wait = max(10, key_wait // 2)
        elif key == ord('-'):
            key_wait = min(2000, key_wait * 2)
        elif key == ord(' '):
            print("PAUSE...")
            while True:
                key = cv2.waitKey(10)
                if key == ord(' '):
                    break
