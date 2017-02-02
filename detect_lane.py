from moviepy.editor import VideoFileClip
import cv2
import numpy as np
from ImageThresholding import *
import argparse
from Color import color
from Drawing import *
import Utils
import time
from LaneDetector import LaneDetector
from FilterPipeline import HSVPipeline,YUVPipeline
from  imageutils import *


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
parser.add_argument('-s', action="store", dest="scale", default=4, type=int)
parser.add_argument('--render', action="store_true", dest="render")
parser.add_argument('--annotate', action="store_true", dest="annotate")

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

    if not args.render:
        pipeline.poll()

    detector.process(frame)
    detector.annotate(frame)

    if args.render and not args.annotate:
        new_frame = detector.annotated_frame
    else:
        new_frame = np.zeros((frame.shape[0],frame.shape[1]//8*9,3),np.uint8)
        scale_and_paste(new_frame, detector.annotated_frame, (0,0), factor=0.75)
        scale_and_paste(new_frame, detector.pipeline_input, (0,3), factor=scale*0.25)
        scale_and_paste(new_frame, detector.annotated_detection_input,(1,3), factor=scale*0.25)
        scale_and_paste(new_frame, detector.warped_annotated_frame, (2,3), factor=scale*0.25)
        plot_intermediates(new_frame, detector.pipeline, scale=scale)

    #new_frame = scale_img(new_frame, 2.0)
    h,w = new_frame.shape[0:2]
    tr = TextRenderer(new_frame)
    tr.scale=0.75

    R_left, R_right = detector.get_radii()
    R_mean = 2.0 / (1/R_left + 1/R_right)
    d,W = detector.calc_distance_from_center()

    tr.put_line("RL=%7.2fm  RR=%7.2fm  R=%7.2fm W=%2.2fm  POS=%2.2fm" % (R_left, R_right, R_mean, W, d))
    tr.text_at("%02d.%02d" % (counter//frame_rate,counter%frame_rate), (20,h-40))

    counter += 1

    if args.render:
        new_frame = bgr2rgb(new_frame)

    return new_frame


clip = VideoFileClip(args.video_file)
counter = 0
frame_skip = 1
start_frame = args.t1
scale = args.scale
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
            dir_name = detector.save_screenshots()
            save_img(new_frame, "output_frame", dir_name)
            print("Screenshots saved in %s." % dir_name)
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
