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


parser = argparse.ArgumentParser()
#parser.add_argument('model')
#parser.add_argument('test_data')
parser.add_argument('-1', action="store_const", dest="video_file", const="project")
parser.add_argument('-2', action="store_const", dest="video_file", const="challenge")
parser.add_argument('-3', action="store_const", dest="video_file", const="harder_challenge")
parser.add_argument('-d', action="store_const", dest="delay", const=0.5, default=0.0)
parser.add_argument('-dd', action="store_const", dest="delay", const=1.0, default=0.0)
parser.add_argument('-t', action="store", dest="t1", default=None, type=str)
parser.add_argument('--render', action="store_true", dest="render")
args = parser.parse_args()

if args.video_file == None:
    args.video_file = "project"

if args.t1 == None:
    if args.video_file == "project":
        args.t1 = 9 * 60
    else:
        args.t1 = 0
else:
    t_array = args.t1.split(".")
    args.t1 = int(t_array[0]) * 60
    if len(t_array) == 2:
        args.t1 += int(t_array[1])

args.video_file += "_video.mp4"

pipeline = YUVPipeline()
detector = LaneDetector(pipeline)

counter = 0
frame_skip = 1
start_frame = args.t1
scale = 4


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
    R_left, R_right = detector.calc_radius()
    d = detector.calc_distance_from_center()

    scale_and_paste(new_frame, annotated_frame, (0,0), factor=0.75)
    scale_and_paste(new_frame, detector.warped_frame, (0,3), factor=scale*0.25)
    scale_and_paste(new_frame, annotated_input_img,(1,3), factor=scale*0.25)
    scale_and_paste(new_frame, warped_annotated_frame, (2,3), factor=scale*0.25)

    plot_intermediates(new_frame, pipeline, scale=scale)

    put_text(new_frame, "%02d.%d" % (counter//60,counter%60), (0,0))
    put_text(new_frame, "R1 = %5.2fm  R2 = %5.2fm" % (R_left, R_right), (0,20))

    pos_text = "to the " + ("left" if d <= 0 else "right")
    put_text(new_frame, "Distance from center: %2.2fm %s"  % (abs(d), pos_text), (0,35))

    counter += 1
    new_frame = scale_img(new_frame, 2.0)

    if args.render:
        new_frame = bgr2rgb(new_frame)

    return new_frame



clip = VideoFileClip(args.video_file)

if args.render:
    out_file_name = args.video_file.split(".")[0] + "_annotated.mp4"
    annotated_clip = clip.fl_image(process_frame)
    annotated_clip.write_videofile(out_file_name, audio=False)
else:
    for frame in clip.iter_frames():
        counter += 1
        if counter % frame_skip or counter < start_frame:
            continue

        new_frame = process_frame(frame)

        cv2.imshow("test",new_frame)
        key = cv2.waitKey(10)
        if key == ord(' '):
            Utils.save_screenshot(new_frame)

        if args.delay > 0:
            time.sleep(args.delay)
