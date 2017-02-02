from imageutils import *
import cv2
import numpy as np
from Color import color
from cv2grid import CV2Grid

dir1 = "output_images/2017-01-31_13-09-53.982742"
dir2 = "output_images/2017-01-31_13-10-03.242910"


def place_img(name, pos, scale=0.25, title=None, title_style="topcenter",dir=dir1):
    img = load_img(name, dir)
    if title == None:
        title = name.replace("_"," ")
    c.paste_img(img, pos, scale, title,title_style=title_style,x_anchor="center")



(w,h) = np.array((1440,2250))
c = CV2Grid(w,h,color=color.white, grid=(32 * 3, 8 * 8-14))
c.set_text_color(color.black)
#c.draw_grid()

x = 16 * 3
y = 0.5
place_img("input_frame", (x,y))
c.arrow([(x,y+4),(x,y+6)])

y+=5
c.arrow([(x,y-1.0),(x,y-0.5)])
c.text_frame((x,y), (-1,1), "camera distortion correction", x_anchor="center", y_anchor="center", scale=1.0)

y+=1
place_img("undistorted_frame", (x,y))
c.arrow([(x,y+4),(x,y+4.5)])
c.text_frame((x,y+5.0), (-1,1), "perspective projection", x_anchor="center", y_anchor="center", scale=1.0)
c.arrow([(x,y+5.5),(x,y+6.0)])

y+=6
place_img("warped_frame", (x,y))
c.arrow([(x,y+4),(x,y+5)])
c.arrow([(x,y+4.5),(x+12.0,y+4.5),(x+12.0,y+5)], start_margin=0)
c.arrow([(x,y+4.5),(x-12.0,y+4.5),(x-12.0,y+5)], start_margin=0)

c.arrow([(x-10.5,y+3),(x-10.5-8,y+3)])
place_img("pipeline/u_minus_v", (x-24,y+2), title="abs(U-V)", scale=1.0, title_style="bottomleft")
c.line([(x-24,y+4),(x-24,y+6)],end_margin=7)
c.arrow([(x-24,y+6),(x-24,y+11.5)], start_margin=7)
y+=5
place_img("pipeline/y", (x-12.0,y), title="Y", scale=1.0, title_style="bottomleft")
place_img("pipeline/u", (x,y), title="U", scale=1.0, title_style="bottomleft")
place_img("pipeline/v", (x+12.0,y), title="V", scale=1.0, title_style="bottomleft")

c.arrow([(x,y+2),(x,y+4)])
c.arrow([(x+12.0,y+2),(x+12.0,y+4)])
c.arrow([(x-12.0,y+2),(x-12.0,y+4)])
c.text_frame((x,y+3.0), (-1,1), "histogram equalization", x_anchor="center", y_anchor="center", scale=1.0, margin=[100,0])

y+=4
place_img("pipeline/y_eq", (x-12.0,y), title="EQ(Y)", scale=1.0, title_style="bottomleft")
place_img("pipeline/u_eq", (x,y), title="EQ(U)", scale=1.0, title_style="bottomleft")
place_img("pipeline/v_eq", (x+12.0,y), title="EQ(V)", scale=1.0, title_style="bottomleft")


c.arrow([(x,y+2),(x,y+2.5)])
c.arrow([(x+12.0,y+2),(x+12.0,y+2.5)])
c.arrow([(x-12.0,y+2),(x-12.0,y+2.5)])
c.text_frame((x,y+2.5), (60,1), "threshold and dilate", x_anchor="center")

y+=4
c.arrow([(x+7,y-0.5),(x+7,y)])
c.arrow([(x+7+11,y-0.5),(x+7+11,y)])
c.arrow([(x+7+22,y-0.5),(x+7+22,y)])
place_img("pipeline/y_y", (x+7,y), title="thres(Y)", scale=1.0, title_style="bottomleft")
place_img("pipeline/y_u", (x+7+11,y), title="thres(U)", scale=1.0, title_style="bottomleft")
place_img("pipeline/y_v", (x+7+22,y), title="thres(V)", scale=1.0, title_style="bottomleft")

c.arrow([(x-29,y-0.5),(x-29,y)])
c.arrow([(x-29+11,y-0.5),(x-29+11,y)])
place_img("pipeline/w_y", (x-29,y), title="thres(Y)", scale=1.0, title_style="bottomleft")
place_img("pipeline/w_uv", (x-29+11,y), title="thres(U-V)", scale=1.0, title_style="bottomleft")

c.arrow([(x-17.5,y-7),(x-40,y-7),(x-40,y)])
c.text_frame((x-40,y-1), (-1,1), "mag. grad.", x_anchor="center", y_anchor="center")
place_img("pipeline/mag_y", (x-40,y), title="mag(Y)", scale=1.0, title_style="bottomleft")

c.arrow([(x+17.5,y-7),(x+40,y-7),(x+40,y)])
c.text_frame((x+40,y-1), (-1,1), "mag. grad.", x_anchor="center", y_anchor="center")
place_img("pipeline/mag_v", (x+40,y), title="mag(V)", scale=1.0, title_style="bottomleft")

y+=2.0
c.arrow([(x-29-11,y),(x-29-11,y+0.5)])
c.arrow([(x-29,y),(x-29,y+0.5)])
c.arrow([(x-29+11,y),(x-29+11,y+0.5)])
c.text_frame((x-29,y+0.5), (33,1), "AND",x_anchor="center")

c.arrow([(x+7,y),(x+7,y+0.5)])
c.arrow([(x+7+11,y),(x+7+11,y+0.5)])
c.arrow([(x+7+22,y),(x+7+22,y+0.5)])
c.arrow([(x+7+33,y),(x+7+33,y+0.5)])
c.text_frame((x+23.5,y+0.5), (44,1), "AND",x_anchor="center")

c.arrow([(x+23.5,y+1.5),(x+23.5,y+3),(x+11.5,y+3)])
c.arrow([(x-29,y+1.5),(x-29,y+3),(x-11.5,y+3)])


y+=2.0
place_img("pipeline/w_mag_y", (x-6,y), scale=1.0, title="white", title_style="bottomleft")
place_img("pipeline/y_mag_v", (x+6,y), scale=1.0, title="yellow", title_style="bottomleft")

y+=2.5
c.arrow([(x+6,y-0.5),(x+6,y)])
c.arrow([(x-6,y-0.5),(x-6,y)])
c.text_frame((x,y), (24,1), "OR", x_anchor="center")

y+=1.5
c.arrow([(x,y-0.5),(x,y)])
place_img("detection_input", (x,y), title="detection input", scale=2.0, title_style="topcenter")

y+=4.5
c.arrow([(x,y-0.5),(x,y)])
c.draw_frame((x,y),(52,5.5), x_anchor="center")
c.text((x,y+0.25), "lane detection", horizontal_align="center")
c.text((x,y+3.0), "OR", horizontal_align="center", vertical_align="center")

place_img("annotated_detection_input", (x-13.5,y+1.0), title="sliding window", scale=2.0, dir=dir1)
place_img("annotated_detection_input", (x+13.5,y+1.0), title="direct fitting", scale=2.0, dir=dir2)

y+=6
c.arrow([(x,y-0.5),(x,y)])
c.text_frame((x,y),(-1,1,), "annotation", x_anchor="center")

y+=1.5
c.arrow([(x,y-0.5),(x,y)])
place_img("annotated_frame", (x,y), title="", scale=0.25)


c.save("fig/pipeline")
