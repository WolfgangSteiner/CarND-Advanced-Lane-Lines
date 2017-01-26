from Color import color
import numpy as np
import cv2

def put_text(img, text, pos, color=color.green):
    tw,th = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1.0, 1)
    cv2.putText(img, text, (pos[0],pos[1]+2*th), cv2.FONT_HERSHEY_PLAIN, 1.0, color)

def draw_marker(img, pos, size=4, color=color.red):
    x = pos[0]
    y = pos[1]
    pts = []
    pts.append((x,y-size/2))
    pts.append((x+size/2,y))
    pts.append((x,y+size/2))
    pts.append((x-size/2,y))
    pts = np.array(pts,np.int32)
    cv2.polylines(img, [pts], isClosed=True, color=color)


def draw_pixel(img, pos, color=color.white):
    img[pos[1],pos[0],:] = color
