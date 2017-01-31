from Color import color
import numpy as np
import cv2

def put_text(img, text, pos, font=cv2.FONT_HERSHEY_PLAIN, color=color.green, scale=1.0):
    tw,th = cv2.getTextSize(text, font, scale, 1)
    cv2.putText(img, text, (pos[0],pos[1]+2*th), font, scale, color)
    return tw,th

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


class TextRenderer(object):
    def __init__(self, img):
        self.img = img
        self.h, self.w = self.img.shape[0:2]
        self.x = 10
        self.y = 10
        self.spacing = 30
        self.font = cv2.FONT_HERSHEY_DUPLEX
        self.scale = 1.0
        self.color = color.white


    def put_line(self, text):
        put_text(self.img, text, (self.x,self.y), font=self.font, color=self.color, scale=self.scale)
        self.y += self.spacing


    def text_at(self,text,pos):
        put_text(self.img, text, pos, font=self.font, color=self.color, scale=self.scale)
