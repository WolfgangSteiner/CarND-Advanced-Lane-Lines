from Common import show_image,load_image
import sys
import cv2
import matplotlib.pyplot as plt
from calibrate_camera import undistort_image

f = sys.argv[1]

img = load_image(f)
img = undistort_image(img)

f, ax = plt.subplots(1,1)
ax.imshow(img)

ax.plot([0, 1280], [450, 450], color='r', linestyle='-', linewidth=1)
ax.plot([0, 1280], [460, 460], color='r', linestyle='-', linewidth=1)
ax.plot([0, 1280], [470, 470], color='r', linestyle='-', linewidth=1)
plt.show()
