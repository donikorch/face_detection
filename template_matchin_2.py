import cv2 as cv
import matplotlib.pyplot as plt

img_rgb = cv.imread('ORL_Faces/s1/3.pgm')
img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)

template = cv.imread('ORL_Faces/s1/1.pgm', 0)
crop_img = template[30:110, 10:80].copy()
w, h = crop_img.shape[::-1]

res = cv.matchTemplate(img_gray, crop_img, cv.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)

cv.rectangle(img_rgb,top_left, bottom_right, 255, 2)

fig = plt.figure(figsize=(5, 5))
ax1 = plt.subplot(1, 3, 1)
ax2 = plt.subplot(1, 3, 2)

ax1.imshow(crop_img, cmap=plt.cm.gray)
ax1.set_axis_off()
ax1.set_title('template')

ax2.imshow(img_rgb, cmap=plt.cm.gray)
ax2.set_axis_off()
ax2.set_title('image')

plt.show()
