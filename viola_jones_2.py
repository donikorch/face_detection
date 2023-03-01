import cv2 as cv
import matplotlib.pyplot as plt
import os

cascPathface = os.path.dirname(
    cv.__file__) + "/data/haarcascade_frontalface_default.xml"
cascPatheyes = os.path.dirname(
    cv.__file__) + "/data/haarcascade_eye_tree_eyeglasses.xml"

img = cv.imread('ORL_Faces/s1/1.pgm')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

face_cascade = cv.CascadeClassifier(cascPathface)

faces = face_cascade.detectMultiScale(gray,
                                      scaleFactor=1.1,
                                      minNeighbors=1,
                                      flags=cv.CASCADE_SCALE_IMAGE,
                                      minSize=(70, 70))

for (x, y, w, h) in faces:
    cv.rectangle(img, (x, y), (x + w, y + h),(0,255,0), 1)

fig = plt.figure(figsize=(7, 5))
ax1 = plt.subplot(1, 3, 1)

ax1.imshow(img, cmap=plt.cm.gray)
ax1.set_axis_off()
ax1.set_title('result')

plt.show()