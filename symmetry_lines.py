import cv2 as cv
import math as m
import matplotlib.pyplot as plt
import os

cascPathface = os.path.dirname(
    cv.__file__) + "/data/haarcascade_frontalface_alt2.xml"
cascPatheyes = os.path.dirname(
    cv.__file__) + "/data/haarcascade_eye_tree_eyeglasses.xml"

img = cv.imread('images_self/image_1.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

face_cascade = cv.CascadeClassifier(cascPathface)

eye_cascade = cv.CascadeClassifier(cascPatheyes)

faces = face_cascade.detectMultiScale(gray,
                                      scaleFactor=1.1,
                                      minNeighbors=5,
                                      flags=cv.CASCADE_SCALE_IMAGE,
                                      minSize=(100, 100))

for (x, y, w, h) in faces:
    center_x = int(x + 0.5 * w)
    center_y = int(y + 0.5 * h)
    cv.line(img, (center_x, center_y + 1000), (center_x, center_y - 1000), (0, 0, 255), 5)

    eyes = eye_cascade.detectMultiScale(gray)
    
    for (ex, ey, ew, eh) in eyes:
        eye_x = int(ex + 0.5*ew)
        eye_y = int(ey + 0.5*eh)
        cv.line(img, (eye_x, eye_y + 1000), (eye_x, eye_y - 1000), (0, 0, 255), 5)
        distance = m.sqrt((center_x - eye_x)**2 + (center_y - eye_y)**2)
        print(distance)


fig = plt.figure(figsize=(8, 5))
ax1 = plt.subplot(1, 3, 1)

ax1.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB), cmap=plt.cm.gray)
ax1.set_axis_off()
ax1.set_title('result')

plt.show()