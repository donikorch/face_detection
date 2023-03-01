import cv2 as cv
import math as m
import os

cascPathface = os.path.dirname(
    cv.__file__) + "/data/haarcascade_frontalface_alt2.xml"
cascPatheyes = os.path.dirname(
    cv.__file__) + "/data/haarcascade_eye_tree_eyeglasses.xml"

img = cv.imread('C:/Users/danya/Downloads/images/image_1.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

face_cascade = cv.CascadeClassifier(cascPathface)

eye_cascade = cv.CascadeClassifier(cascPatheyes)

faces = face_cascade.detectMultiScale(gray,
                                      scaleFactor=1.1,
                                      minNeighbors=5,
                                      flags=cv.CASCADE_SCALE_IMAGE,
                                      minSize=(50, 50))

for (x, y, w, h) in faces:
    roi_gray = gray[y:y + h, x:x + w]
    roi_color = img[y:y + h, x:x + w]
    center_x = int(0.5 * w)
    center_y = int(0.5 * h)
    cv.line(roi_color, (center_x, center_y + 1000), (center_x, center_y - 1000), (0, 0, 255), 2)

eyes = eye_cascade.detectMultiScale(roi_gray)

for (ex, ey, ew, eh) in eyes:
    eye_x = int(ex + 0.5*ew)
    eye_y = int(ey + 0.5*eh)
    cv.line(roi_color, (eye_x, eye_y + 1000), (eye_x, eye_y - 1000), (0, 0, 255), 2)
    distance = m.sqrt((center_x - eye_x)**2 + (center_y - eye_y)**2)
    print(distance)


cv.imshow('Eyes Detection', img)
cv.waitKey(0)
cv.destroyAllWindows()