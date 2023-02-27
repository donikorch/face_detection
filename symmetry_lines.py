# import required libraries
import cv2 as cv
import math as m
import os

cascPathface = os.path.dirname(
    cv.__file__) + "/data/haarcascade_frontalface_alt2.xml"
cascPatheyes = os.path.dirname(
    cv.__file__) + "/data/haarcascade_eye_tree_eyeglasses.xml"

# read input image
img = cv.imread('C:/Users/danya/Downloads/images/image_1.jpg')

# convert to grayscale of each frames
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# read the haarcascade to detect the faces in an image
face_cascade = cv.CascadeClassifier(cascPathface)

# read the haarcascade to detect the eyes in an image
eye_cascade = cv.CascadeClassifier(cascPatheyes)

# detects faces in the input image
faces = face_cascade.detectMultiScale(gray,
                                      scaleFactor=1.1,
                                      minNeighbors=5,
                                      flags=cv.CASCADE_SCALE_IMAGE,
                                      minSize=(50, 50))

# loop over the detected faces
for (x, y, w, h) in faces:
    roi_gray = gray[y:y + h, x:x + w]
    roi_color = img[y:y + h, x:x + w]
    center_x = int(0.5 * w)
    center_y = int(0.5 * h)
    cv.line(roi_color, (center_x, center_y + 1000), (center_x, center_y - 1000), (0, 0, 255), 2)

# detects eyes of within the detected face area (roi)
eyes = eye_cascade.detectMultiScale(roi_gray)

# draw a rectangle around eyes
for (ex, ey, ew, eh) in eyes:
    eye_x = int(ex + 0.5*ew)
    eye_y = int(ey + 0.5*eh)
    cv.line(roi_color, (eye_x, eye_y + 1000), (eye_x, eye_y - 1000), (0, 0, 255), 2)
    distance = m.sqrt((center_x - eye_x)**2 + (center_y - eye_y)**2)
    print(distance)


# display the image with detected eyes
cv.imshow('Eyes Detection', img)
cv.waitKey(0)
cv.destroyAllWindows()