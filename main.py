import numpy as np
from cv2 import cv2 as cv
from matplotlib import pyplot as plt

face_cascade = cv.CascadeClassifier('haarcascade_eye.xml')

img = cv.imread('D:\\giwawa.jpg')
imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
pathf = 'D:\\haarcascade_eye.xml'
face_cascade.load(pathf)

faces = face_cascade.detectMultiScale(imgGray, 1.3, 5)

for (x,y,w,h) in faces:
    cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

cv.namedWindow('img', cv.WINDOW_NORMAL)
cv.imshow('img', img)
cv.waitKey(0)
cv.destroyAllWindows()

face_cascade.load(pathf)


