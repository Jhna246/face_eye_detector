import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
%matplotlib inline

face = cv2.CascadeClassifier('DATA/haarcascades/haarcascade_frontalface_default.xml')
eye = cv2.CascadeClassifier('DATA/haarcascades/haarcascade_eye.xml')

def detect_face(img):
    face_img = img.copy()
    face_rectangle = face.detectMultiScale(face_img, scaleFactor=1.15, minNeighbors=5)
    for (x,y,w,h) in face_rectangle:
        cv2.rectangle(face_img, (x,y), (x+w,y+h), (255,0,0), 10)
    return face_img

def detect_eyes(img):
    eye_img = img.copy()
    eye_rectangles = eye.detectMultiScale(eye_img, scaleFactor=1.15, minNeighbors=5)
    for (x,y,w,h) in eye_rectangles:
        cv2.rectangle(eye_img, (x,y), (x+w,y+h), (255,0,0), 10)
    return eye_img

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read(0)
    frame = detect_face(frame)
    time.sleep(0)
    frame = detect_eyes(frame)
    cv2.imshow('Face Detector', frame)
    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()
