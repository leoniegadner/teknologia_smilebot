#Facetracking 

import numpy as np
import cv2
import os
import time
import picamera

import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BCM)

GPIO.setup(18, GPIO.OUT)
GPIO.setup(24, GPIO.OUT)
pwm = GPIO.PWM(18, 100)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
cap.set(3,640) # set Width
cap.set(4,480) # set Height

x=230
y=110

with picamera.PiCamera() as camera:
    camera.resolution = (1024, 768)
    camera.start_preview()
    #Camera warm-up time
    time.sleep(2)
    camera.capture('foo.jpg')


while True:
    ret, img = cap.read()
    #img = cv2.flip(img, -1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(20,20)
    )

    for (x,y,w,h) in faces:
        print(x,y,w,h)
        #cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        #roi_gray = gray[y:y+h, x:x+w]
        #roi_color = img[y:y+h, x:x+w]
            
        if x in range(220,240):
            pwm.stop()
            time.sleep(0.0001)
        elif x > 240:
            GPIO.output(24, GPIO.LOW)
            pwm.start(10)
            pwm.ChangeFrequency(((x-240)*10)^2)

        elif x<220:
            GPIO.output(24, GPIO.HIGH)
            pwm.start(10)
            pwm.ChangeFrequency(((220-x)*10)^2)
                
        if y in range(60,140):
            time.sleep(0.0001)
            pwm.stop()
        elif y> 140:
            print("turn up")
            time.sleep(0.0001) 
        elif y<60:
            print("turn down")
            time.sleep(0.01)
        
    #cv2.imshow('video',img)
    
    
    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break
cap.release()
cv2.destroyAllWindows()
pwm.stop()
GPIO.cleanup()