import numpy as np
import cv2
import os
import time
import picamera
import RPi.GPIO as GPIO

from tensorflow.keras import Sequential #for emotion recognition
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

GPIO.setmode(GPIO.BCM)

GPIO.setup(18, GPIO.OUT)
GPIO.setup(24, GPIO.OUT)

pwm = GPIO.PWM(18, 100)
GPIO.output(24, GPIO.LOW)

# Load the model
model = Sequential()
classifier = load_model('ferjj.h5') # This model has a set of 6 classes
# We have 6 labels for the model
class_labels = {0: 'Angry', 1: 'Fear', 2: 'Happy', 3: 'Neutral', 4: 'Sad', 5: 'Surprise'}
classes = list(class_labels.values())
# print(class_labels)


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') #for tracking & emotion

cap = cv2.VideoCapture(0)   #tracking
cap.set(3,640) # set Width
cap.set(4,480) # set Height

with picamera.PiCamera() as camera:
    camera.resolution = (1024, 768)
    camera.start_preview()
    #Camera warm-up time
    time.sleep(2)
    camera.capture('foo.jpg')


def face_detector_video(img):
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return (0, 0, 0, 0), np.zeros((48, 48), np.uint8), img

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
        roi_gray = gray[y:y + h, x:x + w]

    roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

    return (x, w, y, h), roi_gray, img


def emotionVideo(cap):
    ret, frame = cap.read()
    rect, face, image = face_detector_video(frame)
    [x,w,y,h] = rect

    #frame = cv2.flip(frame, -1)     #facetrack 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(20,20)
        )
            
    if np.sum([face]) != 0.0:
        roi = face.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        # make a prediction on the ROI, then lookup the class
        preds = classifier.predict(roi)[0]
        label = class_labels[preds.argmax()]
        print(label)
        #if label == "Happy":
            #monitor happy
        #elif label == "Sad":
            #monitor sad
        #elif label == "Angry":
            #monitor angry
        #elif label == "Surprise":
            #monitor surprise

        #label_position = (rect[0] + rect[1]//50, rect[2] + rect[3]//50)
        #text_on_detected_boxes(label, label_position[0], label_position[1], image) # You can use this function for your another opencv projects.
        fps = cap.get(cv2.CAP_PROP_FPS)
        #cv2.putText(image, str(fps),(5, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        #ret, img = cap.read()


def facetrack():
    print("Searching...")
    x=230
    y=110
    
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
            #print("Face found!")
            #print(x,y,w,h)
            #cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            #roi_gray = gray[y:y+h, x:x+w]
            #roi_color = img[y:y+h, x:x+w]   
            if x in range(190,270):
                pwm.stop()
                emotionVideo(cap)
            elif x > 240:
                GPIO.output(24, GPIO.LOW)
                pwm.start(10)
                pwm.ChangeFrequency(((x-240)*10)^2)

            elif x < 220:
                GPIO.output(24, GPIO.HIGH)
                pwm.start(10)
                pwm.ChangeFrequency(((220-x)*10)^2)
                
            #if y in range(60,140):
            #    time.sleep(0.0001)
            #elif y > 190:
            #    print("turn up")
            #    time.sleep(0.0001) 
            #elif y < 10:
            #    print("turn down")
            #    time.sleep(0.01)
        
        #cv2.imshow('video',img)
        ret, frame = cap.read()
        rect, face, image = face_detector_video(frame)
        if np.sum([face]) == 0.0:
            pwm.start(10)
            pwm.ChangeFrequency(1000)
    
    
        k = cv2.waitKey(30) & 0xff
        if k == 27: # press 'ESC' to quit
            break
#try:
facetrack()
#except:
#    print("Stopping...")
#finally:
cap.release()
#    cv2.destroyAllWindows()
pwm.stop()
GPIO.cleanup()