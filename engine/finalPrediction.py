import cv2
import imutils
import numpy as np
import pyttsx3
import argparse
from keras.models import load_model
from predict import Prediction
from threading import Thread
import sys


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--mode", required=False,
	help="path to output trained model")

args = vars(ap.parse_args())

mode = args["mode"]
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 0.9)
bg = None
model = load_model('E:/an4/sem II/licenta/electron-quick-start/engine/output/model_licenta.model')

def run_avg(image, aWeight):
    global bg
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return

    # compute average
    cv2.accumulateWeighted(image, bg, aWeight)

def segment(image, threshold = 35):
    global bg
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)
  

    thresholded = cv2.threshold(diff,
                                threshold,
                                255,
                                cv2.THRESH_BINARY)[1]
    
    # get the contours in the image after thresholdinf
    (cnts, _) = cv2.findContours(thresholded.copy(),
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

 
    if len(cnts) == 0:
        return
    else:
        # get the maximum contour (the hand)
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)

def say_text(text):
    while engine._inLoop:
        pass
    engine.say(text)
    engine.runAndWait()

def post_processing(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return image

def main():
    # initialize alfa
    aWeight = 0.7
    is_voice_on = True
    camera = cv2.VideoCapture(0)

    # ROI coordinates
    top, right, bottom, left = 100, 350, 425, 650

    num_frames = 0
    image_num = 0
    count_same_frame = 0
    text= ""
    word = ""
    print(mode)

    while(True):
     
        old_text = text

        (grabbed, frame) = camera.read()

        frame = imutils.resize(frame, width=700)

        frame = cv2.flip(frame, 1)

        clone = frame.copy()

        blackboard = np.zeros((700, 900, 3), dtype=np.uint8)
        cv2.putText(blackboard, 'Hello buddy!', (1,42), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 255, 255))
        cv2.putText(blackboard, 'Please put your hand in the green square.', (1,80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
        cv2.putText(blackboard, 'Nobody is perfect! So if I\'m making a silly mistake forgive me and just press D to delete.', (1,120), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
        cv2.putText(blackboard, 'When you\'re done press Q to exit.', (1,160), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
        
        (height, width) = frame.shape[:2]

        # get the ROI
        roi = frame[top:bottom, right:left]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        if(mode == ' Debug'):
            cv2.imshow("ROI", gray)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        if(mode == ' Debug'):
            cv2.imshow("Gaussian Blur", gray)
        gray = cv2.medianBlur(gray, 15)
        if(mode == ' Debug'):
            cv2.imshow("frame",frame)
            cv2.imshow("Final gray", gray)

        # to get the background, keep looking till a threshold is reached
        if num_frames < 60:
            run_avg(gray, aWeight)
            print(num_frames)
        else:
            hand = segment(gray)

            if hand is not None:      
                (thresholded, segmented) = hand
                if(mode == ' Debug'):
                    cv2.imshow('thresholded without post process', thresholded)
                thresholded = post_processing(thresholded)
                cv2.imshow("Thesholded", thresholded)

                # draw the segmented region
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                if cv2.contourArea(segmented) > 10000:
                    text, prob = Prediction.predict(thresholded, model, 'E:/an4/sem II/licenta/electron-quick-start/engine/output/label_bin_final.pickle', 50, 50, -1);
                    if prob*100 > 94:
                        if old_text == text:
                            count_same_frame+=1
                            print(count_same_frame)
                        else:
                            count_same_frame=0
                        if count_same_frame >= 25:
                            word = word + text
                            if(mode == ' Speak'):
                                Thread(target=say_text, args=(word, )).start() 
                            count_same_frame=0
                        else:
                            if(mode == ' Speak'):
                                Thread(target=say_text, args=('', )).start()
                elif cv2.contourArea(segmented) < 1000:
                    text = ""
            else: 
                text = ""

        if cv2.waitKey(1) == ord("d"):
            word = word[:-1]

        if(mode == ' Speak' and len(word) >= 1 and cv2.waitKey(1) == ord("s") ):
            print("hereeee")
            Thread(target=say_text, args=(word, )).start()
        

        cv2.putText(blackboard, word, (30, 240), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 255, 255))
        cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)
        num_frames += 1
    
         # display the frame with segmented hand
        cv2.imshow("Video", clone)
        cv2.imshow("Result", blackboard)
        # if the user pressed "q", then stop looping

        if cv2.waitKey(1) == ord("q"):
            break
        

       

Prediction.predict(np.zeros((50, 50), dtype=np.uint8), model, 'E:/an4/sem II/licenta/electron-quick-start/engine/output/label_bin_final.pickle', 50, 50, -1);
main()