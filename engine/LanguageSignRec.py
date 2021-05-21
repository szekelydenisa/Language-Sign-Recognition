import cv2
import imutils
import numpy as np
import sys

bg = None
folder = sys.argv[1]
name = sys.argv[2]

def run_avg(image, aWeight):
    global bg
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return

    # compute average
    cv2.accumulateWeighted(image, bg, aWeight)

def segment(image, threshold=20):
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

def main():
    # initialize alfa
    aWeight = 0.7

    camera = cv2.VideoCapture(0)

    # ROI coordinates
    top, right, bottom, left = 100, 350, 425, 650

    num_frames = 0
    image_num = 0
    start_recording = False


    while(True):
        # get the current frame and edit it
        (grabbed, frame) = camera.read()

        frame = imutils.resize(frame, width=700)

        frame = cv2.flip(frame, 1)

        clone = frame.copy()

        cv2.imshow("frame",clone);

        (height, width) = frame.shape[:2]

        # get the ROI
        roi = frame[top:bottom, right:left]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        height, width = gray.shape
        segmentation = np.zeros((height, width, 1), dtype = "uint8")

        # to get the background, keep looking till a threshold is reached
        if num_frames < 60:
            run_avg(gray, aWeight)
            print(num_frames)
        else:
            hand = segment(gray)

            if hand is not None:
               
                (thresholded, segmented) = hand

                # draw the segmented region and display the frame
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                cv2.drawContours(segmentation, [segmented ], -1, (255, 255, 255), cv2.FILLED)
                if start_recording:
                    print (folder + "\\" + name + "_test" + str(image_num) + '.png')
                    cv2.imwrite(folder + "\\" + name + "_test" + str(image_num) + '.png', segmentation)
                    image_num += 1
                    print(image_num)
                    sys.stdout.flush()
                cv2.imshow("Thesholded", thresholded)
                cv2.imshow("Segmentation", segmentation)

        cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)

        num_frames += 1

        # display the frame with segmented hand
        cv2.imshow("Video", clone)

        keypress = cv2.waitKey(1) & 0xFF

        # if the user pressed "q", then stop looping
        if keypress == ord("q") or image_num > 3000:
            break
        
        if keypress == ord("s"):
            start_recording = True

main()
cv2.destroyAllWindows()