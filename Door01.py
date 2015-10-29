# import the necessary packages
from collections import deque
import numpy as np
import argparse
import imutils
import cv2
import os
import matplotlib.pyplot as plt

BUF_SIZE=20
 

# Callback Function for Trackbar (but do not any work)
def nothing(*arg):
    pass


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
	help="max buffer size")
args = vars(ap.parse_args())

 
# if a video path was not supplied, grab the reference
# to the webcam
if not args.get("video", False):
	camera = cv2.VideoCapture(0)
 
# otherwise, grab a reference to the video file
else:
	camera = cv2.VideoCapture(args["video"])


# keep looping
while True:
	# grab the current frame
	#(grabbed, frame) = camera.read()
	img = [imutils.resize(cv2.cvtColor(camera.read()[1], cv2.COLOR_BGR2GRAY),600) for i in xrange(BUF_SIZE)]
	#img = [camera.read()[1] for i in xrange(BUF_SIZE)]


	avg_img=0
	for i in xrange(BUF_SIZE):
		avg_img=avg_img+img[i]

	avg_img=avg_img/BUF_SIZE

	var_frame=np.var(img, axis=0)

	frame=abs(img[0]-avg_img.astype(np.uint8))


	print(np.sum(var_frame),frame.size)
	#ret,frame = cv2.threshold(frame,200,255,0)

	#frame = imutils.resize(frame, width=600)
	cv2.imshow('HP',frame)
	cv2.imshow('LP',avg_img.astype(np.uint8))
	cv2.imshow('Var_Frame',var_frame.astype(np.uint8))
 
	


	
 
	# show the frame to our screen
	#cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break
 
 	
# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()	