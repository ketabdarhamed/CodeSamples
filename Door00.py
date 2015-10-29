# import the necessary packages
from collections import deque
import numpy as np
import argparse
import imutils
import cv2
import os
import matplotlib.pyplot as plt
 

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

fgbg = cv2.BackgroundSubtractorMOG(50, 1, 0.9, .1)
fgbg2 = cv2.BackgroundSubtractorMOG(50, 1, 0.9, .1)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#face_cascade = cv2.CascadeClassifier('body.xml')


#(grabbed, frame) = camera.read()
#frame = imutils.resize(frame, width=600)
#frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#oldframe=frame

# keep looping
while True:
	# grab the current frame
	(grabbed, frame) = camera.read()
	#img = [cv2.cvtColor(camera.read()[1], cv2.COLOR_BGR2GRAY) for i in xrange(1)]

	#frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 	
	#acc=0
 	#for i in img:
 	#	acc=acc+i
 	#frame=acc/1

	frame = imutils.resize(frame, width=600)
	fgmask = fgbg.apply(frame)
	cv2.imshow('frame',fgmask)
 

	# resize the frame, blur it, and convert it to the HSV
	# color space
	

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	print(np.var(gray),np.mean(fgmask))
	
	ret,thresh = cv2.threshold(gray,127,255,0)
	contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	cv2.drawContours(frame, contours, -1, (0,255,255), 3)

	cv2.putText(frame, "Press ESC to close.", (5, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,1,255))

	cv2.line(frame,(15,20),(70,50),(255,0,0),5)
	cv2.rectangle(frame,(15,20),(70,50),(0,255,0),3)


	framebw = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
             cv2.THRESH_BINARY_INV,11,2)

	kernel = np.ones((5,5),np.uint8)
	#frameErod = cv2.dilate(framebw,kernel,iterations = 1)

	opening = cv2.morphologyEx(framebw, cv2.MORPH_OPEN, kernel)

	cv2.imshow('BW',framebw)

	cv2.imshow('Denoise',opening)

	
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	for (x,y,w,h) in faces:
		myface=gray[y-10:y+h+10,x-10:x+w+10]
		cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),3)


	cv2.imshow('Face Detection',gray)

	myface = imutils.resize(myface, width=600)
	cv2.imshow('Myface',myface)
	cv2.imwrite('face.jpeg',myface)


	grass=cv2.imread('grass4.jpg')
	grass = imutils.resize(grass, width=600)
	gray_grass=cv2.cvtColor(grass, cv2.COLOR_BGR2GRAY)
	bw_grass = cv2.adaptiveThreshold(gray_grass,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
             cv2.THRESH_BINARY,11,2)
	
	
	kernel = np.ones((3,3),np.uint8)
	erod_grass=cv2.erode(bw_grass,kernel,iterations = 3)
	dilate_grass=cv2.dilate(erod_grass,kernel,iterations = 30)

	backtorgb = cv2.cvtColor(dilate_grass,cv2.COLOR_GRAY2RGB)

	print cv2.mean(backtorgb)

	backtorgb[:,:,1:2]=0
	detection=cv2.add(grass,backtorgb)
	
	cv2.imshow('Grass',detection)
	cv2.imwrite('Detection_results.jpg',detection)
	#cv2.imshow('BWCAR',bw_grass)









	#cv2.namedWindow('image')

	# create trackbars for color change
	#cv2.createTrackbar('R','image',0,255, nothing)

	#frame = cv2.GaussianBlur(frame, (11, 11), 0)
	#hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
 
	#difff=abs(img[1]-frame)

	#difff = cv2.GaussianBlur(difff, (11, 11), 0)

	#cv2.imshow("difff",difff) 

	
	#plt.hist(frame.flatten(), 256, range=(0,255))
	#plt.show()

	
 
	# show the frame to our screen
	#cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break
 
 	
 	oldframe=frame

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()	