import os
os.chdir('/Users/lyhbarry/Documents/Y4S1/CS4243/Lab/Lab5')
import cv2
import numpy as np

def get_info(cap):
	frameWidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
	frameHeight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
	fps = cap.get(cv2.CAP_PROP_FPS)
	frameCount = cap.get(cv2.CAP_PROP_FRAME_COUNT)

	print 'frame width: ', frameWidth, '\n', \
		  'frame height: ', frameHeight, '\n', \
		  'fps: ', fps, '\n', \
		  'frame count: ', frameCount

	return int(frameWidth), int(frameHeight), int(fps), int(frameCount)

def main():
	cap = cv2.VideoCapture('traffic.mp4')
	frameWidth, frameHeight, fps, frameCount = get_info(cap)

	_, img = cap.read()
	avgImg = np.float32(img)

	for fr in range(1, frameCount):
		_, img = cap.read()
		alpha = 1.0 / (fr + 1.0)

		cv2.accumulateWeighted(img, avgImg, alpha)
		normImg = cv2.convertScaleAbs(avgImg)

		cv2.imshow('img', img)
		cv2.imshow('normImg', normImg)
		print 'fr = ', fr, ' alpha = ', alpha
	cv2.waitKey(0)
	cv2.imwrite('bg.jpg', normImg)	
	cv2.destroyAllWindows()
	cap.release()	

main()
	
