# importing libraries 
import cv2 
import numpy as np 

# Create a VideoCapture object and read from input file 
cap = cv2.VideoCapture(r'C:/Users/eTech/Desktop/ML Project/outputVideo.mp4') 
if (cap.isOpened()== False): 
	print("Error opening video file") 


while(cap.isOpened()): 
# Capture frame-by-frame 
	ret, frame = cap.read() 
	if ret == True: 
	# Display the resulting frame 
		cv2.imshow('Frame', frame) 
		
	# Press Q on keyboard to exit 
		if cv2.waitKey(25) & 0xFF == ord('q'): 
			break


	else: 
		break

# When everything done, release 
# the video capture object 
cap.release() 
cv2.destroyAllWindows() 

