# Read a Video From Web cam Using openCV
import cv2

#Creat a Camera Object..
cam = cv2.VideoCapture(0)

#Read Image From Camera Object
while True:
	success, img = cam.read()
    if not success:
    	print("Reading Camera Failed!")
    	
	cv2.imshow("Image Window", img)
	cv2.waitKey(1) # Pause here for 1 ms before you read the next image


	    
    
   

