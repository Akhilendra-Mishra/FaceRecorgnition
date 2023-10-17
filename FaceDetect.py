#Read a Video From Web cam Using openCV...
#Face detection in Video...
import cv2

#Creat a Camera Object..
cam = cv2.VideoCapture(0)

#Model....
model = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

#Read Image From Camera Object
while True:
	success, img = cam.read()
	if not success:
		print("Reading Camera Failed!")

	faces = model.detectMultiScale(img,1.3,5)

	for f in faces:
		x,y,w,h = f
		cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)


	cv2.imshow("Image Window", img)
	key = cv2.waitKey(1) # pause here for 1 ms befoure u read the next image

	if key == ord('q'):
		break

# Release camera and Destroy Window

cam.release()
cv2.destroyAllWindows()