# Read a Video From Web cam Using openCV
# Face detection in Video
# Click 20 pictures of the person who comes in the front of the camera and save them as numpy

import cv2

# Create acamera Object
cam = cv2.VideoCapture(0)

# Ask the Name
fileName = input("Enter the name of the person :")
dataset_path = "./data/"

#Model....
model = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

#Read Image From Camera Object
while True:
    success, img = cam.read()
    if not success:
        print("Reading Camera Failed!")

    # Store the gray images
    grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = model.detectMultiScale(img,1.3,5)

    #pick the faces with largest bounding box
    faces = sorted(faces,key=lambda f:f[2]*f[3])

    #pick the largest face
    if len(faces)>0:
       f = faces[-1]

       x,y,w,h = f
       cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

       #crop and save the largest faces
       cropped_face = img[y:y+h,x:x+w]

    cv2.imshow("Image Window", img)
    cv2.imshow("Cropped Face", cropped_face)

    key = cv2.waitKey(1) # pause here for 1 ms befoure u read the next image

    if key == ord('q'):
        break

# Release camera and Destroy Window

cam.release()
cv2.destroyAllWindows()




