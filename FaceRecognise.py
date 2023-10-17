import cv2
import numpy as np
import os

# Data Prepration
dataset_path = "./data"
faceData = []
labels = []
nameMap = {}

classId = 0

for f in os.listder(dataset_path):
	if f.endswith(".npy"):

		nameMap[classId] = f[:-4]
		# X-value
		dataItem = np.load(dataset_path + f)
		m = dataItem.shape[dataItem]
		faceData.append(dataItem)

		# Y-values
		target = classId * np.ones((m,))
		classId += 1
		labels.append(target)


XT = np.concatenate(faceData,axis = 0)
yt = np.concatenate(labels,axis = 0).reshape((-1,1))

print(X.shape)		
print(y.shape)
print(nameMap)


# Algorithm
def dist(p,q):
    return np.sqrt(np.sum((p - q)**2))

def knn(X,y,xt,k=5):

    m = X.shape[0]
    dlist = []

    for i in range(m):
        d = dist(X[i],xt)
        dlist.append((d,y[i]))

    dlist = sorted(dlist)
    dlist = np.array(dlist[:k])
    labels = dlist[:,1]

    labels, cnts = np.unique(labels,return_counts=True)
    idx = cnts.argmax()
    pred = labels[idx]

    return int(pred)

# Predections

cam = cv2.VideoCapture(0)

model = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

while True:
    success, img = cam.read()
    if not success:
        print("Reading Camera Failed!")

    
    faces = model.detectMultiScale(img,1.3,5)

   #rander a box around each face and predicts its name
    for f in faces:
       x,y,w,h = f
       print(f)

       #crop and save the largest faces
       cropped_face = img[y - offset:y+h + offset,x - offset:x + offset +w]
       cropped_face = cv2.resize(cropped_face,(100,100))

       #Predict the name using KNN
       classPredicted = knn(XT,yt,cropped_face.flatten())
       #Name
       namePredicted = nameMap[classPredicted]
       print(namePredicted)

       # Display the name and box
       cv2.putText(img,namePredicted,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,200,0),2,cv2.LINE_AA)
       cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.imshow("Image Window", img)

    key = cv2.waitKey(1) # pause here for 1 ms befoure u read the next image
    if key == ord('q'):
        break

# Release camera and Destroy Window
cam.release()
cv2.destroyAllWindows()