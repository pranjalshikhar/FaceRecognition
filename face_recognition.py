# Recognise Faces using some classification algorithm - like Logistics, KNN, SVM, etc.

# 1. Load the training data (numpy arrays of all the persons)
#         x - values are stored in the numpy arrays
#         y - values we need to assign to each person
# 2. Read a Video stream using opencv
# 3. Extract faces out of it
# 4. Use KNN to find the prediction of the face (int)
# 5. Map the predicted id to name of the user
# 6. Display the predictions on the screen - bounding box and name

import cv2
import numpy as np
import os


#######   KNN CODE   #######

def distance(v1,v2):
    return np.sqrt(sum((v1-v2)**2))  # Euclidean Distance Formula

def knn(train,test,k=5):  # Class1,CLass2,QueryPoint,No. of neighbours

    dist = []

    for i in range(train.shape[0]):
        ix = train[i, :-1]
        iy = train[i, -1]
        # Compute the distance from test point
        d = distance(test, ix)
        dist.append([d,iy])  # Append distance and label


    # Sort based on distance from test point
    dk = sorted(dist, key = lambda x: x[0])[:k]
    # Retrieve only the labels
    labels = np.array(dk)[:, -1]

    # Get frequencies of each label
    output = np.unique(labels, return_counts = True)
    # Find max frequency and corresponding label
    index = np.argmax(output[1])
    return output[0][index]

##########################################

# Init Camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Face Detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip = 0
dataset_path = "C:/Users/shikh/OneDrive/Desktop/Data Science Masters/9. Project - Face Recognition/"

face_data = []
labels = []

class_id = 0
names = {}   # Mapping b/w ID and NAME



# Data Preperation
for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        # Create a mapping b/w ID and NAME
        names[class_id] = fx[:-4]
        data_item = np.load(dataset_path + fx)
        face_data.append(data_item)

        # Create Labels for the class
        target = class_id*np.ones((data_item.shape[0],))
        class_id += 1
        labels.append(target)

face_dataset = np.concatenate(face_data, axis = 0)
face_labels = np.concatenate(labels, axis = 0).reshape((-1,1))

print(face_dataset.shape)
print(face_labels.shape)

trainset = np.concatenate((face_dataset, face_labels), axis = 1)
print(trainset.shape)


# Testing
while True:
    ret, frame = cap.read()

    if ret == False:
        continue

    faces = face_cascade.detectMultiScale(frame,1.3,5)

    for face in faces:
        x,y,w,h = face

        # Get the face Region of Interest
        offset = 10
        face_section = frame[y-offset: y+h+offset, x-offset: x+w+offset]
        face_section = cv2.resize(face_section,(100,100))

        # Predicted Label (output)
        out = knn(trainset, face_section.flatten())

        # Display on the name and rectangle around it
        pred_name = names[int(out)]
        cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,255), 2)

    cv2.imshow("Faces", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
