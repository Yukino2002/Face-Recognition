import os
import cv2
import numpy as np

people = []

DIR = r'Face Recognition Training Images'
for i in os.listdir(DIR):
    people.append(i)

haar_cascade = cv2.CascadeClassifier('haarcascade_face.xml')

# creating a function
# our training set will contain two lists, features(image array of faces), labels()
features = []
labels = []

def create_train():
    # iterating through every person in the peoples list, and linking the folder path individually everytime
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)
        # label is the associated index of the person we are currently training for in the people list
         
        # now we are inside the individual folders for every person
        # iterating through all the images inside every folder
        for image in os.listdir(path):
            image_path = os.path.join(path, image)
            # the os.path.join here, joins the the folder path with the image

            # we can access the image using cv2.imread, now that we have it's path
            image_array = cv2.imread(image_path)
            gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

            # converting into grayscale
            # the coordinates for the rectangle
            face_rectangle = haar_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 4)

            for (x, y, w, h) in face_rectangle:
                # cropping as per the region of interest in the image, basically where the face was detected using haar_cascade
                face_roi = gray[y:y + h, x:x + w]
                
                #now we have the database of faces, we can append them to the list
                features.append(face_roi)
                # label is basically the index, the idea behind it is to reduce the strain
                # by mapping the string value with a key
                labels.append(label)

create_train()
print(f'Features = {len(features)}')
print(f'Labels = {len(labels)}')
print('Training Done :3')

# then we are converting the features and labels list, into numerical numpy arrays
features = np.array(features, dtype = 'object')
labels = np.array(labels)

# declaring the variable function to train our model
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# training the recognize, passing the data set we tabulated
face_recognizer.train(features, labels)

# saving the trained model data set in a .yml file, with the numpy arrays as .np
face_recognizer.save('Face_trained.yml')
np.save('Features.npy', features)
np.save('Labels.npy', labels)
