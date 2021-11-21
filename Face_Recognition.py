import os
import cv2
import numpy as np

# Declaring the list
people = []

DIR = r'Face Recognition Training Images'
for i in os.listdir(DIR):
    people.append(i)

# defining the variable for face detection, and loading the data set from the opencv source
haar_cascade = cv2.CascadeClassifier('haarcascade_face.xml')

# defining the variable for face recognition, and loading the previously trained model
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('Face_trained.yml')

# Taking an image from the Validation folder, user can change
# currently can choose amongst 'emu', 'thor', 'stoned', 'leo', 'alexa'
image = cv2.imread(r'Validation\stoned.jpg')

# converting the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# getting the coordinates of the region of interest, that is the face
face_rectangle = haar_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 11)


for (x, y, w, h) in face_rectangle:
    # cropping the input grayscale image
    face_roi = gray[y:y + h, x:x + w]

    # using the .predict function on the trained model variable
    # 0 indicates perfect match, and accuracy reduces further ahead
    label, confidence = face_recognizer.predict(face_roi)
    print(f'Label = {people[label]} with a conifdence value of = {confidence}')

    # putting the name of the person in the photo, and highlighting their face
    cv2.putText(image, str(people[label]), (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 255, 150), thickness =  2)
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 150), 2)

# displaying the initial image with the changes
cv2.imshow('Detected Face', image)
cv2.waitKey(0)