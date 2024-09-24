import cv2 
import numpy as np
import face_recognition

imgRol = face_recognition.load_image_file('cr7.jpg')
imgRol = cv2.cvtColor(imgRol,cv2.COLOR_BGR2RGB)

imgTest = face_recognition.load_image_file('cr77.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)


#fiding the faces in the image

Facloc = face_recognition.face_locations(imgRol) [0]
encodeRol = face_recognition.face_encodings(imgRol) [0]
cv2.rectangle(imgRol,(Facloc[1],Facloc[2]),(Facloc[3],Facloc[0]),(255,0,255),2)

FacTest = face_recognition.face_locations(imgTest) [0]
encodeTest = face_recognition.face_encodings(imgTest) [0]
cv2.rectangle(imgTest,(FacTest[1],FacTest[2]),(FacTest[3],FacTest[0]),(255,0,255),2)

#comparing images and finding the distance between them 
results = face_recognition.compare_faces([encodeRol], encodeTest)
print(results)



cv2.imshow('christino Rolando',imgRol)
cv2.imshow('christino ',imgTest)
cv2.waitKey(0)
