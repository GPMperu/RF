import cv2
import os
import tkinter as tk
from tkinter import messagebox
import sys

dataPath = 'C:\g0\VS\Data' #Cambia a la ruta donde hayas almacenado Data
imagePaths = os.listdir(dataPath)
print('imagePaths=',imagePaths)

#face_recognizer = cv2.face.EigenFaceRecognizer_create()
#face_recognizer = cv2.face.FisherFaceRecognizer_create()
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Leyendo el modelo
#face_recognizer.read('modeloEigenFace.xml')
#face_recognizer.read('modeloFisherFace.xml')
face_recognizer.read('modeloLBPHFace.xml')

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
#cap = cv2.VideoCapture('Video.mp4')

 
def resource_path(relative_path):
    """Obtiene la ruta absoluta al recurso, funciona para dev y para PyInstaller."""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# para PyInstaller
    
cascade_path = resource_path("cv2/data/haarcascades/haarcascade_frontalface_default.xml")
faceClassif = cv2.CascadeClassifier(cascade_path)   

#  para interprete
  
#faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

while True:
	ret,frame = cap.read()
	if ret == False: break
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	auxFrame = gray.copy()

	faces = faceClassif.detectMultiScale(gray,1.3,5)

	for (x,y,w,h) in faces:
		rostro = auxFrame[y:y+h,x:x+w]
		rostro = cv2.resize(rostro,(150,150),interpolation= cv2.INTER_CUBIC)
		result = face_recognizer.predict(rostro)

		cv2.putText(frame,'{}'.format(result),(x,y-5),1,1.3,(255,255,0),1,cv2.LINE_AA)
		
		# EigenFaces
		#if result[1] < 5700:
		#	cv2.putText(frame,'{}'.format(imagePaths[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
		#	cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
		#else:
		#	cv2.putText(frame,'Desconocido',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
		#	cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
		
		# FisherFace
		#if result[1] < 500:
		#	cv2.putText(frame,'{}'.format(imagePaths[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
		#	cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
		#else:
		#	cv2.putText(frame,'Desconocido',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
		#	cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
		
		# LBPHFace
		if result[1] < 70:
			cv2.putText(frame,'{}'.format(imagePaths[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
			cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
		else:
			cv2.putText(frame,'Desconocido',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
			cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
		
	cv2.imshow('frame',frame)
	k = cv2.waitKey(1)
	if k == 27:
		break
	if k in [10, 13]:  # Enter en Linux o Windows
		#	cv2.destroyAllWindows()
		root = tk.Tk()
		root.withdraw()
		messagebox.showinfo(
		"Validación",
		f"Llamada a rutina de validación de socio\n\nimagePaths: {imagePaths[result[0]]}"
		)
		#	break
		# acción para Enter
cap.release()
cv2.destroyAllWindows()
