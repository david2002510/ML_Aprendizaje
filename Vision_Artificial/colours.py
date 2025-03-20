import cv2 as cv #Importamos OpenCV
import numpy as np 

# Definicion de funciones 

#Función de reescalado para optimizar la salida de imagen o frame -> Para archivos grandes en resolucion
def reescalado(frame,scale = .75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width,height)
    return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)

img = cv.imread('fotos/park.jpg') #Escogemos la fotos a leer

img_resized = reescalado(img)


b,g,r = cv.split(img_resized)

cv.imshow('Original',img_resized) #Mostramos la foto original reescalando
cv.imshow('Canal Red',r)
cv.imshow('Canal Blue',b)
cv.imshow('Canal Green',g)



cv.waitKey(0)