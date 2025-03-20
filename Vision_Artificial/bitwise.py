import cv2 as cv #Importamos OpenCV
import numpy as np 

blank = np.zeros((400,400),dtype='uint8')

rectangle = cv.rectangle(blank.copy(),(30,30),(370,370),255,-1)
circle = cv.circle(blank.copy(),(200,200),200,255,-1)

#cv.imshow('Rectangulo',rectangle)
#cv.imshow('Circulo',circle)

#Operadores bitwise : AND, OR , XOR , NOT etc..


bitwise_AND = cv.bitwise_and(rectangle,circle) #Pixeles en común entre las dos fotos
bitwise_OR = cv.bitwise_or(rectangle,circle) #Pixeles que estén en ambas fotos
bitwise_XOR = cv.bitwise_xor(rectangle,circle) #Pixeles que deben estar exclusivamente en una figura sobre la otra

bitwise_NOT = cv.bitwise_not(circle) #Pixeles que no pertenezcan a la figura


cv.imshow('AND',bitwise_AND)
cv.imshow('OR',bitwise_OR)
cv.imshow('XOR',bitwise_XOR)
cv.imshow('NOT',bitwise_NOT)







cv.waitKey(0)