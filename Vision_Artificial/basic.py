import cv2 as cv #Importamos OpenCV


img = cv.imread('fotos/park.jpg') #Escogemos la fotos a leer

#---Las 5 funciones básicas de OpenCV para vision artificial---

#Para pasar una imagen a blanco/negro, util para discriminación 
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

#Desenfoque Gaussiano
blur = cv.GaussianBlur(img,(7,7),cv.BORDER_DEFAULT) # ----> ¡El Kernel debe ser una tupla de numeros IMPARES! <----


#Edge Cascade
canny = cv.Canny(blur,125,175) 

#Dilatando
dilated = cv.dilate(canny,(7,7),iterations=3)

#Eroding o Erosionada
eroded = cv.erode(dilated,(7,7),iterations= 1)

#Resized, ajustarlo a una resolucin determinada
resized = cv.resize(eroded,(320,200),interpolation=cv.INTER_AREA)


cv.imshow('Original',img) #Mostramos la foto original
#cv.imshow('Gray',gray)
#cv.imshow('Blur',blur)
#cv.imshow('Canny Edges',canny)
#cv.imshow('Dilated',dilated)
#cv.imshow('Eroded',eroded)
cv.imshow('Final',resized) #Imagen final despues de aplicarle todos los filtros anteriores en cadena


cv.waitKey(0)