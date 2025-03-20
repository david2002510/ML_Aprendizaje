import cv2 as cv #Importamos OpenCV


# Definicion de funciones 

#Función de reescalado para optimizar la salida de imagen o frame -> Para archivos grandes en resolucion
def reescalado(frame,scale = .75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width,height)
    return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)


#Para una imagen

img = cv.imread('fotos/park.jpg') #Escogemos la fotos a leer

img_resized = reescalado(img)

#cv.imshow('Cat',img) #Mostramos la foto

cv.imshow('Cat_Resized',img_resized)



#Para un video

capture = cv.VideoCapture('videos/dog.mp4') #Si utilizamos Webcam, se pasa 0 como argumento de normal

while True:
    isTrue,frame = capture.read() #Lee el video frame a frame

    frame_resized = reescalado(frame,scale=.2)

    #cv.imshow('Video',frame)
    cv.imshow('Video_Resized',frame_resized)

    if cv.waitKey(20) & 0xFF==ord('d'):
        break

capture.release()
cv.destroyAllWindows()

