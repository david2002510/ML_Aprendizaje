import cv2 as cv
import mediapipe as mp
import time


capture = cv.VideoCapture(0) #Fuente de video: Webcam

#Nuestras manos, como mucho podemos detectar nuestras 2 manos por defecto del modulo Hands
mpHands = mp.solutions.hands 
hands = mpHands.Hands() 
#Dibujar los puntos sobre nuestra mano cuando lo detecte
mpDraw = mp.solutions.drawing_utils

#Contador de fotogramas
pTime = 0
cTime = 0 



while True:
    isTrue,frame = capture.read() #Lee el video frame a frame
    imgRGB = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks: #Si detecta la mano...
        for handLms in results.multi_hand_landmarks:
            # id de cada una de las 21 lms o nodos que conforma cada mano
            for id,lm in enumerate(handLms.landmark):
                h,w,c = frame.shape 
                cx,cy = int(lm.x*w), int(lm.y*h) #Normaliza a pixeles de la pantalla
                
                if id == 0: #Muñeca de la mano, primer nodo o landmark
                    cv.circle(frame,(cx,cy),10,(255,0,255),cv.FILLED)
                

            mpDraw.draw_landmarks(frame,handLms,mpHands.HAND_CONNECTIONS)

    #FPS
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = time.time()

    cv.putText(frame,str(int(fps)),(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)

    #Importante ponerlo al final el show para que dibuje todo
    cv.imshow('Video',frame)
    if cv.waitKey(20) & 0xFF==ord('d'): #Detener con "d"
        break