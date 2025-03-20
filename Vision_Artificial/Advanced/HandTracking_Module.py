import cv2 as cv
import mediapipe as mp
import time


class handDetector():
    def __init__(self, mode=False,maxHands = 2 , modelComplexity = 1, detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplex = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands  = mp.solutions.hands 
        self.hands  = self.mpHands.Hands(self.mode,self.maxHands,self.modelComplex,self.detectionCon,self.trackCon) 
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self,frame, draw = True):
        imgRGB = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)
        if results.multi_hand_landmarks: #Si detecta la mano...
            for handLms in results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame,handLms,self.mpHands.HAND_CONNECTIONS)
        return frame
    
def main():

    capture = cv.VideoCapture(0) #Fuente de video: Webcam

    #Contador de fotogramas
    pTime = 0
    cTime = 0 

    detector = handDetector()

    while True:
        isTrue,frame = capture.read() #Lee el video frame a frame
        frame = detector.findHands(frame) #Definicion hecho allí arriba
        #FPS
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = time.time()

        cv.putText(frame,str(int(fps)),(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)

        cv.imshow('Video',frame)
        if cv.waitKey(20) & 0xFF==ord('d'): #Detener con "d"
            break
    
    
if __name__ == "__main__":
    main()