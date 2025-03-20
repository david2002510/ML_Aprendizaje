import cv2 as cv
import mediapipe as mp
import time as tm


class FaceMesh():
    def __init__(self, static_image_mode=True,max_num_faces=2,refine_landmarks=True,min_detection_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.min_detection_confidence = min_detection_confidence

        self.mpFaceMesh  = mp.solutions.face_mesh
        self.faceMesh  = self.mpFaceMesh.FaceMesh(self.static_image_mode,self.max_num_faces,self.refine_landmarks,self.min_detection_confidence) 
        self.mpDraw = mp.solutions.drawing_utils

    def findMesh(self,frame, draw = True):
        imgRGB = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
        results = self.faceMesh.process(imgRGB)
        if results.multi_face_landmarks: #Si detecta la mano...
            for faceLms in results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame,faceLms,self.mpFaceMesh.FACEMESH_TESSELATION)
        return frame




def main():
    capture = cv.VideoCapture(0) #Fuente de video: Webcam
    #Contador de fotogramas
    pTime = 0
    cTime = 0
    detector = FaceMesh()
    while True:
        isTrue,frame = capture.read() #Lee el video frame a frame
        frame = detector.findMesh(frame)
        #FPS
        cTime = tm.time()
        fps = 1/(cTime-pTime)
        pTime = tm.time()
        cv.putText(frame,str(int(fps)),(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
        cv.imshow('Video',frame)
        if cv.waitKey(20) & 0xFF==ord('d'): #Detener con "d"
            break


if __name__ == '__main__':
    main()