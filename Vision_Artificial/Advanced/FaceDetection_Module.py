import cv2 as cv
import mediapipe as mp
import time as tm


class FaceDetection():
    def __init__(self, min_detection_confidence=0.5, model_selection=0):
        self.min_detection_confidence = min_detection_confidence
        self.model_selection = model_selection
        self.mpFaceDetection  = mp.solutions.face_detection
        self.faceDetection  = self.mpFaceDetection.FaceDetection(self.min_detection_confidence,self.model_selection) 
        self.mpDraw = mp.solutions.drawing_utils

    def findFaces(self,frame, draw = True):
        imgRGB = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
        results = self.faceDetection.process(imgRGB)
        if results.detections: #Cuando detecte las caras
            for id,detection in enumerate(results.detections):
                box = detection.location_data.relative_bounding_box
                h,w,c = frame.shape
                bbox = int(box.xmin*w), int(box.ymin*h) , \
                int(box.width*w), int(box.height*h)
                if draw:
                    cv.rectangle(frame,bbox,(255,0,255),2)
        return frame




def main():
    capture = cv.VideoCapture(0) #Fuente de video: Webcam
    #Contador de fotogramas
    pTime = 0
    cTime = 0
    detector = FaceDetection()
    while True:
        isTrue,frame = capture.read() #Lee el video frame a frame
        frame = detector.findFaces(frame)
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