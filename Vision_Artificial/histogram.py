import cv2 as cv
import matplotlib.pyplot as plt
img = cv.imread('fotos/lady.jpg') 
cv.imshow('Imagen',img) 

#Calculo histograma o distribución de pixeles

plt.figure()
plt.title('Colour Histogram')
plt.xlabel('Bins')
plt.ylabel('# of pixels')
colours = ('r','g','b')

for i,col in enumerate(colours):
    hist = cv.calcHist([img],[i],None,[256],[0,256]) #Calculo por cada color de la imagen
    plt.plot(hist,color=col)
    plt.xlim([0,256])
plt.show()

cv.waitKey(0)
