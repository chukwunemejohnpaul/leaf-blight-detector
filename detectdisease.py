import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread('ricedisease6.jpg')
blurred = cv2.GaussianBlur(img,(5,5),-1)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


#bb = cv2.medianBlur(hsv,3,1)
## mask of green (36,25,25) ~ (86, 255,255)
# mask = cv2.inRange(hsv, (36, 25, 25), (86, 255,255))
mask = cv2.inRange(hsv, (10, 25, 25), (31, 255,255))

## slice the green
imask = mask>0
green = np.zeros_like(img, np.uint8)
ggg = np.zeros_like(img, np.uint8)
green[imask] = img[imask]

_,thresh = cv2.threshold(mask,126,200,cv2.THRESH_BINARY)
kernel = np.ones((3,3),np.uint8)
#opening = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel, iterations = 2)

_,contours, _ = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)



image = cv2.drawContours(img,contours,-1,(0,0,255),2)

for cnt in contours:
	(x,y,w,h) = cv2.boundingRect(cnt)
	cv2.rectangle(image,(x,y),(x+w,y+h),(200,45,67),2)
cv2.putText(img,"rice bleat detected", (30, 50),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)



cv2.imshow("pixels",green)
cv2.imshow("pix",image)
k = cv2.waitKey(0)
cv2.destroyAllWindows()