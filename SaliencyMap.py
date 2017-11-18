from imutils import contours
from skimage import measure
import numpy as np
import argparse
import imutils
import cv2

def main() :
        cam = cv2.VideoCapture(0)
        s, img = cam.read()

        cv2.namedWindow("mask")
        cv2.namedWindow("image")

        while s:
                img = cv2.flip(img, 1)
                mask = np.zeros(img.shape[:2])

                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                blur = cv2.GaussianBlur(hsv, (41, 41), 0)
                combine = np.add(blur[:, :, 1], blur[:,:,2])

                orig = img.copy()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (41, 41), 0)
                img = orig.copy()

                for itter in range (100) :
                        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
                        cv2.circle(mask, maxLoc, 5, (0, 255, 0), -1)
                        cv2.circle(gray, maxLoc, 5, (0, 0, 0), -1)
                        cv2.circle(img, maxLoc, 10, (255, 0, 0), 2)

                        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(combine)
                        cv2.circle(mask, maxLoc, 5, (255, 255, 255), -1)
                        cv2.circle(combine, maxLoc, 5, (0, 0, 0), -1)
                        cv2.circle(img, maxLoc, 10, (0, 0, 255), 2)

                cv2.imshow("mask", mask)
                cv2.imshow("image", img)

                s, img = cam.read()
                key = cv2.waitKey(10)
                
                if key == 27 :
                        cv2.destroyWindow("mask")
                        cv2.destroyWindow("image")
                        break
        print ("Goodbye")

if __name__=="__main__" :
        main()