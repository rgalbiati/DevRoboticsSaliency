from imutils import contours
from skimage import measure
import numpy as np
import argparse
import imutils
import cv2


def diffImg (t0, t1, t2):
        d1 = cv2.absdiff(t2, t1)
        d2 = cv2.absdiff(t1, t0)
        return cv2.bitwise_and(d1, d2)

def main() :
        cv2.namedWindow("mask")
        cv2.namedWindow("image")
        cam = cv2.VideoCapture(0)


        t_minus = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
        s, img = cam.read()
        t = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        t_plus = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)

        while s:

                mask = cv2.flip(diffImg(t_minus, t, t_plus),1)
                mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)

                img = cv2.flip(img, 1)

                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                blur = cv2.GaussianBlur(hsv, (41, 41), 0)
                half = np.full(hsv.shape[:2], 2)
                combine = np.add(np.divide(blur[:, :, 1], half), np.divide(blur[:, :, 2], half))

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

                t_minus = t
                t = t_plus
                s, img = cam.read()
                t_plus = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                key = cv2.waitKey(10)
                
                if key == 27 :
                        cv2.destroyWindow("mask")
                        cv2.destroyWindow("image")
                        break
        print ("Goodbye")

if __name__=="__main__" :
        main()