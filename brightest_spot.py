# code from https://www.pyimagesearch.com/2014/09/29/finding-brightest-spot-image-using-python-opencv/
# import the necessary packages
from imutils import contours
from skimage import measure
import numpy as np
import argparse
import imutils
import cv2
 
# construct the argument parse and parse the arguments


def main () :

        cam = cv2.VideoCapture(0)

        s, image = cam.read()
        
        winName = "Movement Indicator"
        cv2.namedWindow(winName)
        
        while s:
                image = cv2.flip(image, 1)

                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (11, 11), 0)

                # threshold the image to reveal light regions in the
                # blurred image
                thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1]
                # perform a series of erosions and dilations to remove
                # any small blobs of noise from the thresholded image
                thresh = cv2.erode(thresh, None, iterations=2)
                thresh = cv2.dilate(thresh, None, iterations=4)

                # perform a connected component analysis on the thresholded
                # image, then initialize a mask to store only the "large"
                # components
                labels = measure.label(thresh, neighbors=8, background=0)
                mask = np.zeros(thresh.shape, dtype="uint8")
                 
                # loop over the unique components
                for label in np.unique(labels):
                        # if this is the background label, ignore it
                        if label == 0:
                                continue
                 
                        # otherwise, construct the label mask and count the
                        # number of pixels 
                        labelMask = np.zeros(thresh.shape, dtype="uint8")
                        labelMask[labels == label] = 255
                        numPixels = cv2.countNonZero(labelMask)
                 
                        # if the number of pixels in the component is sufficiently
                        # large, then add it to our mask of "large blobs"
                        if numPixels > 300:
                                mask = cv2.add(mask, labelMask)
        #         orig = image.copy()
        #         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #         gray = cv2.GaussianBlur(gray, (41, 41), 0)
        #         (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
        #         image = orig.copy()
        #         cv2.circle(image, maxLoc, 41, (255, 0, 0), 2)
                cv2.imshow( winName, mask)
                
                s, image = cam.read()
                key = cv2.waitKey(10)
                if key == 27:
                        cv2.destroyWindow(winName)
                        break
        print ("Goodbye")

if __name__=="__main__" :
        main()