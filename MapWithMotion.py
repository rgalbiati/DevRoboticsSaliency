from imutils import contours
from skimage import measure
import numpy as np
import argparse
import imutils
import cv2
import sys
import heapq
import time 

maskWin = "Mask"
videoWin = "Video"
cv2.namedWindow(maskWin)
cv2.namedWindow(videoWin)
cam = cv2.VideoCapture(0)

brightnessColor = (255, 0, 0)
hsvColor = (0, 0, 255)
whiteColor = (255, 255, 255)

# ---------------------------- MOVEMENT FUNCTIONS ---------------------------- #
# Purpose: creates and returns movement map where movement pixels are 
#       represented in white and all others are black
def diffImg (t0, t1, t2):
        d1 = cv2.absdiff(t2, t1)
        d2 = cv2.absdiff(t1, t0)
        return cv2.bitwise_and(d1, d2)

# ----------------------- HSV AND BRIGHTNESS FUNCTIONS ----------------------- #
# Purpose: finds the 100 brightest pixels and the 100 pixels with the highest 
# combined saturation and value and returns the image with the pixels marked and 
# a mask with the pixels marked

def findMaxVals (array, size) :
        indices =  np.argpartition(array.flatten(), -1)[-100:]
        max_vals = np.vstack(np.unravel_index(indices, array.shape)).T
        return max_vals

def graphMaxVals (img, mask, color, max_val_indices) :
        for index in max_val_indices :
                cv2.circle(img, (index[1], index[0]), 15, color, 2)
                cv2.circle(mask, (index[1], index[0]), 5, whiteColor, -1)
        return img, mask

def findIntensePixels(gray, combine, img, mask):
        size = img.shape[:2]

        max_indices = findMaxVals (gray, size)
        img, mask = graphMaxVals(img, mask, brightnessColor, max_indices)

        max_indices = findMaxVals (combine, size)
        img, mask = graphMaxVals(img, mask, hsvColor, max_indices)

        return img, mask

# Purpose: returns an array with combined value and saturation values
def getSaturationArray (img) :
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        blur = cv2.GaussianBlur(hsv, (41, 41), 0)
        return blur[:, :, 1]

# Purpose: returns grayscale version of image
def getGrayImg (img) :
        orig = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (41, 41), 0)
        img = orig.copy()
        return gray

# ----------------------------- Camera Functions ----------------------------- #
# Purpose: gets new image and updates previous and next image for movement elem
def getImage () :
        global cam
       
        t_minus = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
        s, img = cam.read()
        t = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        t_plus = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
        return img, s, t_minus, t, t_plus

# Purpose: shows mask and image in seperate windows
def showImages(img, mask) :
        cv2.imshow(maskWin, mask)
        cv2.imshow(videoWin, img)

# Purpose: closes video windows
def closeWebcam () :
        cv2.destroyWindow(maskWin)
        cv2.destroyWindow(videoWin)
        print ("Goodbye")

# Purpose: Processes next frame of video
def processNextFrame() :
        
        img, s, t_minus, t, t_plus = getImage()
        
        if not s :
                closeWebcam()

        movementImage = cv2.flip(diffImg(t_minus, t, t_plus),1)
        mask = cv2.cvtColor(movementImage,cv2.COLOR_GRAY2BGR).copy()

        img = cv2.flip(img, 1)

        combine = getSaturationArray(img)
        gray = getGrayImg(img)
        
        img, mask = findIntensePixels(gray, combine, img, mask)

        showImages(img, mask)
        
        if cv2.waitKey(10) == 27 :
                closeWebcam()
        else : 
                processNextFrame()

# ------------------------------ MAIN FUNCTIONS ------------------------------ #
def main() :
        processNextFrame()

if __name__=="__main__" :
        main()