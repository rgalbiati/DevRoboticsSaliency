import cv2
import sys
import numpy as np

cap = cv2.VideoCapture('IMG_2156.MOV')
windowName = "Video"
windowMask = "Mask"
cv2.namedWindow(windowName)
cv2.namedWindow(windowMask)

brightnessColor = (255, 0, 0)
hsvColor = (0, 0, 255)
whiteColor = (255, 255, 255)

# ---------------------------- MOVEMENT FUNCTIONS ---------------------------- #
# Purpose: finds diff between three arrays and ands them together
def diffImg (t0, t1, t2):
        d1 = cv2.absdiff(t2, t1)
        d2 = cv2.absdiff(t1, t0)
        return cv2.bitwise_and(d1, d2)

# Purpose: creates and returns movement map where movement pixels are 
#       represented in white and all others are black
def findMovement (t_minus, t, t_plus) :
        t_minus = cv2.cvtColor(t_minus, cv2.COLOR_RGB2GRAY)
        t = cv2.cvtColor(t, cv2.COLOR_RGB2GRAY)
        t_plus = cv2.cvtColor(t_plus, cv2.COLOR_RGB2GRAY)

        movementImage = diffImg(t_minus, t, t_plus)
        mask = cv2.cvtColor(movementImage,cv2.COLOR_GRAY2BGR).copy()
        return mask

# ----------------------- HSV AND BRIGHTNESS FUNCTIONS ----------------------- #
# Purpose: finds the 100 brightest pixels and the 100 pixels with the highest 
#       combined saturation and value and returns the image with the pixels 
#       marked and a mask with the pixels marked
def findMaxVals (array, size) :
        indices =  np.argpartition(array.flatten(), -1)[-100:]
        max_vals = np.vstack(np.unravel_index(indices, array.shape)).T
        return max_vals

# Purpose: circles indices in max_val_indices on mask and image
def graphMaxVals (img, mask, color, max_val_indices) :
        for index in max_val_indices :
                cv2.circle(img, (index[1], index[0]), 15, color, 2)
                cv2.circle(mask, (index[1], index[0]), 5, whiteColor, -1)
        return img, mask

# Purpose: Finds brightest and most saturated pixels on image
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
# Purpose: sets initial images 
def  Init ():
        global cap
        ret_minus, frame_minus = cap.read()
        ret, frame = cap.read()
        ret_plus, frame_plus = cap.read()

        return frame_minus, frame, frame_plus

# Purpose: Gets a new frame from video
def getImage (curr_frame, next_frame) :
        global cap
        ret, new_frame = cap.read()
        return curr_frame, next_frame, new_frame

# Purpose: shows mask and image in seperate windows
def showImages(img, mask) :
        small_frame = cv2.resize(img, (0,0), fx=0.5, fy=0.5) 
        small_mask = cv2.resize(mask, (0,0), fx=0.5, fy=0.5) 
        cv2.imshow(windowMask,small_mask)
        cv2.imshow(windowName,small_frame)

# Purpose: closes video windows
def closeWebcam () :
        cv2.destroyWindow(windowMask)
        cv2.destroyWindow(videoWin)
        print ("Goodbye")

# ------------------------------ MAIN FUNCTIONS ------------------------------ #
def main() :
        frame_minus, frame, frame_plus = Init()

        while cap.isOpened():
                frame_minus, frame, frame_plus = getImage(frame, frame_plus)

                mask = findMovement(frame_minus.copy(), frame.copy(), 
                                        frame_plus.copy())

                combine = getSaturationArray(frame)
                gray = getGrayImg(frame)
                
                frame, mask = findIntensePixels(gray, combine, frame, mask)

                showImages(frame, mask)

                if cv2.waitKey(10) == 27 :
                        break

        closeWebcam()
        sys.exit(0)


if __name__=="__main__" :
        main()