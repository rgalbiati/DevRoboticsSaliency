import cv2
import sys
import numpy as np

cap = None
windowVideo = "Video"
windowMask = "Mask"
cv2.namedWindow(windowVideo)
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
        movement_cont = contour(movementImage)

        # REMOVE LATER mask = cv2.cvtColor(movementImage,cv2.COLOR_GRAY2BGR).copy()
        return movement_cont

def contour (mask) :
        ret, thresh = cv2.threshold(mask, 20, 255, 0) # play with threshold

        im, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        return im

# ----------------------- HSV AND BRIGHTNESS FUNCTIONS ----------------------- #
# Purpose: finds the 100 brightest pixels and the 100 pixels with the highest 
#       combined saturation and value and returns the image with the pixels 
#       marked and a mask with the pixels marked
def findMaxVals (array, size) :
        indices =  np.argpartition(array.flatten(), -1)[-500:]
        max_vals = np.vstack(np.unravel_index(indices, array.shape)).T
        return max_vals

# Purpose: circles indices in max_val_indices on mask and image
def graphMaxVals (img, mask, color, max_val_indices) :
        for index in max_val_indices :
                # Vec3b & color2 = img.at<Vec3b>(index[1], index[0]);
                # color2[2] = 13;
                # Vec3b color2 = img.at<Vec3b>(Point(index[1], index[0]))
                # img.at<Vec3b>(Point(index[1], index[0])) = color2
                img[index[0], index[1]] = (255, 255, 255)
                mask[index[0], index[1]] = (255, 255, 255)
                # cv2.circle(img, (index[1], index[0]), 15, color, 2)
                # cv2.circle(mask, (index[1], index[0]), 5, whiteColor, -1)
        return img, mask

# Purpose: Finds brightest and most saturated pixels on image
def findIntensePixels(gray, combine, img, size):
        sat_mask = np.zeros((size[0],size[1],3), np.uint8)
        bright_mask = np.zeros((size[0],size[1],3), np.uint8)

        max_indices = findMaxVals (gray, size)
        img, bright_mask = graphMaxVals(img, bright_mask, brightnessColor, max_indices)

        max_indices = findMaxVals (combine, size)
        img, sat_mask = graphMaxVals(img, sat_mask, hsvColor, max_indices)

        return img, sat_mask, bright_mask

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
def nothing(x):
    pass

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
        return curr_frame, next_frame, new_frame, ret

def func (name, mask) :
        cv2.createTrackbar(name, windowMask, 255, 255, nothing)

        # get current positions of trackbars
        importance = cv2.getTrackbarPos(name,windowMask)

        if importance == 0 :
                importance = 1

        ret, mask = cv2.threshold(mask, 0, importance, cv2.THRESH_BINARY)
        
        return mask
        
# Purpose: shows mask and image in seperate windows
def showImages(img, mov_mask, sat_mask, bright_mask) :
        mov_mask = func('Movement', mov_mask)
        sat_mask = func('Saturation', sat_mask)
        bright_mask = func('Brightness', bright_mask)

        mask = np.asarray(sat_mask) | np.asarray(bright_mask) | np.asarray(mov_mask)

        surf_mask = surf(mask)
        mask = surf_mask # delete! 

        img = cv2.flip(img, 1)
        img = cv2.flip(img, 1)
        mask = cv2.flip(mask, 1)
        mask = cv2.flip(mask, 1)
        small_frame = cv2.resize(img, (0,0), fx=0.5, fy=0.5) 
        small_mask = cv2.resize(mask, (0,0), fx=0.5, fy=0.5) 
        cv2.imshow(windowMask,small_mask)
        cv2.imshow(windowVideo,small_frame)

# Purpose: closes video windows
def closeWebcam () :
        cv2.destroyWindow(windowMask)
        cv2.destroyWindow(windowVideo)
        print ("Goodbye")

# ------------------------------ SURF FUNCTIONS ------------------------------ #
def surf (img) :
        thresh = 400
        surf = cv2.xfeatures2d.SURF_create(thresh)

        # Find keypoints and descriptors directly
        kp, des = surf.detectAndCompute(img,None)

        if len(kp) > 50 :
                while (len(kp) > 50) :
                        thresh = thresh * 5
                        surf.setHessianThreshold(thresh * 5)

                        kp, des = surf.detectAndCompute(img,None)

        img2 = cv2.drawKeypoints(img,kp,None,(255,0,0),4)
        return img2
        # surf.upright = True
        # kp = surf.detect(img,None)
        # img2 = cv2.xfeatures2d.drawKeypoints(img,kp,None,(255,0,0),4)







# TODO USAGE INSTRUCTIONS / USAGE ERROR HANDLEING 
# ------------------------------ MAIN FUNCTIONS ------------------------------ #
def main() :
        global cap
        vidName = sys.argv[1]

        cap = cv2.VideoCapture(vidName)

        frame_minus, frame, frame_plus = Init()
        size = frame.shape[:2]

        while cap.isOpened():
                mask = np.zeros((size[0],size[1],3), np.uint8)
                frame_minus, frame, frame_plus, ret = getImage(frame, frame_plus)
                
                if not ret :
                        break

                movement = findMovement(frame_minus.copy(), frame.copy(), 
                                        frame_plus.copy())


                combine = getSaturationArray(frame)
                gray = getGrayImg(frame)
                
                frame, sat_mask, bright_mask = findIntensePixels(gray, combine, frame, size)

     
                sat_mask = cv2.cvtColor(sat_mask, cv2.COLOR_RGB2GRAY)
                bright_mask = cv2.cvtColor(bright_mask, cv2.COLOR_RGB2GRAY)

        
                showImages(frame, movement, sat_mask, bright_mask)

                if cv2.waitKey(10) == 27 :
                        break

        closeWebcam()
        sys.exit(0)


if __name__=="__main__" :
        main()