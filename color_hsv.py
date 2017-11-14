# import the necessary packages
import cv2
import numpy as np

def main (): 
        cam = cv2.VideoCapture(0)
        s, img = cam.read()
        winName = "Movement Indicator"

        cv2.namedWindow(winName)
        while s:
                img = cv2.flip(img, 1)
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

                # define range of blue color in HSV
                lower_value = np.array([50,50,110])
                upper_value = np.array([225,255,130])

                # Threshold the HSV image to get only blue colors
                mask = cv2.inRange(hsv, lower_value, upper_value)

                # Bitwise-AND mask and original image
                res = cv2.bitwise_and(img,img, mask= mask)

                cv2.imshow( winName,res )
                s, img = cam.read()
                key = cv2.waitKey(10)
                
                if key == 27 :
                        cv2.destroyWindow(winName)
                        break
        print ("Goodbye")

if __name__=="__main__" :
        main()
