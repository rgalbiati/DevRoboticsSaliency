# code inspired by from http://www.steinm.com/blog/motion-detection-webcam-python-opencv-differential-images/

import cv2
import numpy

def diffImg (t0, t1, t2):
        d1 = cv2.absdiff(t2, t1)
        d2 = cv2.absdiff(t1, t0)
        return cv2.bitwise_and(d1, d2)

def main ():
        cam = cv2.VideoCapture(0)
        winName = "Movement Indicator"
        cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
        
        # Read three images first:
        t_minus = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
        s, norm = cam.read()
        t = cv2.cvtColor(norm, cv2.COLOR_RGB2GRAY)
        t_plus = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)

        while True:
                blk = cv2.flip(diffImg(t_minus, t, t_plus),1)
                blk = cv2.cvtColor(blk,cv2.COLOR_GRAY2BGR)
                norm = cv2.flip(norm, 1)
                
                both = numpy.concatenate((blk, norm))
                small = cv2.resize(both, (0,0), fx=0.5, fy=0.5) 

                cv2.imshow( winName, small)
                # Read next image
                t_minus = t
                t = t_plus
                s, norm = cam.read()
                t_plus = cv2.cvtColor(norm, cv2.COLOR_RGB2GRAY)

                key = cv2.waitKey(10)

                if key == 27:
                        cv2.destroyWindow(winName)
                        break

        print ("Goodbye")
        return 0

if __name__ == "__main__" :
        main()

