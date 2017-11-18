# import the necessary packages
import cv2
import numpy as np

def main (): 
        cam = cv2.VideoCapture(0)
        s, img = cam.read()

        cv2.namedWindow("mask")
        cv2.namedWindow("image")

        while s:
                img = cv2.flip(img, 1)
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

                # define range of blue color in HSV
                # lower_value = np.array([0,0,0])
                # upper_value = np.array([255,255,225])

                # Threshold the HSV image to get only blue colors
                # mask = cv2.inRange(hsv, lower_value, upper_value)

                # Bitwise-AND mask and original image
                # res = cv2.bitwise_and(img,img, mask= mask)
                
                blur = cv2.GaussianBlur(hsv, (41, 41), 0)
                
                # value_img = blur.copy()
                # sat_img = blur.copy()


                mask = np.zeros(hsv.shape[:2])
                combine = np.add(hsv[:, :, 1], hsv[:,:,2])
                # np.set_printoptions(threshold=np.inf)
                # print (combine)
                # cv2.imwrite("image.png", img)               
                for itter in range (100) :
                        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(combine)

                        cv2.circle(mask, maxLoc, 5, (255, 255, 255), -1)
                        # cv2.circle(mask, s_maxLoc, 5, (255, 255, 255), -1)
                    
                        cv2.circle(combine, maxLoc, 5, (0, 0, 0), -1)
                        # cv2.circle(value_img, v_maxLoc, 5, (0, 0, 0), -1)
                        # cv2.circle(sat_img, s_maxLoc, 5, (255, 255, 255), -1)

                        cv2.circle(hsv, maxLoc, 10, (255, 0, 0), 2)
                        # cv2.circle(hsv, s_maxLoc, 10, (0, 0, 255), 2)



                # (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(blur[:,:,2])
                # cv2.circle(hsv, maxLoc, 20, (255, 0, 255), 15)

                cv2.imshow( "mask",mask )
                cv2.imshow( "image",hsv )

                s, img = cam.read()
                key = cv2.waitKey(10)
                
                if key == 27 :
                        cv2.destroyWindow("mask")
                        cv2.destroyWindow("image")
                        break
        # print ("Goodbye")

if __name__=="__main__" :
        main()
