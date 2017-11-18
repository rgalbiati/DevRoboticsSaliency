import cv2
# import numpy as np

import numpy as np
from PIL import Image

# im = Image.open('img.png')
# im = im.convert('HSV')
# data = np.array(im)


# h1, s1, v1 = 255,200, 200
# h2, s2, v2 = 255, 0

# hue, sat, value = data[:,:,0], data[:,:,1], data[:,:,2]
# mask = (sat < s1) & (value < v1)

# data[:,:,:3][mask] = [_, s2, v2]
def main (): 
        cam = cv2.VideoCapture(0)
        s, img = cam.read()

        cv2.namedWindow("image")
        
        while s:
                img = cv2.flip(img, 1)

                img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

                # im = img.convert('RGBA')
                data = np.array(img)

                r1, g1, b1 = 180, 180, 180 # Original value
                r2, g2, b2, a2 = 0, 0, 0, 0 # Value that we want to replace it with

                red, green, blue, alpha = data[:,:,0], data[:,:,1], data[:,:,2], data[:,:,3]
                mask = (red > r1) & (green > g1) & (blue > b1)
                data[:,:,:4][mask] = [r2, g2, b2, a2]

                # hsv = cv2.cvtColor(data, cv2.COLOR_BGR2HSV)


                # im = Image.fromarray(data)
                cv2.imshow("image", data)
                # cv2.imshow("otherimage", img)


                s, img = cam.read()
                key = cv2.waitKey(10)
                
                if key == 27 :
                        cv2.destroyWindow("image")
                        break

        print ("Goodbye")

if __name__ == "__main__" :
        main()
# def main (): 
#         cam = cv2.VideoCapture(0)
#         s, img = cam.read()

#         cv2.namedWindow("image")

#         while s:
#                 img = cv2.flip(img, 1)
#                 hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#                 blur = cv2.GaussianBlur(hsv, (41, 41), 0)
                
#                 value_img = blur.copy()


#                 black_areas = (red < 70) & (blue < 70) & (green < 70)

#                 data[..., :-1][black_areas.T] = (255, 255, 255)

