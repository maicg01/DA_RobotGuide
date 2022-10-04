from email.mime import image
import cv2

cam_p = 1
cam = cv2.VideoCapture(cam_p)

for i in range(20):
    result, image = cam.read()
    cv2.imshow("img", image)
    cv2.waitKey()