import argparse
import os
from time import time
from datetime import datetime
import pickle
import faiss
import torch

import cv2
import numpy as np

from utils.load_model import SCRFD
from utils.process_data import take_box_detector, alignment, process_kps
import matplotlib.pyplot as plt

test_image_path = r'E:\python\DA_me\lfw'
save_image_dir_good = r'E:\python\DA_me\detect_output_lfw\good'
save_image_dir_bad = r'E:\python\DA_me\detect_output_lfw\bad'
# test_image_path = '/home/maicg/Documents/Me/face-recognition-demo/test_detect_lfw'
# save_image_dir = '/home/maicg/Documents/Me/face-recognition-demo/save_test_detect_lfw'

def main():
    import glob
    #load moder 
    detector = SCRFD(model_file=r'E:\python\DA_me\onnx\scrfd_2.5g_bnkps.onnx')
    detector.prepare(-1)

    for dir in sorted(os.listdir(test_image_path)):
        pathdir = os.path.join(test_image_path,dir)
        for image in sorted(os.listdir(pathdir)):
            pathName = os.path.join(pathdir,image)

            img2 = cv2.imread(pathName)
            img2 = cv2.copyMakeBorder(img2, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=(255,255,255))
            # plt.imshow(img2)
            # plt.show()
            # img2 = change_brightness(img2, 1.0, 5)
            # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            bboxes, kpss = take_box_detector(img2, detector)
            cout = 0
            for i in range(bboxes.shape[0]):
                print("chay nay")
                bbox = bboxes[i]
                x1,y1,x2,y2,_ = bbox.astype(np.int32)
                _,_,_,_,score = bbox.astype(np.float32)
                print("x1,y1,x2,y2: ", x1,y1,x2,y2)
                if x1 < 0 or x2 < 0 or y1 < 0 or y2 < 0:
                    print("error outside")
                    # continue
                print("cout: ", cout)
                if cout == 0:
                    crop_img = img2[y1:y2, x1:x2]
                    # plt.imshow(crop_img)
                    # plt.show()
                    if kpss is not None:
                        kps = kpss[i]
                        distance12, distance_nose1, distance_nose2, distance_center_eye_mouth, distance_nose_ceye, distance_nose_cmouth, distance_eye, distance_mouth, l_eye, r_eye, l_mouth, r_mouth = process_kps(kps)
                        if distance12 >= distance_nose1 and distance12 >= distance_nose2:
                                if distance_center_eye_mouth >= distance_nose_ceye and distance_center_eye_mouth >= distance_nose_cmouth:
                                    rotate_img = alignment(crop_img, l_eye, r_eye)
                                    rotate_img = cv2.resize(rotate_img, (112,112))
                                    save_path = save_image_dir_good + '/' + dir
                                    os.makedirs(save_path, exist_ok=True)
                                    print(save_path + '/' + str(image))
                                    # print("show anh")
                                    # plt.imshow(rotate_img)
                                    # plt.show()
                                    cv2.imwrite(save_path + '/' + str(image) , rotate_img)
                        else:
                            rotate_img = alignment(crop_img, l_eye, r_eye)
                            rotate_img = cv2.resize(rotate_img, (112,112))
                            save_path = save_image_dir_bad + '/' + dir
                            os.makedirs(save_path, exist_ok=True)
                            print(save_path + '/' + str(image))
                            # print("show anh")
                            # plt.imshow(rotate_img)
                            # plt.show()
                            cv2.imwrite(save_path + '/' + str(image) , rotate_img)
                cout = cout + 1
            if cout == 2:
                print("path error: ", image)

main()