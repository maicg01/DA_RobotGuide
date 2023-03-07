import cv2
import os
import time
import pickle
import numpy as np
import faiss
import torch
import torchvision
import shutil
from facenet_pytorch import MTCNN, InceptionResnetV1

from facenetPreditctFunction import SCRFD, alignment, process_image_package, xyz_coordinates
from fuction_compute import process_onnx, load_model_onnx, take_image

path_dir = './dataTest/test1'
path_data = './dataTest/database'
k=0
def main():
    BACKBONE = load_model_onnx('./onnx/Resnet2F.onnx')
    QUALITY = load_model_onnx('./onnx/Quality.onnx')

    faceEncode = []
    labelOriginSet = []
    with open('X.pkl', 'rb') as f:
        faceEncode = pickle.load(f)

    with open('y.pkl', 'rb') as f:
        labelOriginSet = pickle.load(f)
    
    faceEncode = np.array(faceEncode,dtype=np.float32)
    # create index with faiss
    face_index = faiss.IndexFlatIP(512)
    # add vector
    face_index.add(faceEncode)
    for path_img in os.listdir(path_data):
        path_name = os.path.join(path_data, path_img)
        for img in os.listdir(path_name):
            img_name = os.path.join(path_name, img)
            rotate_img = cv2.imread(img_name)
            try:
                quality, emb = process_onnx(rotate_img, BACKBONE, QUALITY)
            except:
                continue
            print(emb.shape)
            emb = emb.cpu().detach().numpy()
            emb = np.array(emb,dtype=np.float32)
            w, result = face_index.search(emb, k=1)
            label = [labelOriginSet[i] for i in result[0]]

            if w[0][0] >= 0.4:
                directory = label[0]
                print(directory)
                try:
                    dir_fold = os.path.join(path_dir, directory)
                    os.makedirs(dir_fold, exist_ok = True)
                    frame_img_path = dir_fold + '/frame' + str(k) + '_' + str(round(w[0][0], 2))  + '.jpg'
                    print(frame_img_path)
                    # img_save = cv2.resize(img_detect, (160,160))
                    cv2.imwrite(frame_img_path, rotate_img)
                    print("Directory created successfully")
                    k=k+1
                except OSError as error:
                    print("Directory can not be created")
            else:
                print("unknow")
                output_path = 'quality_result_bad'
                dir_fold = os.path.join(path_dir, output_path)
                os.makedirs(dir_fold, exist_ok = True)
                frame_img_path = dir_fold + '/frame' + str(k) + '_' + str(round(quality[0], 4))  + '.jpg'
                cv2.imwrite(frame_img_path, rotate_img)


main()  