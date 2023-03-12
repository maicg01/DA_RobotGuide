import os
import math
import pickle
import cv2
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import faiss
import matplotlib.pyplot as plt

from utils.load_model import SCRFD, load_model_onnx
from utils.process_data import take_box_detector, alignment, process_kps, process_onnx

pathkk =  './data_input'
path_detect = "./save_detect"

def train(train_dir):
    import glob
    X=[]
    y=[]
    z=[]

    #load moder
    detector = SCRFD(model_file='./onnx/scrfd_2.5g_bnkps.onnx')
    detector.prepare(-1)
    BACKBONE = load_model_onnx('./onnx/Resnet2F.onnx')
    QUALITY = load_model_onnx('./onnx/Quality.onnx')

    for path_dir in os.listdir(train_dir):
        path_name = os.path.join(train_dir, path_dir)
        for image in os.listdir(path_name):
            pathName = os.path.join(path_name,image)

            img2 = cv2.imread(pathName)
            # plt.imshow(img2)
            # plt.show()
            # img2 = change_brightness(img2, 1.0, 5)
            # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            try:
                bboxes, kpss = take_box_detector(img2, detector)
            except:
                continue
            for i in range(bboxes.shape[0]):
                bbox = bboxes[i]
                x1,y1,x2,y2,_ = (np.int32)
                _,_,_,_,score = bbox.astype(np.float32)

                crop_img = img2[y1:y2, x1:x2]
                # plt.imshow(crop_img)
                # plt.show()
                if kpss is not None:
                    kps = kpss[i]
                    distance12, distance_nose1, distance_nose2, distance_center_eye_mouth, distance_nose_ceye, distance_nose_cmouth, distance_eye, distance_mouth, l_eye, r_eye = process_kps(kps)
                    
                    try:
                        rotate_img = alignment(crop_img, l_eye, r_eye)
                        rotate_img = cv2.resize(rotate_img, (112,112))
                        dir_fold = os.path.join(path_detect, path_dir)

                        os.makedirs(dir_fold, exist_ok = True)
                        frame_img_path = dir_fold + "/" + image
                        cv2.imwrite(frame_img_path, rotate_img)
                        _, emb = process_onnx(rotate_img, BACKBONE, QUALITY)
                        print(emb)
                    except:
                        continue

                    print(emb.shape)
                    emb = emb.cpu().detach().numpy()
                    emb = np.array(emb,dtype=np.float32).reshape(512,)
                    print(type(emb))
                    print(emb.shape)
                    # print(path_img)
                    if image.find(".jpg") > 0:
                        index_label = image.find(".jpg")
                        directory = image[:index_label]
                    else:
                        index_label = image.find(".JPG")
                        directory = image[:index_label]
                    print(path_dir)
                    print(directory)

                    X.append(emb)
                    y.append(path_dir)
                    z.append(directory)
    return X, y, z


X, y, z = train(train_dir=pathkk)
faceEncode = np.array(X,dtype=np.float32)
print(faceEncode.shape)
face_index = faiss.IndexFlatIP(512)
# add vector
face_index.add(faceEncode)

len_ID_person = [0, len(y)]
print(len_ID_person[0], len_ID_person[1])

with open('X.pkl', 'wb') as f:
    pickle.dump(X, f)  

with open('y.pkl', 'wb') as f:
    pickle.dump(y, f) 

with open('z.pkl', 'wb') as f:
    pickle.dump(z, f) 

with open('len_ID_person.pkl', 'wb') as f:
    pickle.dump(len_ID_person, f) 


