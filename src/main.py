import cv2
import os
import time
import pickle
import numpy as np
import faiss
import torch
import torchvision
import pyrealsense2 as rs
import pandas as pd

from utils.load_model import SCRFD, load_model_onnx
from utils.process_data import take_box_detector, alignment, process_kps, process_onnx
from utils.check_kc import check_kc
from utils.process_ID import process_ID


def main():
    # khoi tao kc
    # Khởi tạo kết nối với camera
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    # Khởi tạo bộ lọc cho ảnh độ sâu
    depth_filter = rs.align(rs.stream.color)

    #file csv
    classes_data = pd.read_csv('DSDB.csv')
    import glob
    #load moder
    detector = SCRFD(model_file='./onnx/scrfd_2.5g_bnkps.onnx')
    detector.prepare(-1)
    BACKBONE = load_model_onnx('./onnx/Resnet2F.onnx')
    QUALITY = load_model_onnx('./onnx/quality.onnx')

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
    faceEncode = list(faceEncode)
    # print("ty==========", type(faceEncode))
    # cam_port=1
    # cap = cv2.VideoCapture(cam_port)
    cap = cv2.VideoCapture(1)
    # cap = cv2.VideoCapture('/home/maicg/Documents/python-image-processing/people_check.mp4')
    
    path = './data_test/origin'
    path_dir = './dataTest/t1'
    k=0
    name_id = len(labelOriginSet)
    print("name ID: ", name_id)
    if cap.isOpened():
        while True:
            frames = pipeline.wait_for_frames()
            for i in range(4):
                try:
                    result, img = cap.read()
                except KeyboardInterrupt:
                    print("error image")
            if check_kc(frames,depth_filter) < 0.5:
                try:
                    bboxes, kpss = take_box_detector(img, detector)
                except:
                    print("erro boxxes")

                h, w, c = img.shape
                area_base = h*w
                tl = 0
                tl1 = 0
                for i in range(bboxes.shape[0]):
                    time_start = time.time()
                    bbox = bboxes[i]
                    x1,y1,x2,y2,_ = bbox.astype(int)
                    _,_,_,_,score = bbox.astype(float)

                    crop_img = img[y1:y2, x1:x2]
                    
                    h1 = int(crop_img.shape[0])
                    w1 = int(crop_img.shape[1])
                    area_crop = h1*w1
                    if kpss is not None:
                        kps = kpss[i]
                        distance12, distance_nose1, distance_nose2, distance_center_eye_mouth, distance_nose_ceye, distance_nose_cmouth, distance_eye, distance_mouth, l_eye, r_eye = process_kps(kps)
                        if (distance_nose1-distance_nose2) <= 0:
                            # print("=====================dt1,dt2",distance_nose1,distance_nose2)
                            tl = distance_nose1/distance_nose2
                        else: 
                            # print("else=====================dt1,dt2",distance_nose1,distance_nose2)
                            tl = distance_nose2/distance_nose1
                        
                        if (distance_nose_ceye - distance_nose_cmouth) <= 0:
                            tl1 = distance_nose_ceye/distance_nose_cmouth
                        else:
                            tl1 = distance_nose_cmouth/distance_nose_ceye

                        # print(tl)
                        if area_crop == 0:
                            break
                        elif (area_base/area_crop) > ((1080*1920)/(64*64)):
                            print("hinh nho")
                            cv2.putText(img, 'Hinh nho', (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,255), thickness=2)
                        else:
                            if distance12 >= distance_nose1 and distance12 >= distance_nose2:
                                if distance_center_eye_mouth >= distance_nose_ceye and distance_center_eye_mouth >= distance_nose_cmouth:
                                    # if tl >= 0.6 and tl1 >= 0.6
                                    rotate_img = alignment(crop_img, l_eye, r_eye)
                                    rotate_img = cv2.resize(rotate_img, (112,112))
                                    try:
                                        quality, emb = process_onnx(rotate_img, BACKBONE, QUALITY)
                                    except:
                                        continue
                                    # print(emb.shape)
                                    emb = emb.cpu().detach().numpy()
                                    emb = np.array(emb,dtype=np.float32)
                                    w, result = face_index.search(emb, k=1)
                                    label = [labelOriginSet[i] for i in result[0]]
                                    if quality[0] > 0.25:
                                        if w[0][0] >= 0.55:
                                            directory = label[0]
                                            # print(directory)
                                            robot_talk = process_ID(directory, classes_data)
                                            print("robot: ", robot_talk)
                                            cv2.putText(img, directory, (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,255), thickness=2 )
                                            try:
                                                dir_fold = os.path.join(path_dir, directory)
                                                os.makedirs(dir_fold, exist_ok = True)
                                                frame_img_path = dir_fold + '/frame' + str(k) + '_' + str(round(quality[0], 4)) + '_' + str(round(w[0][0], 2))  + '.jpg'
                                                # print(frame_img_path)
                                                # img_save = cv2.resize(img_detect, (160,160))
                                                cv2.imwrite(frame_img_path, rotate_img)
                                                print("Directory created successfully")
                                                k=k+1
                                            except OSError as error:
                                                print("Directory can not be created")
                                    else:
                                        print("unknow")
                                        cv2.putText(img, 'unknow', (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,255), thickness=2 )

                            else:
                                try: 
                                    cv2.putText(img, 'unknow', (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,255), thickness=2 )
                                    output_path = 'quality_result_bad_reject'
                                    dir_fold = os.path.join(path_dir, output_path)
                                    os.makedirs(dir_fold, exist_ok = True)
                                    frame_img_path = dir_fold + '/frame' + str(k) + '_bad.jpg'
                                    cv2.imwrite(frame_img_path, img)
                                except OSError as error:
                                    print("Directory can not be created")

                    time_end = time.time()
                    avr_time = round(((time_end-time_start)), 2)
                    # print(avr_time)                                   
                                        
                    # display video
                    cv2.rectangle(img, (x1,y1)  , (x2,y2) , (255,0,0) , 2)
            else:
                cv2.putText(img, 'ERORR!!!', (550,550), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,255), thickness=2 )
            cv2.imshow("image", img)
            if cv2.waitKey(1) == ord('q'):
                break
        cap.release()
    pipeline.stop()
    cv2.destroyAllWindows()

main()