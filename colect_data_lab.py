import imp
from types import new_class
import numpy as np
import matplotlib.pyplot as plt
from pynput import keyboard
import time
import os
import cv2
import random


# chuong trinh auto chay
# - trong chuong trinh auto chay hien ra dong chu: neu ban muon diem danh thi nhan 1
# - trong chuong trinh auto chay khi xuat hien ky tu 1 se chup anh va luu vao folder rieng
# - Neu ko co tin hieu gi tu ban phim se tiep tuc lay du lieu vs time = 1h

def on_press(key):
    if key == keyboard.Key.esc:
        return False  # stop listener
    try:
        k = key.char  # single-char keys
    except:
        k = key.name  # other keys
    if k in ['1', '2', '3', 'i']:  # keys of interest
        # self.keys.append(k)  # store it in global-like variable
        # print('Key pressed: ' + k)
        take_image()
        return False  # stop listener; remove this if want more keys

def make_resize(cap):
    cap.set(3,1920)
    cap.set(4,1080)

cam_port = 0
def take_image():
    print("Chi nhap ten rieng cua ban, viet khong dau: ")
    name = str(input())
    title = random.random()
    cam = cv2.VideoCapture(cam_port)
    make_resize(cam) 
    result, image = cam.read()
    time.sleep(1)
    if result:
        print("------ dang lay anh -----")
        pkg =  str(title)
        cv2.imwrite("train_test/Face/"+ name + pkg +".jpg", image)
        print("***Lay anh thanh cong***")
    else:
        print("Error!!!!!!!!!!!!!!!")


while(True):    
    print("Nhap 1 de tiep tuc: ")
    # listener = keyboard.Listener(on_press=on_press)
    # listener.start()  # start to listen on a separate thread
    # listener.join()  # remove if main thread is polling self.keys 

    action = int(input())  
    if action == 1:
        take_image()  

