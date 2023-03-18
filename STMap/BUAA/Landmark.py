import cv2
import os
import numpy as np
import shutil
import face_alignment
import pandas as pd
import json
import scipy.io as scio
from scipy import interpolate
import csv
# Function: calculate lmk, align the time and get the labels



def get_file(dir_path, file_type):
    for subfile in os.listdir(dir_path):
            if subfile.endswith(file_type):
                return subfile
    print(subfile)
    print('*************')
    return 0


fileRoot = '/home/haolu/Data/BUAA/'
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cuda:4')
file_list_p = os.listdir(fileRoot)
z = 0
for subfile_p in file_list_p[12:]:
    now_path_p = os.path.join(fileRoot, subfile_p)
    file_list = os.listdir(now_path_p)
    for subfile in file_list:

        print(z)
        z = z + 1
        now_path = os.path.join(now_path_p, subfile)
        video_path = os.path.join(now_path, get_file(now_path, 'avi'))
        print('video_path')
        print(video_path)
        save_path = os.path.join(now_path, 'Label')
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        # 读取 video
        cap = cv2.VideoCapture(video_path)
        Num = cap.get(7)
        Num = int(Num)
        # 读取 time
        # 计算lmk
        # 计算lmk
        lmk = []
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                preds = fa.get_landmarks(frame)
                if preds is None:
                    lmk.append([0 for _ in range(136)])
                else:
                    preds = preds[0]
                    lmk.append(preds.reshape(136))
            else:
                break
        with open(os.path.join(save_path, 'RGB_lmk.csv'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in lmk:
                writer.writerow(row)
        print(now_path)
        cap.release()


