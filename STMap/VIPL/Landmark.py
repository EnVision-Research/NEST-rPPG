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

fileRoot = '/home/hlu/Data/VIPL'
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cuda:2')
file_list = os.listdir(fileRoot)
z = 0
for subfile in file_list:
    print(z)
    z = z + 1
    now_path = os.path.join(fileRoot, subfile)
    video_path = os.path.join(now_path, 'video.avi')
    HR_path = os.path.join(now_path, 'gt_HR.csv')
    SpO2_path = os.path.join(now_path, 'gt_SpO2.csv')
    wave_path = os.path.join(now_path, 'wave.csv')
    time_path = os.path.join(now_path, 'time.txt')
    GT_save = os.path.join(now_path, 'gt')
    if not os.path.exists(GT_save):
        os.mkdir(GT_save)
    # Read video
    cap = cv2.VideoCapture(video_path)
    Num = cap.get(7)
    Num = int(Num)
    # Read HR SPO2 BVP
    HR = pd.read_csv(HR_path)
    SpO2 = pd.read_csv(SpO2_path)
    Wave = pd.read_csv(wave_path)
    HR = HR.HR._values
    SpO2 = SpO2.SpO2._values
    Wave = Wave.Wave._values
    # Read time
    time = []
    if os.path.exists(time_path):
        with open(time_path, "r") as f:  # 打开文件
            time = f.read()
            time = time.split()
            time = list(map(int, time))
            scio.savemat(os.path.join(GT_save, 'Timestamp.mat'), {'Timestamp': time})
            if len(time) != Num:
                shutil.rmtree(now_path)
                # os.rmdir(now_path)
                print(GT_save)
                print(len(time))
                print(Num)
                print('cuowu')
            else:
                a = 0

    else:
        time = np.linspace(0, len(HR) * 1000, Num)
        scio.savemat(os.path.join(GT_save, 'Timestamp.mat'), {'Timestamp': time})
    # Save HR SPO2 BVP
    if len(time) > 5:
        t_HR = interpolate.splrep(np.linspace(0, time[-1], len(HR)), HR)
        HR_filed = interpolate.splev(time, t_HR)
        t_SPO2 = interpolate.splrep(np.linspace(0, time[-1], len(SpO2)), SpO2)
        SPO2_filed = interpolate.splev(time, t_SPO2)
        t_wave = interpolate.splrep(np.linspace(0, time[-1], len(Wave)), Wave)
        wave_filed = interpolate.splev(time, t_wave)
    else:
        print('CUOWU')
        t_HR = interpolate.splrep(np.linspace(0, 10, len(HR)), HR)
        HR_filed = interpolate.splev(np.linspace(0, 10, Num), t_HR)
        t_SPO2 = interpolate.splrep(np.linspace(0, 10, len(SpO2)), SpO2)
        SPO2_filed = interpolate.splev(np.linspace(0, 10, Num), t_SPO2)
        t_wave = interpolate.splrep(np.linspace(0, 10, len(Wave)), Wave)
        wave_filed = interpolate.splev(np.linspace(0, 10, Num), t_wave)
    scio.savemat(os.path.join(GT_save, 'HR.mat'), {'HR': HR_filed})
    scio.savemat(os.path.join(GT_save, 'SPO2.mat'), {'SPO2': SPO2_filed})
    scio.savemat(os.path.join(GT_save, 'BVP.mat'), {'BVP': wave_filed})

    # get lmk
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
    cap.release()
    with open(GT_save + '/RGB_lmk.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in lmk:
            writer.writerow(row)
    print(now_path)

