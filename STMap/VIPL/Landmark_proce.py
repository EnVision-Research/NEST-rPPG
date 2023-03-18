import sys
import re
import os
import shutil
import scipy.io as io
import xlrd
import csv
import cv2
import numpy as np
import copy
from scipy import interpolate

# Function: The abnormal landmarks are interpolated using sequential continuity.

fileRoot = r'/home/hlu/Data/VIPL'
file_list = os.listdir(fileRoot)
Pic_num = []
for subfile in file_list:
    now_path = os.path.join(fileRoot, subfile)
    Video_path = os.path.join(now_path, 'video.avi')
    lmk_path = os.path.join(now_path, os.path.join('gt', 'RGB_lmk.csv'))
    new_lmk_path = os.path.join(now_path, os.path.join('gt', 'new_lmk.csv'))
    # Read lmk
    lmk_all = []
    with open(os.path.join(lmk_path), "r") as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lmk_all.append(line)

    # Number of frames read
    cap = cv2.VideoCapture(Video_path)
    Num = int(cap.get(7))
    cap.release()

    # Save LMKS of normal detection
    lmk_all_new = []
    time_new = []
    time = np.linspace(0, Num, Num)
    new_num = 0
    for lmk_index in range(Num):
        lmk = np.array(lmk_all[lmk_index])
        if not (round(float(lmk[66])) == 0):
            lmk_all_new.append(lmk)
            time_new.append(time[lmk_index])
    # Interpolate for abnormal lmk
    lmk_final = []
    time_new = np.array(time_new).reshape(-1)
    lmk_all_new = np.array(lmk_all_new)
    for lmk_index in range(68*2):
        lmk_temp = lmk_all_new[:, lmk_index].reshape(-1)
        t = interpolate.splrep(time_new, lmk_temp)
        lmk_final.append(interpolate.splev(time, t))
    lmk_final = np.array(lmk_final).T
    with open(new_lmk_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in lmk_final:
            writer.writerow(row)
    print('keyi')


