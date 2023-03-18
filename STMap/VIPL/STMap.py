import sys
import re
import os
import shutil
import scipy.io as io
import xlrd
import math
import csv
import cv2
import numpy as np
from math import *
from scipy import signal
import scipy.io as scio
from scipy import interpolate
from scipy import signal

# Function:  Generate a STMap (30FPS) with aligned faces and timestamps

def PointRotate(angle,valuex,valuey,pointx,pointy):
    valuex = np.array(valuex)
    valuey = np.array(valuey)
    Rotatex = (valuex - pointx) * math.cos(angle) - (valuey - pointy) * math.sin(angle) + pointx
    Rotatey = (valuex - pointx) * math.sin(angle) + (valuey - pointy) * math.cos(angle) + pointy
    return Rotatex, Rotatey



def getValue(img, lmk=[], type = 1, lmk_type=2):

    ##  type: 1 Align the by three points 2. Align the by two points
    ##  lmk_type: 1.81 landmarks 2. 68 landmarks
    Value = []
    h, w, c = img.shape
    if type == 1:
        w_step = int(w / 5)
        h_step = int(h / 5)
        for w_index in range(5):
            for h_index in range(5):
                temp = img[h_index * h_step: (h_index + 1) * h_step, w_index * w_step:(w_index + 1) * w_step, :]
                temp1 = np.nanmean(np.nanmean(temp, axis=0), axis=0)
                Value.append(temp1)
    elif type == 2:
        lmk = np.array(lmk, np.float32).reshape(-1, 2)
        min_p = np.min(lmk, 0)
        max_p = np.max(lmk, 0)
        min_p = np.maximum(min_p, 0)
        max_p = np.minimum(max_p, [w - 1, h-1])
        if lmk_type == 1:
            left_eye = lmk[0:8]
            right_eye = lmk[9:17]
            left = np.array([lmk[60], lmk[62], lmk[65]])
            right = np.array([lmk[61], lmk[63], lmk[73]])
        else:
            left_eye = lmk[36:41]
            right_eye = lmk[42:47]
            left = np.array([lmk[0], lmk[1], lmk[2]])
            right = np.array([lmk[14], lmk[15], lmk[16]])
        left_eye = np.nanmean(left_eye, 0)
        right_eye = np.nanmean(right_eye, 0)
        left = np.nanmean(left, 0)
        right = np.nanmean(right, 0)
        top = max((left[1] + right[1])/2 - 0.5*(max_p[1] - (left[1] + right[1])/2), 0)
        rotate_angular = math.atan((right_eye[1] - left_eye[1]) / (0.00001+right_eye[0] - left_eye[0])) * (180 / math.pi)
        # 旋转
        cent_point = [w/2, h/2]
        matRotation = cv2.getRotationMatrix2D((w/2, h/2), rotate_angular, 1)
        face_rotate = cv2.warpAffine(img, matRotation, (w, h))
        left[0], left[1] = PointRotate(math.radians(rotate_angular), left[0], left[1], cent_point[0], cent_point[1])
        right[0], right[1] = PointRotate(math.radians(rotate_angular), right[0], right[1], cent_point[0], cent_point[1])
        # 截取
        face_crop = face_rotate[int(top):int(max_p[1]), int(left[0]):int(right[0]), :]
        # cv2.imshow('a', face_crop)
        # cv2.waitKey(0)
        h, w, c = face_crop.shape
        w_step = int(w / 5)
        h_step = int(h / 5)
        for w_index in range(5):
            for h_index in range(5):
                temp = face_crop[h_index * h_step: (h_index + 1) * h_step, w_index * w_step:(w_index + 1) * w_step, :]
                temp1 = np.mean(np.mean(temp, axis=0), axis=0)
                Value.append(temp1)
    return np.array(Value)



def mySTMap(imglist_root, lmk_all=[], Time=[]):
    # b, a = signal.butter(5, 0.12 / (30 / 2), 'highpass')
    # b, a = signal.butter(5, [0.5 / (30 / 2), 3 / (30 / 2)], 'bandpass')
    img_list = os.listdir(imglist_root)
    z = 0
    STMap = []
    for imgPath_sub in img_list:
        now_path = os.path.join(imglist_root, imgPath_sub)
        img = cv2.imread(now_path)
        Value = getValue(img, lmk=lmk_all[z])
        if np.isnan(Value).any():
            Value[:, :] = 100
        STMap.append(Value)
        z = z + 1
    STMap = np.array(STMap)
    CSI_Time = np.arange(0, Time[-1], 33.3333)
    STMap_CSI = np.zeros((len(CSI_Time), STMap.shape[1], STMap.shape[2]))
    # CSI
    for c in range(STMap.shape[2]):
        for w in range(STMap.shape[1]):
            # STMap[:, w, c] = signal.filtfilt(b, a, np.squeeze(STMap[:, w, c]+0.01))
            t = interpolate.splrep(Time, STMap[:, w, c])
            STMap_CSI[:, w, c] = interpolate.splev(CSI_Time, t)

    # Normal
    for c in range(STMap.shape[2]):
        for w in range(STMap.shape[1]):
            STMap_CSI[:, w, c] = 255 * ((STMap_CSI[:, w, c] - np.nanmin(STMap_CSI[:, w, c])) / (
                    0.001 + np.nanmax(STMap_CSI[:, w, c]) - np.nanmin(STMap_CSI[:, w, c])))
    STMap_CSI = np.swapaxes(STMap_CSI, 0, 1)
    STMap_CSI = np.rint(STMap_CSI)
    STMap_CSI = np.array(STMap_CSI, dtype='uint8')
    return STMap_CSI

if __name__ == '__main__':
    fileRoot = r'/home/hlu/Data/VIPL'
    STMap_name = 'STMap.png'
    file_list = os.listdir(fileRoot)
    z = 0
    Pic_num = []
    for subfile in file_list:
        print(z)
        z = z + 1
        now_path = os.path.join(fileRoot, subfile)
        lmk_path = os.path.join(now_path, 'gt/new_lmk.csv')
        Time_path = os.path.join(now_path, 'gt/Timestamp.mat')
        Time = scio.loadmat(Time_path)
        Time = Time['Timestamp'].flatten()
        RGB_path = os.path.join(now_path, 'Align')
        STMap_path = os.path.join(now_path, 'STMap')
        lmk_all = []
        with open(os.path.join(lmk_path), "r") as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                lmk_all.append(line)
        if not os.path.exists(STMap_path):
            os.makedirs(STMap_path)
        STMap = mySTMap(RGB_path, lmk_all=lmk_all, Time=Time)
        cv2.imwrite(os.path.join(STMap_path, STMap_name), STMap, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        print(subfile)
