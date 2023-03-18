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


def PointRotate(angle, valuex, valuey, pointx, pointy):
    valuex = np.array(valuex)
    valuey = np.array(valuey)
    Rotatex = (valuex - pointx) * math.cos(angle) - (valuey - pointy) * math.sin(angle) + pointx
    Rotatey = (valuex - pointx) * math.sin(angle) + (valuey - pointy) * math.cos(angle) + pointy
    return Rotatex, Rotatey


def getValue(img, lmk=[], type = 1, lmk_type=2, channels='YUV'):
    Value = []
    # 1.三点对齐 2.两点对齐
    # 1.81点 2.68 点
    h, w, c = img.shape
    # if channels == 'YUV':
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
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


def choose_windows(name='Hamming', N=20):
    # Rect/Hanning/Hamming
    if name == 'Hamming':
        window = np.array([0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1)) for n in range(N)])
    elif name == 'Hanning':
        window = np.array([0.5 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) for n in range(N)])
    elif name == 'Rect':
        window = np.ones(N)
    return window


def CHROM(STMap_CSI):
    LPF = 0.7  # low cutoff frequency(Hz) - specified as 40bpm(~0.667Hz) in reference
    HPF = 2.5  # high cutoff frequency(Hz) - specified as 240bpm(~4.0Hz) in reference
    WinSec = 1.6  # (was a 48 frame window with 30 fps camera)
    NyquistF = 15  # 30fps
    FS = 30 # 30fps
    FN = STMap_CSI.shape[0]
    B, A = signal.butter(3, [LPF/NyquistF, HPF/NyquistF], 'bandpass')
    WinL = int(WinSec * FS)
    if (WinL % 2):  # force even window size for overlap, add of hanning windowed signals
        WinL = WinL + 1
    if WinL <= 18:
        WinL = 20
    NWin = int((FN - WinL / 2) / (WinL / 2))
    S = np.zeros(FN)
    WinS = 0  # Window Start Index
    WinM = WinS + WinL / 2  # Window Middle Index
    WinE = WinS + WinL   # Window End Index
    #T = np.linspace(0, FN, FN)
    BGRNorm = np.zeros((WinL, 3))
    for i in range(NWin):
        #TWin = T[WinS:WinE, :]
        for j in range(3):
            BGRBase = np.nanmean(STMap_CSI[WinS:WinE, j])
            BGRNorm[:, j] = STMap_CSI[WinS:WinE, j]/(BGRBase+0.0001) - 1
        Xs = 3*BGRNorm[:, 2] - 2*BGRNorm[:, 1]  # 3Rn-2Gn
        Ys = 1.5*BGRNorm[:, 2] + BGRNorm[:, 1] - 1.5*BGRNorm[:, 0]  # 1.5Rn+Gn-1.5Bn

        Xf = signal.filtfilt(B, A, np.squeeze(Xs))
        Yf = signal.filtfilt(B, A, np.squeeze(Ys))

        Alpha = np.nanstd(Xf)/np.nanstd(Yf)
        SWin = Xf - Alpha*Yf
        SWin = choose_windows(name='Hanning', N=WinL)*SWin
        if i == 0:
            S[WinS:WinE] = SWin
            #TX[WinS:WinE] = TWin
        else:
            S[WinS: WinM - 1] = S[WinS: WinM - 1] + SWin[0: int(WinL/2) - 1]
            S[WinM: WinE] = SWin[int(WinL/2):]
            #TX[WinM: WinE] = TWin[WinL/2 + 1:]
        WinS = int(WinM)
        WinM = int(WinS + WinL / 2)
        WinE = int(WinS + WinL)
    return S


def POS(STMap_CSI):
    LPF = 0.7  # low cutoff frequency(Hz) - specified as 40bpm(~0.667Hz) in reference
    HPF = 2.5  # high cutoff frequency(Hz) - specified as 240bpm(~4.0Hz) in reference
    WinSec = 1.6  # (was a 48 frame window with 30 fps camera)
    NyquistF = 15  # 30fps
    FS = 30  # 30fps
    N = STMap_CSI.shape[0]
    l = int(WinSec * FS)
    H = np.zeros(N)
    Cn = np.zeros((3, l))
    P = np.array([[0, 1, -1], [-2, 1, 1]])
    for n in range(N-1):
        m = n - l
        if m >= 0:
            Cn[0, :] = STMap_CSI[m:n, 2]/np.nanmean(STMap_CSI[m:n, 2])
            Cn[1, :] = STMap_CSI[m:n, 1]/np.nanmean(STMap_CSI[m:n, 1])
            Cn[2, :] = STMap_CSI[m:n, 0]/np.nanmean(STMap_CSI[m:n, 0])
            S = np.dot(P, Cn)
            h = S[0, :] + ((np.nanstd(S[0, :])/np.nanstd(S[1, :]))*S[1, :])
            H[m: n] = H[m: n] + (h - np.nanmean(h))
    return H


def mySTMap(imglist_root, lmk_all=[]):
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
    # # CHROM
    # for w in range(STMap.shape[1]):
    #     STMap_CSI[:, w, 0] = np.squeeze(POS(STMap_CSI[:, w, :]))
    # Normal
    for c in range(STMap.shape[2]):
        for w in range(STMap.shape[1]):
            STMap[:, w, c] = 255 * ((STMap[:, w, c] - np.nanmin(STMap[:, w, c])) / (
                    0.001 + np.nanmax(STMap[:, w, c]) - np.nanmin(STMap[:, w, c])))
    STMap = np.swapaxes(STMap, 0, 1)
    STMap = np.rint(STMap)
    STMap = np.array(STMap, dtype='uint8')
    return STMap

def get_file(dir_path, file_type):
    for subfile in os.listdir(dir_path):
        if subfile.endswith(file_type):
            return subfile
    print(subfile)
    print('*************')
    return 0


if __name__ == '__main__':

    fileRoot = '/home/haolu/Data/BUAA/'
    STMap_name = 'STMap_RGB.png'
    file_list = os.listdir(fileRoot)
    z = 0

    Pic_num = []
    for subfile in file_list:
        print(z)
        z = z + 1
        now_path = os.path.join(fileRoot, subfile)
        lmk_path = os.path.join(now_path, 'Label/RGB_lmk.csv')
        # RGB_path = os.path.join(now_path, 'video.avi')
        RGB_path = os.path.join(now_path, 'Align')
        STMap_path = os.path.join(now_path, 'STMap')
        if not os.path.exists(STMap_path):
            os.makedirs(STMap_path)
        # 读取lmk
        lmk_all = []
        with open(os.path.join(lmk_path), "r") as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                lmk_all.append(line)

        STMap = mySTMap(RGB_path, lmk_all=lmk_all)
        cv2.imwrite(os.path.join(STMap_path, STMap_name), STMap, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        print(subfile)
