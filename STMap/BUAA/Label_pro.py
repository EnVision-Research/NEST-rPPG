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
import h5py
from scipy import signal


LPF = 0.7  # low cutoff frequency(Hz) - specified as 40bpm(~0.667Hz) in reference
HPF = 2.5  # high cutoff frequency(Hz) - specified as 240bpm(~4.0Hz) in reference
NyquistF = 15  # 15fps
FS = 30  # 30fps
B, A = signal.butter(3, [LPF / NyquistF, HPF / NyquistF], 'bandpass')
z = 0

fileRoot = '/home/haolu/Data/BUAA/'
file_list_p = os.listdir(fileRoot)
for subfile_p in file_list_p:
    now_path_p = os.path.join(fileRoot, subfile_p)
    file_list = os.listdir(now_path_p)
    for subfile in file_list:
        now_path = os.path.join(now_path_p, subfile)
        print(now_path)
        data_path = os.path.join(now_path, 'PPGData.mat')
        STMap_path = os.path.join(now_path, 'STMap/STMap_RGB.png')
        save_path = os.path.join(now_path, 'Label')
        temp = cv2.imread(STMap_path)
        Num = temp.shape[1]

        pulse = scio.loadmat(data_path)['PPG']['data'][0][0]
        print(pulse)
        pulse = np.array(np.array(pulse).astype('float32')).reshape(-1)

        print('len(pulse)', len(pulse))
        Time = np.linspace(0, Num, len(pulse))
        CSI_Time = np.linspace(0, Num, Num)

        t = interpolate.splrep(Time, pulse)
        pulse_csi = interpolate.splev(CSI_Time, t)


        pulse_csi = (pulse_csi - np.min(pulse_csi)) / (np.max(pulse_csi) - np.min(pulse_csi))
        EXG2_f = signal.filtfilt(B, A, pulse_csi)
        EXG2_f = (EXG2_f - np.min(EXG2_f)) / (np.max(EXG2_f) - np.min(EXG2_f))
        scio.savemat(os.path.join(save_path, 'BVP_Filt.mat'), {'BVP': EXG2_f})
        scio.savemat(os.path.join(save_path, 'BVP.mat'), {'BVP': pulse_csi})


