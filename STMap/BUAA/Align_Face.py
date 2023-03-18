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

def get_file(dir_path, file_type):
    for subfile in os.listdir(dir_path):
            if subfile.endswith(file_type):
                return subfile
    print(subfile)
    print('*************')
    return 0

fileRoot = '/home/haolu/Data/BUAA/'
file_list_p = os.listdir(fileRoot)
Pic_num = []
aaa = 0
ppp = 0
for subfile_p in file_list_p[10:11]:
    now_path_p = os.path.join(fileRoot, subfile_p)
    file_list = os.listdir(now_path_p)
    for subfile in file_list:
        print(aaa)
        print(ppp)
        aaa = aaa + 1
        ppp = ppp + 1
        now_path = os.path.join(now_path_p, subfile)
        Video_path = os.path.join(now_path, get_file(now_path, 'avi'))
        lmk_path = os.path.join(now_path, os.path.join('Label', 'RGB_lmk.csv'))
        Mask_path = os.path.join(now_path, 'Mask')
        Align_path = os.path.join(now_path, 'Align')
        Align_Mask_path = os.path.join(now_path, 'Align_Mask')
        if not os.path.exists(Mask_path):
            os.makedirs(Mask_path)
        if not os.path.exists(Align_path):
            os.makedirs(Align_path)
        if not os.path.exists(Align_Mask_path):
            os.makedirs(Align_Mask_path)
        # 读取lmk
        lmk_all = []
        with open(os.path.join(lmk_path), "r") as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                lmk_all.append(line)
        # 读取 video
        cap = cv2.VideoCapture(Video_path)
        Num = cap.get(7)
        Num = int(Num)
        z = 10000
        lmk_index = 0
        while (cap.isOpened()):
            ret, img_temp = cap.read()
            if ret:
                lmk = np.array(lmk_all[lmk_index]).reshape(-1, 2)
                lmk_index = lmk_index + 1
                img_Masked = copy.deepcopy(img_temp)
                Mask1 = np.zeros([img_temp.shape[0], img_temp.shape[1], img_temp.shape[2]], dtype="uint8")
                Mask2 = np.zeros([img_temp.shape[0], img_temp.shape[1], img_temp.shape[2]], dtype="uint8")
                Mask3 = np.zeros([img_temp.shape[0], img_temp.shape[1], img_temp.shape[2]], dtype="uint8")
                Mask4 = np.zeros([img_temp.shape[0], img_temp.shape[1], img_temp.shape[2]], dtype="uint8")
                Mask5 = np.zeros([img_temp.shape[0], img_temp.shape[1], img_temp.shape[2]], dtype="uint8")

                ROI1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 26, 22, 21, 17]
                ROI2 = [36, 17, 21, 39, 40, 41]  # eye
                ROI3 = [42, 22, 26, 45, 46, 47]  # eye
                ROI4 = [31, 33, 35, 30]  # noise
                ROI5 = [51, 53, 54, 55, 57, 59, 48, 49]  # mouse
                ROI_xy1 = []
                ROI_xy2 = []
                ROI_xy3 = []
                ROI_xy4 = []
                ROI_xy5 = []

                for i in range(len(ROI1)):
                    ROI_xy1.append([round(float(lmk[ROI1[i], 0])), round(float(lmk[ROI1[i], 1]))])
                for i in range(len(ROI2)):
                    ROI_xy2.append([round(float(lmk[ROI2[i], 0])), round(float(lmk[ROI2[i], 1]))])
                for i in range(len(ROI3)):
                    ROI_xy3.append([round(float(lmk[ROI3[i], 0])), round(float(lmk[ROI3[i], 1]))])
                for i in range(len(ROI4)):
                    ROI_xy4.append([round(float(lmk[ROI4[i], 0])), round(float(lmk[ROI4[i], 1]))])
                for i in range(len(ROI5)):
                    ROI_xy5.append([round(float(lmk[ROI5[i], 0])), round(float(lmk[ROI5[i], 1]))])

                cv2.fillPoly(Mask1, np.int32([ROI_xy1]), (255, 255, 255))
                cv2.fillPoly(Mask2, np.int32([ROI_xy2]), (255, 255, 255))
                cv2.fillPoly(Mask3, np.int32([ROI_xy3]), (255, 255, 255))
                cv2.fillPoly(Mask4, np.int32([ROI_xy4]), (255, 255, 255))
                cv2.fillPoly(Mask5, np.int32([ROI_xy5]), (255, 255, 255))
                cv2.bitwise_and(Mask1, cv2.bitwise_not(Mask2), Mask1)
                cv2.bitwise_and(Mask1, cv2.bitwise_not(Mask3), Mask1)
                cv2.bitwise_and(Mask1, cv2.bitwise_not(Mask4), Mask1)
                cv2.bitwise_and(Mask1, cv2.bitwise_not(Mask5), Mask1)
                cv2.bitwise_and(img_temp, Mask1, img_Masked)

                # 65 73 64
                old = np.array([[lmk[1, 0], lmk[1, 1]],
                                [lmk[15, 0], lmk[15, 1]],
                                [lmk[8, 0], lmk[8, 1]]],
                               np.float32)
                new = np.array([[0, 48],
                                [128, 48],
                                [64, 128]],
                               np.float32)
                M = cv2.getAffineTransform(old, new)
                Face_align = cv2.warpAffine(img_temp, M, (img_temp.shape[1], img_temp.shape[0]))
                Face_align_masked = cv2.warpAffine(img_Masked, M, (img_Masked.shape[1], img_Masked.shape[0]))
                Mask_now = os.path.join(Mask_path, str(z) + '.png')
                Align_now = os.path.join(Align_path, str(z) + '.png')
                Align_Mask_now = os.path.join(Align_Mask_path, str(z) + '.png')
                cv2.imwrite(Mask_now, Mask1, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                cv2.imwrite(Align_now, Face_align[0:128, 0:128, :], [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                # cv2.imwrite(Align_Mask_now, Face_align_masked[0:128, 0:128, :], [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                z = z + 1
            else:
                break

        cap.release()
        print(Video_path)