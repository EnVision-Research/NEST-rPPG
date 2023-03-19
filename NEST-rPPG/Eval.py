import cv2
import os
import numpy as np
import shutil
import pandas as pd
import scipy.io as scio
from scipy import interpolate
import scipy.io as io

# Function: Calculate average HR per video

def MyEval(HR_pr, HR_rel):
    HR_pr = np.array(HR_pr).reshape(-1)
    HR_rel = np.array(HR_rel).reshape(-1)
    temp = HR_pr - HR_rel
    me = np.mean(temp)
    std = np.std(temp)
    mae = np.sum(np.abs(temp))/len(temp)
    rmse = np.sqrt(np.sum(np.power(temp, 2))/len(temp))
    mer = np.mean(np.abs(temp) / HR_rel)
    p = np.sum((HR_pr - np.mean(HR_pr))*(HR_rel - np.mean(HR_rel))) / (
                0.01 + np.linalg.norm(HR_pr - np.mean(HR_pr), ord=2) * np.linalg.norm(HR_rel - np.mean(HR_rel), ord=2))
    print('| me: %.4f' % me,
          '| std: %.4f' % std,
          '| mae: %.4f' % mae,
          '| rmse: %.4f' % rmse,
          '| mer: %.4f' % mer,
          '| p: %.4f' % p
          )
    return me, std, mae, rmse, mer, p

if __name__ == '__main__':

    gt_name = 'Label/HR.mat'
    frames_num = 256
    pr_path = r'./Result/rPPGNet_V4VSpatial0.5Temporal0.1HR_pr.mat'
    gt_path = r'./Result/rPPGNet_V4VSpatial0.5Temporal0.1HR_rel.mat'
    Idex_files = r'./STMap/STMap_Index/V4V'

    gt_av = []
    pr_av = []
    gt_ps = []
    pr_ps = []

    pr = scio.loadmat(pr_path)['HR_pr']
    pr = np.array(pr.astype('float32')).reshape(-1)
    gt = scio.loadmat(gt_path)['HR_rel']
    gt = np.array(gt.astype('float32')).reshape(-1)
    files_list = os.listdir(Idex_files)
    files_list = sorted(files_list)

    temp = scio.loadmat(os.path.join(Idex_files, files_list[0]))
    lastPath = str(temp['Path'][0])
    pr_temp = []
    gt_temp = []
    print(pr.size)
    print(len(files_list))
    for HR_index in range(pr.size-1):
        temp = scio.loadmat(os.path.join(Idex_files, files_list[HR_index]))
        nowPath = str(temp['Path'][0])
        Step_Index = int(temp['Step_Index'])
        a = pr[HR_index]

        if lastPath != nowPath:
            if pr_temp is None:
                print(nowPath)
                print(lastPath)
                pr_temp = []
                gt_temp = []
            else:
                pr_av.append(np.nanmean(pr_temp))
                gt_av.append(np.nanmean(gt_temp))
                gt_ps.append(gt_temp)
                pr_ps.append(pr_temp)

                pr_temp = []
                gt_temp = []
        else:

            pr_temp.append(pr[HR_index])
            gt_temp.append(gt[HR_index])
        lastPath = nowPath

    io.savemat('gt_ps.mat', {'HR': gt_ps})
    io.savemat('pr_ps.mat', {'HR': pr_ps})
    io.savemat('HR_rel.mat', {'HR': gt_av})
    io.savemat('HR_pr.mat', {'HR': pr_av})
    MyEval(gt_av, pr_av)
