""" Utilities """
import os
import logging
import shutil
import torch
import torchvision.datasets as dset
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
import numpy as np
import argparse
from torch.autograd import Variable
import json
import math
import pandas as pd
import sys

def Drop_HR(whole_max_idx, delNum=4):
    Row_Num, Individual_Num = whole_max_idx.shape
    HR = []
    for individual in range(Individual_Num):
        HR_sorted = np.sort(whole_max_idx[:, individual])
        HR.append(np.mean(HR_sorted[delNum:-delNum]))
    return np.array(HR)

def time_to_str(t, mode='min'):
    if mode=='min':
        t  = int(t)/60
        hr = t//60
        min = t%60
        return '%2d hr %02d min'%(hr,min)
    elif mode=='sec':
        t   = int(t)
        min = t//60
        sec = t%60
        return '%2d min %02d sec'%(min,sec)
    else:
        raise NotImplementedError

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.file = None

    def open(self, file, mode=None):
        if mode is None:
            mode = 'w'
        self.file = open(file, mode)
    def write(self, message, is_terminal=1, is_file=1):
        if '\r' in message:
            is_file = 0
        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()
        if is_file == 1:
            self.file.write(message)
            self.file.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass

def get_args():
    parser = argparse.ArgumentParser(description='Train ',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # 训练参数
    parser.add_argument('-g', '--GPU', dest='GPU', type=str, default='0',
                        help='the index of GPU')
    parser.add_argument('-p', '--pp', dest='num_workers', type=int, default=2,
                        help='num_workers')
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=64*64*64,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=100,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-fn', '--fold_num', type=int, default=5,
                        help='fold_num', dest='fold_num')
    parser.add_argument('-fi', '--fold_index', type=int, default=0,
                        help='fold_index:0-fold_num', dest='fold_index')
    parser.add_argument('-rT', '--reTrain', dest='reTrain', type=int, default=0,
                        help='Load model')
    parser.add_argument('-rD', '--reData', dest='reData', type=int, default=1,
                        help='re Data')
    parser.add_argument('-mi', '--max_iter', dest='max_iter', type=int, default=20000,
                        help='re Data')
    parser.add_argument('-s', '--seed', dest='seed', type=int, default=0,
                        help='seed')

    parser.add_argument('-k1', '--k1', type=float, default=1.0,
                        help='k1', dest='k1')
    parser.add_argument('-k2', '--k2', type=float, default=0.1,
                        help='k2', dest='k2')
    parser.add_argument('-k3', '--k3', type=float, default=1.0,
                        help='k3', dest='k3')
    parser.add_argument('-k4', '--k4', type=float, default=0.1,
                        help='k4', dest='k4')
    parser.add_argument('-k5', '--k5', type=float, default=1.0,
                        help='k5', dest='k5')
    parser.add_argument('-k6', '--k6', type=float, default=0.1,
                        help='k6', dest='k6')
    parser.add_argument('-k7', '--k7', type=float, default=0.1,
                        help='k7', dest='k7')
    parser.add_argument('-k8', '--k8', type=float, default=0.1,
                        help='k8', dest='k8')

    parser.add_argument('-tr', '--temporal_aug_rate', type=float, default=0.1,
                        help='temporal_aug_rate', dest='temporal_aug_rate')
    parser.add_argument('-sr', '--spatial_aug_rate', type=float, default=0.5,
                        help='spatial_aug_rate', dest='spatial_aug_rate')

    # 图片参数
    parser.add_argument('-f', '--form', dest='form', type=str, default='Resize',
                        help='the form of input img')
    parser.add_argument('-wg', '--weight', dest='weight', type=int, default=36,
                        help='the weight of img')
    parser.add_argument('-hg', '--height', dest='height', type=int, default=36,
                        help='the height of img')
    parser.add_argument('-n', '--frames_num', dest='frames_num', type=int, default=256,
                        help='the num of frames')
    parser.add_argument('-t', '--tgt', dest='tgt', type=str, default='VIPL',
                        help='the name of target domain: VIPL, COH, V4V, UBFC...')
    return parser.parse_args()

def MyEval(HR_pr, HR_rel):
    HR_pr = np.array(HR_pr).reshape(-1)
    HR_rel = np.array(HR_rel).reshape(-1)
    temp = HR_pr-HR_rel
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


def param_size(model):
    """ Compute parameter size in MB """
    n_params = sum(
        np.prod(v.size()) for k, v in model.named_parameters() if not k.startswith('aux_head'))
    return n_params / 1024. / 1024.
