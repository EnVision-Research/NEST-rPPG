# -*- coding: UTF-8 -*-
import numpy as np
import scipy.io as io
import torch
import MyDataset
import MyLoss
import model
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms
import utils
from datetime import datetime
import os
import time
from utils import Logger, time_to_str
from timeit import default_timer as timer
import time
import random

TARGET_DOMAIN = {'VIPL': ['V4V',  'PURE', 'BUAA', 'UBFC'], \
                 'V4V': ['VIPL',  'PURE', 'BUAA', 'UBFC'], \
                 'PURE': ['VIPL', 'V4V', 'BUAA', 'UBFC'], \
                 'BUAA': ['VIPL', 'V4V', 'PURE', 'UBFC'], \
                 'UBFC': ['VIPL', 'V4V', 'PURE', 'BUAA']}

FILEA_NAME = {'VIPL': ['VIPL', 'VIPL', 'STMap_RGB_Align_CSI'], \
              'V4V': ['V4V', 'V4V', 'STMap_RGB'], \
              'PURE': ['PURE', 'PURE', 'STMap'], \
              'BUAA': ['BUAA', 'BUAA', 'STMap_RGB'], \
              'UBFC': ['UBFC', 'UBFC', 'STMap']}

if __name__ == '__main__':


    args = utils.get_args()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    Source_domain_Names = TARGET_DOMAIN[args.tgt]
    root_file = r'./STMap/'
    # 参数
    File_Name_0 = FILEA_NAME[Source_domain_Names[0]]
    source_name_0 = Source_domain_Names[0]
    source_fileRoot_0 = root_file + File_Name_0[0]
    source_saveRoot_0 = root_file + 'STMap_Index/' + File_Name_0[1]
    source_map_0 = File_Name_0[2] + '.png'

    File_Name_1 = FILEA_NAME[Source_domain_Names[1]]
    source_name_1 = Source_domain_Names[1]
    source_fileRoot_1 = root_file + File_Name_1[0]
    source_saveRoot_1 = root_file + 'STMap_Index/' + File_Name_1[1]
    source_map_1 = File_Name_1[2] + '.png'

    File_Name_2 = FILEA_NAME[Source_domain_Names[2]]
    source_name_2 = Source_domain_Names[2]
    source_fileRoot_2 = root_file + File_Name_2[0]
    source_saveRoot_2 = root_file + 'STMap_Index/' +  File_Name_2[1]
    source_map_2 = File_Name_2[2] + '.png'

    File_Name_3 = FILEA_NAME[Source_domain_Names[3]]
    source_name_3 = Source_domain_Names[3]
    source_fileRoot_3 = root_file + File_Name_3[0]
    source_saveRoot_3 = root_file + 'STMap_Index/' + File_Name_3[1]
    source_map_3 = File_Name_3[2] + '.png'


    FILE_Name = FILEA_NAME[args.tgt]
    Target_name = args.tgt
    Target_fileRoot = root_file + FILE_Name[0]
    Target_saveRoot = root_file + 'STMap_Index/' + FILE_Name[1]
    Target_map = FILE_Name[2] + '.png'
    
    # 训练参数
    batch_size_num = args.batchsize
    epoch_num = args.epochs
    learning_rate = args.lr

    test_batch_size = args.batchsize
    num_workers = args.num_workers
    GPU = args.GPU

    # 图片参数
    input_form = args.form
    reTrain = args.reTrain
    frames_num = args.frames_num
    fold_num = args.fold_num
    fold_index = args.fold_index

    best_mae = 99

    print('batch num:', batch_size_num, ' epoch_num:', epoch_num, ' GPU Inedex:', GPU)
    print(' frames num:', frames_num, ' learning rate:', learning_rate, )
    print('fold num:', frames_num, ' fold index:', fold_index)

    if not os.path.exists('./Result_log'):
        os.makedirs('./Result_log')
    rPPGNet_name = 'rPPGNet_' + Target_name + 'Spatial' + str(args.spatial_aug_rate) + 'Temporal' + str(args.temporal_aug_rate)
    log = Logger()
    log.open('./Result_log/' + rPPGNet_name + '_log.txt', mode='a')
    log.write("\n----------------------------------------------- [START %s] %s\n\n" % (
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 51))

    # 运行媒介
    if torch.cuda.is_available():
        device = torch.device('cuda:' + GPU if torch.cuda.is_available() else 'cpu')  #
        print('on GPU')
    else:
        print('on CPU')

    # 数据集
    if args.reData == 1:
        source_index_0 = os.listdir(source_fileRoot_0)
        source_index_1 = os.listdir(source_fileRoot_1)
        source_index_2 = os.listdir(source_fileRoot_2)
        source_index_3 = os.listdir(source_fileRoot_3)
        Target_index = os.listdir(Target_fileRoot)

        source_Indexa_0 = MyDataset.getIndex(source_fileRoot_0, source_index_0, \
                                             source_saveRoot_0, source_map_0, 10, frames_num)
        source_Indexa_1 = MyDataset.getIndex(source_fileRoot_1, source_index_1, \
                                             source_saveRoot_1, source_map_1, 10, frames_num)
        source_Indexa_2 = MyDataset.getIndex(source_fileRoot_2, source_index_2, \
                                             source_saveRoot_2, source_map_2, 10, frames_num)
        source_Indexa_3 = MyDataset.getIndex(source_fileRoot_3, source_index_3, \
                                             source_saveRoot_3, source_map_3, 10, frames_num)
        Target_Indexa = MyDataset.getIndex(Target_fileRoot, Target_index, \
                                           Target_saveRoot, Target_map, 10, frames_num)

    source_db_0 = MyDataset.Data_DG(root_dir=source_saveRoot_0, dataName=source_name_0, \
                                    STMap=source_map_0, frames_num=frames_num, args=args)
    source_db_1 = MyDataset.Data_DG(root_dir=source_saveRoot_1, dataName=source_name_1, \
                                    STMap=source_map_1, frames_num=frames_num, args=args)
    source_db_2 = MyDataset.Data_DG(root_dir=source_saveRoot_2, dataName=source_name_2, \
                                    STMap=source_map_2, frames_num=frames_num, args=args)
    source_db_3 = MyDataset.Data_DG(root_dir=source_saveRoot_3, dataName=source_name_3, \
                                    STMap=source_map_3, frames_num=frames_num, args=args)
    Target_db = MyDataset.Data_DG(root_dir=Target_saveRoot, dataName=Target_name, \
                                  STMap=Target_map, frames_num=frames_num, args=args)

    src_loader_0 = DataLoader(source_db_0, batch_size=batch_size_num, shuffle=True, num_workers=num_workers)
    src_loader_1 = DataLoader(source_db_1, batch_size=batch_size_num, shuffle=True, num_workers=num_workers)
    src_loader_2 = DataLoader(source_db_2, batch_size=batch_size_num, shuffle=True, num_workers=num_workers)
    src_loader_3 = DataLoader(source_db_3, batch_size=batch_size_num, shuffle=True, num_workers=num_workers)
    tgt_loader = DataLoader(Target_db, batch_size=batch_size_num, shuffle=False, num_workers=num_workers)

    BaseNet = model.BaseNet()

    if reTrain == 1:
        BaseNet = torch.load('./Result_Model/' + rPPGNet_name, map_location=device)
        print('load ' + rPPGNet_name + ' right')
    BaseNet.to(device=device)
    optimizer_rPPG = torch.optim.Adam(BaseNet.parameters(), lr=learning_rate)
    loss_func_NP = MyLoss.P_loss3().to(device)
    loss_func_L1 = nn.L1Loss().to(device)
    loss_func_SP = MyLoss.SP_loss(device, clip_length=frames_num).to(device)
    loss_func_NEST_CM = MyLoss.NEST_CM().to(device)
    loss_func_NEST_DM = MyLoss.NEST_DM().to(device)
    loss_func_NEST_TA = MyLoss.NEST_TA(device, Num_ref=8).to(device)
    src_iter_0 = src_loader_0.__iter__()
    src_iter_per_epoch_0 = len(src_iter_0)

    src_iter_1 = src_loader_1.__iter__()
    src_iter_per_epoch_1 = len(src_iter_1)

    src_iter_2 = src_loader_2.__iter__()
    src_iter_per_epoch_2 = len(src_iter_2)

    src_iter_3 = src_loader_3.__iter__()
    src_iter_per_epoch_3 = len(src_iter_3)


    tgt_iter = iter(tgt_loader)
    tgt_iter_per_epoch = len(tgt_iter)

    max_iter = args.max_iter
    start = timer()
    # Training
    BaseNet.train()
    for iter_num in range(max_iter + 1):
        if (iter_num % src_iter_per_epoch_0 == 0):
            src_iter_0 = src_loader_0.__iter__()
        if (iter_num % src_iter_per_epoch_1 == 0):
            src_iter_1 = src_loader_1.__iter__()
        if (iter_num % src_iter_per_epoch_2 == 0):
            src_iter_2 = src_loader_2.__iter__()
        if (iter_num % src_iter_per_epoch_3 == 0):
            src_iter_3 = src_loader_3.__iter__()

        ######### data prepare #########
        data0, bvp0, HR_rel0, data_aug0, bvp_aug0, HR_rel_aug0, = src_iter_0.__next__()
        data1, bvp1, HR_rel1, data_aug1, bvp_aug1, HR_rel_aug1 = src_iter_1.__next__()
        data2, bvp2, HR_rel2, data_aug2, bvp_aug2, HR_rel_aug2 = src_iter_2.__next__()
        data3, bvp3, HR_rel3, data_aug3, bvp_aug3, HR_rel_aug3 = src_iter_3.__next__()

        data0 = Variable(data0).float().to(device=device)
        bvp0 = Variable(bvp0).float().to(device=device).unsqueeze(dim=1)
        HR_rel0 = Variable(torch.Tensor(HR_rel0)).float().to(device=device)
        data_aug0 = Variable(data_aug0).float().to(device=device)
        bvp_aug0 = Variable(bvp_aug0).float().to(device=device).unsqueeze(dim=1)
        HR_rel_aug0 = Variable(torch.Tensor(HR_rel_aug0)).float().to(device=device)


        data1 = Variable(data1).float().to(device=device)
        bvp1 = Variable((bvp1)).float().to(device=device).unsqueeze(dim=1)
        HR_rel1 = Variable(torch.Tensor(HR_rel1)).float().to(device=device)
        data_aug1 = Variable(data_aug1).float().to(device=device)
        bvp_aug1 = Variable((bvp_aug1)).float().to(device=device).unsqueeze(dim=1)
        HR_rel_aug1 = Variable(torch.Tensor(HR_rel_aug1)).float().to(device=device)


        data2 = Variable(data2).float().to(device=device)
        bvp2 = Variable((bvp2)).float().to(device=device).unsqueeze(dim=1)
        HR_rel2 = Variable(torch.Tensor(HR_rel2)).float().to(device=device)
        data_aug2 = Variable(data_aug2).float().to(device=device)
        bvp_aug2 = Variable((bvp_aug2)).float().to(device=device).unsqueeze(dim=1)
        HR_rel_aug2 = Variable(torch.Tensor(HR_rel_aug2)).float().to(device=device)

        data3 = Variable(data3).float().to(device=device)
        bvp3 = Variable((bvp3)).float().to(device=device).unsqueeze(dim=1)
        HR_rel3 = Variable(torch.Tensor(HR_rel3)).float().to(device=device)
        data_aug3 = Variable(data_aug3).float().to(device=device)
        bvp_aug3 = Variable((bvp_aug3)).float().to(device=device).unsqueeze(dim=1)
        HR_rel_aug3 = Variable(torch.Tensor(HR_rel_aug3)).float().to(device=device)

        optimizer_rPPG.zero_grad()
        d_b0, d_b1, d_b2, d_b3 = data0.shape[0], data1.shape[0], data2.shape[0], data3.shape[0]
        input = torch.cat([data0, data1, data2, data3], dim=0)
        input_aug = torch.cat([data_aug0, data_aug1, data_aug2, data_aug3], dim=0)
        bvp_pre, HR_pr, av = BaseNet(input)
        bvp_pre_aug, HR_pr_aug, av_aug = BaseNet(input_aug)

      

        bvp_pre0, bvp_pre1, bvp_pre2, bvp_pre3 = bvp_pre[0:d_b0], bvp_pre[d_b0:d_b0+d_b1], bvp_pre[d_b0+d_b1:d_b0+d_b1+d_b2], bvp_pre[d_b0+d_b1+d_b2:]
        HR_pr0, HR_pr1, HR_pr2, HR_pr3 = HR_pr[0:d_b0], HR_pr[d_b0:d_b0+d_b1], HR_pr[d_b0+d_b1:d_b0+d_b1+d_b2], HR_pr[d_b0+d_b1+d_b2:]
        bvp_pre_aug0, bvp_pre_aug1, bvp_pre_aug2, bvp_pre_aug3 = bvp_pre_aug[0:d_b0], bvp_pre_aug[d_b0:d_b0 + d_b1], bvp_pre_aug[d_b0 + d_b1:d_b0 + d_b1 + d_b2], bvp_pre_aug[d_b0 + d_b1 + d_b2:]
        HR_pr_aug0, HR_pr_aug1, HR_pr_aug2, HR_pr_aug3 = HR_pr_aug[0:d_b0], HR_pr_aug[d_b0:d_b0 + d_b1], HR_pr_aug[d_b0 + d_b1:d_b0 + d_b1 + d_b2], HR_pr_aug[d_b0 + d_b1 + d_b2:]

        src_loss_0 = MyLoss.get_loss(bvp_pre0, HR_pr0, bvp0, HR_rel0, source_name_0, \
                                     loss_func_NP, loss_func_L1, args, iter_num)
        src_loss_1 = MyLoss.get_loss(bvp_pre1, HR_pr1, bvp1, HR_rel1, source_name_1, \
                                     loss_func_NP, loss_func_L1, args, iter_num)
        src_loss_2 = MyLoss.get_loss(bvp_pre2, HR_pr2, bvp2, HR_rel2, source_name_2, \
                                     loss_func_NP, loss_func_L1, args, iter_num)
        src_loss_3 = MyLoss.get_loss(bvp_pre3, HR_pr3, bvp3, HR_rel3, source_name_3, \
                                     loss_func_NP, loss_func_L1, args, iter_num)

        src_loss_aug_0 = MyLoss.get_loss(bvp_pre_aug0, HR_pr_aug0, bvp_aug0, HR_rel_aug0, source_name_0, \
                                     loss_func_NP, loss_func_L1, args, iter_num)
        src_loss_aug_1 = MyLoss.get_loss(bvp_pre_aug1, HR_pr_aug1, bvp_aug1, HR_rel_aug1, source_name_1, \
                                     loss_func_NP, loss_func_L1, args, iter_num)
        src_loss_aug_2 = MyLoss.get_loss(bvp_pre_aug2, HR_pr_aug2, bvp_aug2, HR_rel_aug2, source_name_2, \
                                     loss_func_NP, loss_func_L1, args, iter_num)
        src_loss_aug_3 = MyLoss.get_loss(bvp_pre_aug3, HR_pr_aug3, bvp_aug3, HR_rel_aug3, source_name_3, \
                                     loss_func_NP, loss_func_L1, args, iter_num)

        HR_rels = torch.cat((HR_rel0, HR_rel1, HR_rel2, HR_rel3), dim=0)
        HR_rel_augs = torch.cat((HR_rel_aug0, HR_rel_aug1, HR_rel_aug2, HR_rel_aug3), dim=0)
        loss_CM = -loss_func_NEST_CM(torch.cat((av, av_aug), dim=0))
        loss_DM = loss_func_NEST_DM(av, av_aug)
        loss_TA = loss_func_NEST_TA(torch.cat((av, av_aug), dim=0), torch.cat((HR_rels, HR_rel_augs), dim=0))

        k = 2.0 / (1.0 + np.exp(-10.0 * iter_num / args.max_iter)) - 1.0

        loss = (src_loss_0 + src_loss_1 + src_loss_2 + src_loss_3) \
               + (src_loss_aug_0 + src_loss_aug_1 + src_loss_aug_2 + src_loss_aug_3) \
               + 0.1 * k * loss_TA + 0.001 * k * loss_CM + 0.01 * k * loss_DM
        if torch.sum(torch.isnan(loss)) > 0:
            print('Nan')
            break
        else:
            loss.backward()
            optimizer_rPPG.step()
        if iter_num % 100 == 0:
            log.write(
                'Train Inter:' + str(iter_num)\
                + ' | loss:  ' + str(loss.data.cpu().numpy()) \
                + ' |' + source_name_0 + ' : ' + str(src_loss_0.data.cpu().numpy()) \
                + ' |' + source_name_1 + ' : ' + str(src_loss_1.data.cpu().numpy()) \
                + ' |' + source_name_2 + ' : ' + str(src_loss_2.data.cpu().numpy()) \
                + ' |' + source_name_3 + ' : ' + str(src_loss_3.data.cpu().numpy()) \
                + ' |' + 'CM' + ' : ' + str(loss_CM.data.cpu().numpy()) \
                + ' |' + 'DM' + ' : ' + str(loss_DM.data.cpu().numpy()) \
                + ' |' + 'TA' + ' : ' + str(loss_TA.data.cpu().numpy()) \
                + ' |' + time_to_str(timer() - start, 'min'))
            log.write('\n')

    # Testing
    BaseNet.eval()
    HR_pr_temp = []
    HR_rel_temp = []
    BVP_ALL = []
    BVP_PR_ALL = []
    for step, (data, bvp, HR_rel, _, _, _) in enumerate(tgt_loader):
        data = Variable(data).float().to(device=device)
        bvp = Variable(bvp).float().to(device=device)
        HR_rel = Variable(HR_rel).float().to(device=device)
        bvp = bvp.unsqueeze(dim=1)
        Wave = bvp
        Wave_pr, HR_pr, av = BaseNet(data)

        HR_rel_temp.extend(HR_rel.data.cpu().numpy())
        HR_pr_temp.extend(HR_pr.data.cpu().numpy())
        BVP_ALL.extend(Wave.data.cpu().numpy())
        BVP_PR_ALL.extend(Wave_pr.data.cpu().numpy())




    if not os.path.exists('./Result'):
        os.makedirs('./Result')
    io.savemat('./Result/' + rPPGNet_name + 'HR_pr.mat', {'HR_pr': HR_pr_temp})
    io.savemat('./Result/' + rPPGNet_name + 'HR_rel.mat', {'HR_rel': HR_rel_temp})
    io.savemat('./Result/' + rPPGNet_name + 'WAVE_ALL.mat',
               {'Wave': BVP_ALL})
    io.savemat('./Result/' + rPPGNet_name + 'WAVE_PR_ALL.mat',
               {'Wave': BVP_PR_ALL})
    if not os.path.exists('./Result_Model'):
        os.makedirs('./Result_Model')
    torch.save(BaseNet, './Result_Model/' + rPPGNet_name)
    print('saveModel As ' + rPPGNet_name)
