import scipy
from scipy import ndimage
import cv2
import numpy as np
import sys

import torch
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
import deeplab_resnet
from collections import OrderedDict
import os
from os import walk
import matplotlib.pyplot as plt
import torch.nn as nn

max_label = 21- 1  # labels from 0,1, ... 20(for VOC)


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def get_iou(pred, gt):
    if pred.shape != gt.shape:
        print 'pred shape', pred.shape, 'gt shape', gt.shape
    assert (pred.shape == gt.shape)
    gt = gt.astype(np.float32)
    pred = pred.astype(np.float32)

    count = np.zeros((max_label + 1,))
    for j in range(max_label + 1):
        x = np.where(pred == j)
        p_idx_j = set(zip(x[0].tolist(), x[1].tolist()))
        x = np.where(gt == j)
        GT_idx_j = set(zip(x[0].tolist(), x[1].tolist()))
        n_jj = set.intersection(p_idx_j, GT_idx_j)
        u_jj = set.union(p_idx_j, GT_idx_j)

        if len(GT_idx_j) != 0:
            count[j] = float(len(n_jj)) / float(len(u_jj))

    result_class = count
    Aiou = np.sum(result_class[:]) / float(len(np.unique(gt)))

    return Aiou

def miou_eval(model,img_list_root):


    im_path = '/home/tk/WSSS/dataset/VOC/JPEGImages/'
    model.eval()
    gt_path = '/home/tk/WSSS/dataset/VOC/gt_bdy/'
    img_list = open(img_list_root).readlines()

    hist = np.zeros((max_label + 1, max_label + 1))
    pytorch_list = [];
    step = 0
    for i in img_list:
        # img = np.zeros((513, 513, 3));
        if step > 100:
            continue
        img_temp = cv2.imread(os.path.join(im_path, i[:-1] + '.jpg')).astype(float)
        img_original = img_temp
        img_temp[:, :, 0] = img_temp[:, :, 0] - 104.008
        img_temp[:, :, 1] = img_temp[:, :, 1] - 116.669
        img_temp[:, :, 2] = img_temp[:, :, 2] - 122.675
        # img[:img_temp.shape[0], :img_temp.shape[1], :] = img_temp
        img = cv2.resize(img_temp,(321,321))
        gt = cv2.imread(os.path.join(gt_path, i[:-1]), 0)
        # gt[gt==255] = 0

        output = model(
            Variable(torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0), volatile=True).cuda())[3]
        interp = nn.UpsamplingBilinear2d(size=(gt.shape[0], gt.shape[1]))
        output = interp(output)
        output = output.cpu().data[0].numpy()
        output = output.transpose(1, 2, 0)
        output = np.argmax(output, axis=2)


        iou_pytorch = get_iou(output, gt)
        pytorch_list.append(iou_pytorch)
        hist += fast_hist(gt.flatten(), output.flatten(), max_label + 1)
        if (step+1)%100 == 0:
            iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
            print step+1, np.sum(iou) / len(iou)
        step +=1
    iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print 'pytorch', iter, "Mean iou = ", np.sum(iou) / len(iou)
    return np.sum(iou) / len(iou)

