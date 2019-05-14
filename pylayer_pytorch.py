import torch
import numpy as np
from scipy.ndimage import zoom
from torch.autograd import Variable
import cPickle
from krahenbuhl2013 import CRF

min_prob = 0.0001
# fc8_SEC_Softmax is bounded by min_prob

def AnnotationLayer(pickle_load, img_ids):
    data_file = pickle_load
    labels = torch.zeros(len(img_ids), 1, 1, 21)
    cues = torch.zeros(len(img_ids), 21, 41, 41)

    for i, image_id in enumerate(img_ids):
        labels_i = data_file['%i_labels' % int(image_id)]
        for j in range(len(labels_i)):
            labels[i, 0, 0, labels_i[j]] = 1.0

        cues_i = data_file['%i_cues' % int(image_id)]
        for k in range(len(cues_i[0])):
            cues[i, cues_i[0,k], cues_i[1,k], cues_i[2,k]] = 1.0

    return labels, cues

def SEC_Softmax(preds):

    preds_max,_ = torch.max(preds, dim=1, keepdim=True)
    preds_exp = torch.exp(preds - preds_max)
    probs = preds_exp / torch.sum(preds_exp, dim=1, keepdim=True) + min_prob
    probs = probs / torch.sum(probs, dim=1, keepdim=True)
    return probs

def SeedLossLayer(fc8_SEC_Softmax, cues):
    # fc8_SEC_Softmax: (batch_size=15,21,41,41)
    # cues: (batch_size=15,21,41,41)

    probs = fc8_SEC_Softmax
    labels = cues
    count = labels.sum(3).sum(2).sum(1)
    loss_balanced = - ((labels * torch.log((probs+1e-4)/(1+1e-4))).sum(3).sum(2).sum(1) / (count)).mean(0)

    return loss_balanced

def ExpandLossLayer(fc8_SEC_Softmax, labels):
    probs_tmp = fc8_SEC_Softmax
    stat_inp = labels

    stat = stat_inp[:, :, :, 1:]

    probs_bg = probs_tmp[:, 0, :, :]
    probs = probs_tmp[:, 1:, :, :]
    probs_max,_ = torch.max(torch.max(probs,3)[0], 2)

    q_fg = 0.996
    probs_sort, _ = torch.sort(probs.contiguous().view(-1, 20, 41 * 41), dim=2)
    weights = np.array([q_fg ** i for i in range(41 * 41 - 1, -1, -1)])[None, None, :]
    Z_fg = np.sum(weights)
    weights_var = Variable(torch.from_numpy(weights).cuda()).squeeze().float()
    probs_mean = ((probs_sort * weights_var) / Z_fg).sum(2)

    q_bg = 0.999
    probs_bg_sort,_ = torch.sort(probs_bg.contiguous().view(-1,41*41), dim=1)
    weights_bg = np.array([q_bg ** i for i in range(41 * 41 - 1, -1, -1)])[None, :]
    Z_bg = np.sum(weights_bg)
    weights_bg_var = Variable(torch.from_numpy(weights_bg).cuda()).squeeze().float()
    probs_bg_mean = ((probs_bg_sort * weights_bg_var) / Z_bg).sum(1)
    stat_2d = (stat[:, 0, 0, :] > 0.5).float()
    loss_1 = -torch.mean(torch.sum((stat_2d * torch.log(probs_mean) / torch.sum(stat_2d, dim=1, keepdim=True)), dim=1))
    loss_2 = -torch.mean(torch.sum(((1 - stat_2d) * torch.log(1 - probs_max) / torch.sum(1 - stat_2d, dim=1, keepdim=True)), dim=1))
    loss_3 = -torch.mean(torch.log(probs_bg_mean))

    loss = loss_1 + loss_2 + loss_3
    return loss

def CRFLayer(fc8_SEC_Softmax, images, iternum):
   # fc8-SEC-Softmax: (batch_size=15,21,41,41)
   # images -> resize: (batch_size=15,21,41,41)

   unary = np.transpose(np.array(fc8_SEC_Softmax.cpu().data), [0, 2, 3, 1])
   mean_pixel = np.array([104.0, 117.0, 123.0])
   im = np.array(images.cpu().data)
   im = zoom(im, (1.0, 1.0, 41.0 / im.shape[2], 41.0 / im.shape[3]), order=1)
   im = im + mean_pixel[None, :, None, None]
   im = np.transpose(np.round(im), [0, 2, 3, 1])
   N = unary.shape[0]
   result = np.zeros(unary.shape)

   for i in range(N):
       result[i] = CRF(im[i], unary[i],maxiter=iternum, scale_factor=12.0)

   result = np.transpose(result, [0, 3, 1, 2])
   result[result < min_prob] = min_prob
   result = result / np.sum(result, axis=1, keepdims=True)
   CRF_result = np.log(result)

   return CRF_result

def ConstrainLossLayer(fc8_SEC_Softmax, CRF_result):
    # fc8-SEC-Softmax: (batch_size=15,21,41,41)
    # CRF_result: (batch_size=15,21,41,41)
    probs = fc8_SEC_Softmax
    probs_smooth_log = Variable(torch.from_numpy(CRF_result).cuda())

    probs_smooth = torch.exp(probs_smooth_log).float()
    loss = torch.mean((probs_smooth * torch.log(probs_smooth / probs)).sum(1))
    return loss

