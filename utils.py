import os
import torch
import torch.nn as nn
import numpy as np
import cv2
import random
import math
import scipy
from torch.autograd import Variable
import torch.nn.functional as F
from model_MSC import InitDeepLabLargeFOV
import skimage.measure


def get_parameters(model, bias=False,final=False):
    if(final):
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                if (m.out_channels == 21):
                    if bias:
                        yield m.bias
                    else:
                        yield m.weight
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                if (not m.out_channels == 21): 
                    if bias:
                        yield m.bias
                    else:
                        yield m.weight


def adjust_learning_rate(optimizer, lr):
    optimizer.param_groups[0]['lr'] = lr
    optimizer.param_groups[1]['lr']=2*lr
    optimizer.param_groups[2]['lr']=10*lr
    optimizer.param_groups[3]['lr']=20*lr
    return optimizer


def chunker(seq, size):
    return (seq[pos:pos+size] for pos in xrange(0,len(seq), size))


def read_file(path_to_file):
    with open(path_to_file) as f:
        img_list = []
        for line in f:
            img_list.append(line[:-1])
    return img_list


def resize_label_batch(label, size):
    label_resized = Variable(torch.zeros((label.shape[3],1,size,size)))
    interp = nn.Upsample(size=(size, size), mode='bilinear')
    labelVar = Variable(torch.from_numpy(label.transpose(3, 2, 0, 1)))
    label_resized[:, :, :, :] = interp(labelVar)
    return label_resized


def flip(I,flip_p):
    if flip_p>0.5:
        return I[:,::-1,:]
    else:
        return I


def blur(img_temp,blur_p):
    if blur_p>0.5:
        return cv2.GaussianBlur(img_temp,(3,3),1)
    else:
        return img_temp


def crop(img_temp,dim,new_p=True,h_p=0,w_p=0):
    h =img_temp.shape[0]
    w = img_temp.shape[1]
    trig_h=trig_w=False
    if(h>dim):
        if(new_p):
            h_p = int(random.uniform(0,1)*(h-dim))
        img_temp = img_temp[h_p:h_p+dim]
    elif(h<dim):
        trig_h = True
    if(w>dim):
        if(new_p):
            w_p = int(random.uniform(0,1)*(w-dim))
        img_temp = img_temp[:,w_p:w_p+dim]
    elif(w<dim):
        trig_w = True
    if(trig_h or trig_w):
        pad = np.zeros((dim,dim,3))
        pad[:img_temp.shape[0],:img_temp.shape[1],:] = img_temp
        return (pad,h_p,w_p)
    else:
        return (img_temp,h_p,w_p)

def crop_cue(img_temp,dim,new_p=True,h_p=0,w_p=0):
    cue_out = np.zeros((3,len(img_temp[0])))
    h =img_temp.shape[0]
    w = img_temp.shape[1]
    trig_h=trig_w=False
    # tmp_h = np.zeros((321,w))
    if(h>dim):
        if(new_p):
            h_p = int(random.uniform(0,1)*(h-dim))
        tmp_h = img_temp[h_p:h_p+dim]
        img_temp = tmp_h
    elif(h<dim):
        trig_h = True
    tmp_w = np.zeros((321,321))
    if(w>dim):
        if(new_p):
            w_p = int(random.uniform(0,1)*(w-dim))
        tmp_w = img_temp[:,w_p:w_p+dim]
        img_temp = tmp_w
    elif(w<dim):
        trig_w = True
    if(trig_h or trig_w):
        pad = np.zeros((dim,dim))
        pad[:img_temp.shape[0],:img_temp.shape[1]] = img_temp
        return (pad,h_p,w_p)
    else:
        return (img_temp,h_p,w_p)



def rotate(img_temp,rot,rot_p):
    if(rot_p>0.5):
        rows,cols,ind = img_temp.shape
        h_pad = int(rows*abs(math.cos(rot/180.0*math.pi)) + cols*abs(math.sin(rot/180.0*math.pi)))
        w_pad = int(cols*abs(math.cos(rot/180.0*math.pi)) + rows*abs(math.sin(rot/180.0*math.pi)))
        final_img = np.zeros((h_pad,w_pad,3))
        final_img[(h_pad-rows)/2:(h_pad+rows)/2,(w_pad-cols)/2:(w_pad+cols)/2,:] = np.copy(img_temp)
        M = cv2.getRotationMatrix2D((w_pad/2,h_pad/2),rot,1)
        final_img = cv2.warpAffine(final_img,M,(w_pad,h_pad),flags = cv2.INTER_NEAREST)
        part_denom = (math.cos(2*rot/180.0*math.pi))
        w_inside = int((cols*abs(math.cos(rot/180.0*math.pi)) - rows*abs(math.sin(rot/180.0*math.pi)))/part_denom)
        h_inside = int((rows*abs(math.cos(rot/180.0*math.pi)) - cols*abs(math.sin(rot/180.0*math.pi)))/part_denom)
        final_img = final_img[(h_pad-h_inside)/2:(h_pad+h_inside)/2,(w_pad- w_inside)/2:(w_pad+ w_inside)/2,:]
        return final_img
    else:
        return img_temp


def get_data_from_chunk_v2_image_level_label(chunk, gt_path, img_path, id=False, aug=True):
    dim = 321
    images = np.zeros((dim, dim, 3, len(chunk)))
    gt = np.zeros((20, len(chunk)))
    img_ids = []
    flip_idx = []
    hhp = []
    wwp = []
    for i, piece in enumerate(chunk):
        img_name = piece.split(' ')[0]
        img_id = piece.split(' ')[1]
        img_ids += [img_id]
        # img_ids += [img_name[:-4]]
        gt_name = piece.split(' ')[0]
        gt_name = gt_name[:-4] + '.png'
        img_temp = cv2.imread(os.path.join(img_path, img_name))
        gt_temp = cv2.imread(os.path.join(gt_path, gt_name))[:, :, 0]
        gt_temp[gt_temp == 255] = 0
        label_temp = np.zeros(20)
        for l in range(20):
            if np.any(gt_temp == (l + 1)):
                label_temp[l] = 1

        flip_p = random.uniform(0, 1)
        flip_idx += [flip_p]
        # rot_p = random([-10,-7,-5,3,0,3,5,7,10],1)[0]
        # scale_p = random.uniform(0, 1)
        # blur_p = random.uniform(.uniform(0, 1)
        # rot = np.random.choice0, 1)
        # if(scale_p>0.75):
        # scale = random.uniform(0.75, 1.5)
        # else:
        # scale = 1
        # if(img_temp.shape[0]<img_temp.shape[1]):
        # ratio = dim*scale/float(img_temp.shape[0])
        # else:
        # ratio = dim*scale/float(img_temp.shape[1])
        # img_temp = cv2.resize(img_temp,(int(img_temp.shape[1]*ratio),int(img_temp.shape[0]*ratio))).astype(float)

        if aug:
            img_temp = flip(img_temp, flip_p)

        # img_temp = rotate(img_temp,rot,rot_p)
        # img_temp = blur(img_temp,blur_p)
        img_temp = img_temp.astype('float')

        img_temp[:, :, 0] = img_temp[:, :, 0] - 104.008
        img_temp[:, :, 1] = img_temp[:, :, 1] - 116.669
        img_temp[:, :, 2] = img_temp[:, :, 2] - 122.675
        # if aug:
        #     img_temp, img_temp_h_p, img_temp_w_p = crop(img_temp, dim)
            # hhp += [h_p]
            # wwp += [w_p]
        # else:
        #     img_temp = cv2.resize(img_temp, (321, 321))
        img_temp = cv2.resize(img_temp, (321, 321))
        images[:, :, :, i] = img_temp
        gt[:, i] = label_temp

    labels = gt.transpose((1, 0))
    labels = Variable(torch.from_numpy(labels).long()).cuda()
    images = images.transpose((3, 2, 0, 1))
    images = Variable(torch.from_numpy(images).float()).cuda()
    if id:
        return images, labels, img_ids, flip_idx
        # return images, labels, gt_name[:-4]
    else:
        return images, labels

def get_data_from_chunk_v2_cue_aug(chunk, gt_path, img_path, loc_cue):
    dim = 321
    images = np.zeros((dim, dim, 3, len(chunk)))
    gt = np.zeros((20, len(chunk)))

    cues_final = np.ndarray((41, 41, 21, len(chunk)))
    img_ids = []
    flip_idx = []
    hhp = []
    wwp = []
    for i, piece in enumerate(chunk):
        cues = np.ndarray((321, 321, 21))

        img_name = piece.split(' ')[0]
        img_id = piece.split(' ')[1]
        # img_ids += [img_id]
        img_ids += [img_name[:-4]]
        gt_name = piece.split(' ')[0]
        gt_name = gt_name[:-4] + '.png'
        img_temp = cv2.imread(os.path.join(img_path, img_name))
        # print img_temp.shape
        gt_temp = cv2.imread(os.path.join(gt_path, gt_name))[:, :, 0]
        gt_temp[gt_temp == 255] = 0
        label_temp = np.zeros(20)
        for l in range(20):
            if np.any(gt_temp == (l + 1)):
                label_temp[l] = 1

######### flip ##################
        flip_1 = random.uniform(0, 1)
        flip_2 = random.uniform(0, 1)
        cues_i = loc_cue[piece.split(' ')[0][:-4] + '_cues']
        cue_temp = np.zeros((gt_temp.shape[0], gt_temp.shape[1],21))
        # print cue_temp.shape
        for k in range(len(cues_i[0])):
            if cues_i[1, k] >= 0 and cues_i[2, k] >= 0:
                cue_temp[cues_i[1, k], cues_i[2, k], cues_i[0, k]] = 1.0
        # print cue_temp.sum(0).sum(0).sum(0)
        # print 'scribble_num', cue_temp.sum(2).sum(1).sum(0)
        # print cue_temp.sum(0).sum(0)
        if flip_1 > 0.5:
            img_temp = img_temp[:, ::-1, :]
            cue_temp = cue_temp[:, ::-1,:]
        # print 'after flip', cue_temp.sum(0).sum(0)
        # if flip_2 > 0.5:
        #     img_temp = img_temp[::-1, :, :]
        #     cue_temp = cue_temp[::-1, :, :]
        # print 'after flip', cue_temp.sum(0).sum(0)

######### crop ###########

        # img_temp = cv2.resize(img_temp, (321, 321))
        # cue_temp = cv2.resize(cue_temp, (321, 321))
        # skimage.measure.block_reduce(cue_temp, (2, 2), np.max)
        # skimage.measure.block_reduce(cue_temp, (2, 2), np.max)

        # img_temp, img_temp_h_p, img_temp_w_p = crop(img_temp, dim)
        # for cls in range(21):
        #     cue_cls_temp, h_p, w_p = crop_cue(cue_temp[:,:,cls], dim, new_p=False, h_p=img_temp_h_p, w_p=img_temp_w_p)
        #     cues[:, :, cls] = cue_cls_temp


        img_temp = cv2.resize(img_temp, (321, 321))
        cue_temp = cv2.resize(cue_temp, (321, 321), cv2.INTER_NEAREST)
        cues = skimage.measure.block_reduce(cues, (8, 8, 1), np.max)
        # print np.max(cues), np.min(cues)
        # print np.max(cv2.resize(cue_temp,(41,41),cv2.INTER_NEAREST))
        cues_final[:, :, :, i] = cues


        img_temp = img_temp.astype('float')
        img_temp[:, :, 0] = img_temp[:, :, 0] - 104.008
        img_temp[:, :, 1] = img_temp[:, :, 1] - 116.669
        img_temp[:, :, 2] = img_temp[:, :, 2] - 122.675
        images[:, :, :, i] = img_temp
        gt[:, i] = label_temp
        # cues = skimage.measure.block_reduce(cues, (8, 8, 1), np.max)
        # cues_final[:,:,:,i] = cues

    # cues_final = np.ndarray((len(chunk),21,41,41))
    # cues_final = cv2.resize(cues, (41, 41))
    # print cues_final.sum(0).sum(0)
    # ref : cv2.resize(img_temp, (int(img_temp.shape[1] * ratio), int(img_temp.shape[0] * ratio))).astype(float)

    labels = gt.transpose((1, 0))
    labels = Variable(torch.from_numpy(labels).long()).cuda()
    images = images.transpose((3, 2, 0, 1))
    images = Variable(torch.from_numpy(images).float()).cuda()
    cues_final = cues_final.transpose((3,2,0,1))
    cues_final = Variable(torch.from_numpy(cues_final).float().cuda())

    # print cues_final.sum(3).sum(2).sum(1)
    return images, cues_final, labels, img_ids



def get_test_data_from_chunk_v2(chunk,im_path):
    dim = 513
    images = np.zeros((dim,dim,3,len(chunk)))
    for i,piece in enumerate(chunk):
        img_temp = cv2.imread(im_path+piece+'.jpg')
        img_temp = img_temp.astype('float')
        img_temp[:,:,0] = img_temp[:,:,0] - 104.008
        img_temp[:,:,1] = img_temp[:,:,1] - 116.669
        img_temp[:,:,2] = img_temp[:,:,2] - 122.675
        img_temp,img_temp_h_p,img_temp_w_p = crop(img_temp,dim)
        images[:,:,:,i] = img_temp
        
    images = images.transpose((3,2,0,1))
    images = Variable(torch.from_numpy(images).float(),volatile= True).cuda()
    return images


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)


def get_iou(pred, gt):
    if pred.shape != gt.shape:
        print 'pred shape', pred.shape, 'gt shape', gt.shape
    assert (pred.shape == gt.shape)
    gt = gt.astype(np.float32)
    pred = pred.astype(np.float32)

    count = np.zeros((21,))
    for j in range(21):
        x = np.where(pred == j)
        p_idx_j = set(zip(x[0].tolist(), x[1].tolist()))
        x = np.where(gt == j)
        GT_idx_j = set(zip(x[0].tolist(), x[1].tolist()))
        # pdb.set_trace()
        n_jj = set.intersection(p_idx_j, GT_idx_j)
        u_jj = set.union(p_idx_j, GT_idx_j)

        if len(GT_idx_j) != 0:
            count[j] = float(len(n_jj)) / float(len(u_jj))

    result_class = count
    Aiou = np.sum(result_class[:]) / float(len(np.unique(gt)))

    return Aiou


def iou_test(weight_name):
    list_path = '../VOC/ImageSets'
    gt_path = '../VOC/cls_png'
    im_path = '../VOC/JPEGImages'

    model = InitDeepLabLargeFOV()
    model.load_state_dict(torch.load(weight_name))
    model.cuda()
    model.eval()

    img_list = open(os.path.join(list_path, 'val.txt')).readlines()
    np.random.shuffle(img_list)

    for iter in range(1, 4):
        hist = np.zeros((21, 21))
        pytorch_list = []
        for i in img_list:
            img = np.zeros((513, 513, 3))

            img_temp = cv2.imread(os.path.join(im_path, i[:-1] + '.jpg')).astype(float)
            img_original = img_temp
            img_temp[:, :, 0] = img_temp[:, :, 0] - 104.008
            img_temp[:, :, 1] = img_temp[:, :, 1] - 116.669
            img_temp[:, :, 2] = img_temp[:, :, 2] - 122.675
            img[:img_temp.shape[0], :img_temp.shape[1], :] = img_temp
            gt = cv2.imread(os.path.join(gt_path, i[:-1] + '.png'), 0)
            # gt[gt==255] = 0

            input = Variable(torch.from_numpy(img[np.newaxis, :].transpose(0, 3, 1, 2)).float(), volatile=True).cuda()
            output = model(input)
            interp = nn.Upsample(size=(513, 513), mode='bilinear')
            output = interp(output).cpu().data[0].numpy()
            output = output[:, :img_temp.shape[0], :img_temp.shape[1]]

            output = output.transpose(1, 2, 0)
            output = np.argmax(output, axis=2)

            iou_pytorch = get_iou(output, gt)
            pytorch_list.append(iou_pytorch)
            hist += fast_hist(gt.flatten(), output.flatten(), 21)
        miou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
        print 'pytorch', iter, "Mean iou = ", np.sum(miou) / len(miou)


if __name__=='__main__':
    weight_name = os.path.join('model_save', 'DeepLabLargeFOV_last.pth.tar')
    iou_test(weight_name)
