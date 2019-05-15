import time
from utils import *
import argparse
import torch.optim as optim
import cPickle
from math import ceil
import math
import scipy
from tensorboardX import SummaryWriter
import cv2
from model_MSC import InitDeepLabLargeFOV
from evalpyt_MSC import miou_eval
from pylayer_pytorch import *
# from pseudo_annot import seed_update
from scipy.ndimage.filters import gaussian_filter

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--init_file', type=str, default='vgg16_20M/vgg16_20M.pth')
parser.add_argument('--max_iter', type=int, default=8000)
parser.add_argument('--wt_decay', type=float, default=0.0005)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--snapshot_dir', type=str, default='model_save')
parser.add_argument('--gt_path', type=str, default='/home/tk/WSSS/dataset/VOC/cls_png')
parser.add_argument('--im_path', type=str, default='/home/tk/WSSS/dataset/VOC/JPEGImages')
parser.add_argument('--list_path', type=str, default='/home/tk/WSSS/dataset/VOC/ImageSets')
parser.add_argument('--save_name', type=str, default='DeepLabLargeFOV_Seed')
args = parser.parse_args()
writer = SummaryWriter()


def train_v1():
    total_start = time.time()

    # hyperparameters, data preparing
    max_iter = int(args.max_iter)
    batch_size = int(args.batch_size)
    wt_decay = float(args.wt_decay)
    momentum = float(args.momentum)
    base_lr = float(args.lr)
    lr = base_lr
    num_class = 20

    gt_path = args.gt_path
    im_path = args.im_path
    img_list = read_file(os.path.join(args.list_path, 'input_list.txt'))

    train_epoch = int(max_iter * batch_size / float(len(img_list)))
    data_list = []
    for i in range(train_epoch + 1):
        np.random.shuffle(img_list)
        data_list.extend(img_list)
    data_list = data_list[:max_iter * batch_size]

    # load model
    model = InitDeepLabLargeFOV()
    model.float()
    model.cuda()
    model.train()

    # optimizer / loss function
    optimizer = optim.SGD([
        {'params': get_parameters(model), 'lr': base_lr, 'weight_decay': wt_decay},
        {'params': get_parameters(model, bias=True), 'lr': 2 * base_lr, 'weight_decay': 0},
        {'params': get_parameters(model, final=True), 'lr': base_lr * 10, 'weight_decay': wt_decay},
        {'params': get_parameters(model, bias=True, final=True), 'lr': base_lr * 20, 'weight_decay': 0}
    ], lr=base_lr, momentum=momentum, weight_decay=wt_decay)
    optimizer.zero_grad()
    dict_src = cPickle.load(open('localization_cues/localization_cues.pickle'))


    # train
    start_time = time.time()

    for iter, chunk in enumerate(chunker(data_list, batch_size)):
        images, DenseGT, labels, img_ids, _ = get_data_from_chunk_v2_FromSnapshot_noScaleCrop(chunk, gt_path, im_path, dict_src)

        model.zero_grad()
        outputs = model(images)
        outputs = outputs[3]

        preds_max, _ = torch.max(outputs, dim=1, keepdim=True)
        preds_max_cpu = Variable(preds_max.data)
        preds_exp = torch.exp(outputs - preds_max_cpu)
        probs = preds_exp / torch.sum(preds_exp, dim=1, keepdim=True) + 1e-4
        fc8_SEC_Softmax = probs / torch.sum(probs, dim=1, keepdim=True)

        loss_s = SeedLossLayer(fc8_SEC_Softmax, DenseGT)
        loss_e = ExpandLossLayer(fc8_SEC_Softmax, labels)
        CRFResult = CRFLayer(fc8_SEC_Softmax, images, 10)
        loss_c = ConstrainLossLayer(fc8_SEC_Softmax, CRFResult)

        print loss_s.cpu().data[0], loss_e.cpu().data[0], loss_c.cpu().data[0]  # , loss_sa.cpu().data[0]
        loss = loss_s + loss_e + loss_c

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        if (iter + 1) % 100 == 0:
            print('[%4d] loss: %f' % (iter + 1, loss.data[0]))
        if (iter + 1) % 500 == 0:
            miou = miou_eval(model, '/home/tk/WSSS/dataset/VOC/ImageSets/val.txt')
            writer.add_scalar('data/test_accuracy', miou, int((iter + 1) / 500))

        if iter == 1500:
            print('saving snapshot')
            save_name = os.path.join(args.snapshot_dir, args.save_name + '_' + str(iter) + '.pth.tar')
            torch.save(model.state_dict(), save_name)
            start_time = time.time()

        if iter % 3000 == 0 and iter != 0:
            lr = lr * 0.1
            optimizer = adjust_learning_rate(optimizer, lr)
            print('learning rate kept at ', lr)
            duration = time.time() - start_time
            print('time: %2dh %2dm %2ds' % (duration // 3600, (duration % 3600) // 60, duration % 60))
            print('saving snapshot')
            save_name = os.path.join(args.snapshot_dir, args.save_name + '_' + str(iter) + '.pth.tar')
            torch.save(model.state_dict(), save_name)
            start_time = time.time()

    print('Last snapshot')
    save_name = os.path.join(args.snapshot_dir, args.save_name + '_last.pth.tar')
    torch.save(model.state_dict(), save_name)
    t_duration = time.time() - total_start
    print('Total Time taken: %2dh %2dm %2ds' % (t_duration // 3600, (t_duration % 3600) // 60, t_duration % 60))


def get_data_from_chunk_v2_FromSnapshot_noScaleCrop(chunk, gt_path, img_path, dict_src):
    dim = 321
    images = np.zeros((dim, dim, 3, len(chunk)))
    # gt = np.zeros((20, len(chunk)))
    DenseGT = np.zeros((41, 41, 21, len(chunk)))
    img_ids = []
    flip_idx = []

    for i, piece in enumerate(chunk):
        img_name = piece.split(' ')[0]
        img_id = piece.split(' ')[1]
        img_ids += [img_name[:-4]]


        flip_p = random.uniform(0, 1)
        cues_i = dict_src[u'' +  str(img_id) + '_cues']
        labels_i = dict_src[u'' + str(img_id) + '_labels']

        ## pre-processing image
        img_temp = cv2.imread(os.path.join(img_path, img_name))
        img_temp = cv2.resize(img_temp, (321, 321))
        img_temp = img_temp.astype('float')
        img_temp[:, :, 0] = img_temp[:, :, 0] - 104.008
        img_temp[:, :, 1] = img_temp[:, :, 1] - 116.669
        img_temp[:, :, 2] = img_temp[:, :, 2] - 122.675
        img_temp = flip(img_temp, flip_p)
        images[:, :, :, i] = img_temp

        ## pre-processing class label
        labels = torch.zeros(len(img_ids), 1, 1, 21)
        for i, image_id in enumerate(img_ids):
            for j in range(len(labels_i)):
                labels[i, 0, 0, labels_i[j]] = 1.0

        ## pre-processing localization cue
        gt_temp = np.zeros((41,41))
        H = gt_temp.shape[0]
        W = gt_temp.shape[1]
        for idx in range(len(cues_i[0])):
            if cues_i[1, idx] < H and cues_i[2, idx] < W:
                gt_temp[cues_i[1, idx], cues_i[2, idx]] = cues_i[0, idx]

        if flip_p > 0.5:
            gt_temp = gt_temp[:, ::-1]
        gt_temp = gt_temp.astype('float')

        gt_temp2 = np.zeros((41, 41, 21))
        for lab in labels_i:
            gt_temp2[:, :, lab] = (gt_temp == lab).astype('float')
        DenseGT[:, :, :, i] = gt_temp2

    labels = Variable(labels.cuda())
    images = images.transpose((3, 2, 0, 1))
    images = Variable(torch.from_numpy(images).float()).cuda()
    DenseGT = DenseGT.transpose((3, 2, 0, 1))
    DenseGT = Variable(torch.from_numpy(DenseGT).float()).cuda()

    return images, DenseGT, labels, img_ids, flip_idx

if __name__ == '__main__':
    train_v1()
    weight_name = os.path.join(args.snapshot_dir, args.save_name + '_last.pth.tar')
