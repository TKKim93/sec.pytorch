import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import vgg16_20M
import DeepLabLargeFOV
import numpy as np

def outS(i):
    i = int(i)
    i = (i+1)/2
    i = int(np.ceil((i+1)/2.0))
    i = (i+1)/2
    return i

class InitDeepLabLargeFOV(nn.Module):
    def __init__(self, pretrained=False):
        super(InitDeepLabLargeFOV, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), (2, 2), (1, 1), ceil_mode=True),
            nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), (2, 2), (1, 1), ceil_mode=True),
            nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), (2, 2), (1, 1), ceil_mode=True),
            nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), (1, 1), (1, 1), ceil_mode=True),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (2, 2), dilation=2),
            nn.ReLU(),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (2, 2), dilation=2),
            nn.ReLU(),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (2, 2), dilation=2),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), (1, 1), (1, 1), ceil_mode=True),
            nn.AvgPool2d((3, 3), (1, 1), (1, 1), ceil_mode=True),  # AvgPool2d,
            nn.Conv2d(512, 1024, (3, 3), (1, 1), (12, 12), dilation=12),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(1024, 1024, (1, 1)),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(1024, 21, (1, 1)),
            # nn.LogSoftmax(dim=1),
        )

        self.load_weight(pretrained)

    def forward(self, x):
        input_size = x.size()[2]
        self.interp1 = nn.UpsamplingBilinear2d(size=(int(input_size * 0.75) + 1, int(input_size * 0.75) + 1))
        self.interp2 = nn.UpsamplingBilinear2d(size=(int(input_size * 0.5) + 1, int(input_size * 0.5) + 1))
        self.interp3 = nn.UpsamplingBilinear2d(size=(outS(input_size), outS(input_size)))
        out = []
        x2 = self.interp1(x)
        x3 = self.interp2(x)
        out.append(self.net(x))  # for original scale
        out.append(self.interp3(self.net(x2)))  # for 0.75x scale
        out.append(self.net(x3))  # for 0.5x scale

        x2Out_interp = out[1]
        x3Out_interp = self.interp3(out[2])
        temp1 = torch.max(out[0], x2Out_interp)
        out.append(torch.max(temp1, x3Out_interp))
        return out

    def load_weight(self, pretrained):
        params = list(self.parameters())

        if pretrained:
            base = DeepLabLargeFOV.DeepLabLargeFOV
            base.load_state_dict(torch.load('DeepLabLargeFOV/DeepLabLargeFOV.pth'))
        else:
            base = vgg16_20M.vgg16_20M
            base.load_state_dict(torch.load('/home/tk/WSSS/robotics_sec/vgg16_20M/vgg16_20M.pth'))
        org_params = list(base.parameters())

        count = 0
        for p, src_p in zip(params, org_params):
            if p.shape == src_p.shape:
                p.data[:] = src_p.data[:]
                count += 1
        print('[%d] pre-trained weights are loaded' % count)


def check_net():
    net = InitDeepLabLargeFOV(True)
    net.cuda()

    input = Variable(torch.zeros([1, 3, 321, 321]).cuda())
    out = net(input)
    print 'out     : ', out.shape


if __name__=='__main__':
    check_net()
