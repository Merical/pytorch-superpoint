"""latest version of SuperpointNet. Use it!

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import math
from models.frelu import *


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, benchmodel):
        super(InvertedResidual, self).__init__()
        self.benchmodel = benchmodel
        self.stride = stride
        assert stride in [1, 2]

        oup_inc = oup // 2

        if self.benchmodel == 1:
            # assert inp == oup_inc
            self.banch2 = nn.Sequential(
                # pw
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm2d(oup_inc),
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )
        else:
            self.banch1 = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )

            self.banch2 = nn.Sequential(
                # pw
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm2d(oup_inc),
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )

    @staticmethod
    def _concat(x, out):
        # concatenate along channel axis
        return torch.cat((x, out), 1)

    def forward(self, x):
        if 1 == self.benchmodel:
            x1 = x[:, :(x.shape[1] // 2), :, :]
            x2 = x[:, (x.shape[1] // 2):, :, :]
            out = self._concat(x1, self.banch2(x2))
        elif 2 == self.benchmodel:
            out = self._concat(self.banch1(x), self.banch2(x))

        return channel_shuffle(out, 2)


class ShufflePointNet(nn.Module):
    def __init__(self, width_mult=0.0):
        super(ShufflePointNet, self).__init__()
        # index 0 is invalid and should never be called.
        # only used for indexing convenience.
        if width_mult == 0.5:
            self.stage_repeats = [2, 2, 2]
            self.stage_out_channels = [-1, 24, 48, 96, 192]
        elif width_mult == 1.0:
            self.stage_repeats = [2, 2, 2]
            self.stage_out_channels = [-1, 24, 116, 232, 464]
        elif width_mult == 0.0:
            self.stage_repeats = [2, 2, 2]
            self.stage_out_channels = [-1, 64, 64, 128, 128]
        else:
            raise ValueError(
                """{} groups is not supported for
                       1x1 Grouped Convolutions""".format(num_groups))

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.conv1 = conv_bn(1, input_channel, 1)

        self.features = []
        # building inverted residual blocks
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]
            for i in range(numrepeat):
                if i == 0:
                    # inp, oup, stride, benchmodel):
                    self.features.append(InvertedResidual(input_channel, output_channel, 2, 2))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, 1))
                input_channel = output_channel

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        c4, c5, d1 = self.stage_out_channels[-1], 256, 256
        det_h = 65
        # Detector Head.
        self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.bnPa = nn.BatchNorm2d(c5)
        self.convPb = torch.nn.Conv2d(c5, det_h, kernel_size=1, stride=1, padding=0)
        self.bnPb = nn.BatchNorm2d(det_h)
        # Descriptor Head.
        self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.bnDa = nn.BatchNorm2d(c5)
        self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)
        self.bnDb = nn.BatchNorm2d(d1)

        # self.Pfrelu = FReLU(c5)
        # self.Dfrelu = FReLU(c5)
        self.Pfrelu = torch.nn.ReLU(inplace=True)
        self.Dfrelu = torch.nn.ReLU(inplace=True)

        self.output = None

    def forward(self, x):
        _, _, height, width = x.shape

        x = self.conv1(x)
        # x = self.maxpool(x)
        x = self.features(x)

        # Detector Head.
        cPa = self.Pfrelu(self.bnPa(self.convPa(x)))
        semi = self.bnPb(self.convPb(cPa))
        # Descriptor Head.
        cDa = self.Dfrelu(self.bnDa(self.convDa(x)))
        desc = self.bnDb(self.convDb(cDa))

        dn = torch.norm(desc, p=2, dim=1)  # Compute the norm.
        desc = desc.div(torch.unsqueeze(dn, 1))  # Divide by norm to normalize.
        output = {'semi': semi, 'desc': desc}
        self.output = output

        return output

    def process_output(self, sp_processer):
        """
        input:
          N: number of points
        return: -- type: tensorFloat
          pts: tensor [batch, N, 2] (no grad)  (x, y)
          pts_offset: tensor [batch, N, 2] (grad) (x, y)
          pts_desc: tensor [batch, N, 256] (grad)
        """
        from utils.utils import flattenDetection
        # from models.model_utils import pred_soft_argmax, sample_desc_from_points
        output = self.output
        semi = output['semi']
        desc = output['desc']
        # flatten
        heatmap = flattenDetection(semi)  # [batch_size, 1, H, W]
        # nms
        heatmap_nms_batch = sp_processer.heatmap_to_nms(heatmap, tensor=True)
        # extract offsets
        outs = sp_processer.pred_soft_argmax(heatmap_nms_batch, heatmap)
        residual = outs['pred']
        # extract points
        outs = sp_processer.batch_extract_features(desc, heatmap_nms_batch, residual)

        # output.update({'heatmap': heatmap, 'heatmap_nms': heatmap_nms, 'descriptors': descriptors})
        output.update(outs)
        self.output = output
        return output

if __name__ == "__main__":
    import os
    from torchstat import stat
    import time

    with torch.no_grad():
        model = ShufflePointNet().cuda()
        model.eval()
        input = torch.rand([1, 1, 960, 1280]).cuda()
        output = model.forward(input)

        tic = time.time()
        for _ in range(100):
            output = model.forward(input)
        toc = time.time()
    print("Forward Flow Done, cost {} ms".format((toc - tic)*1000/100))

    stat(model.cpu(), (1, 480, 640))