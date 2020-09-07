# -*- coding: utf-8 -*-

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from module.det_part.detection_head import coord_trans2real_batch
# Python拥有一些内置的数据类型，比如str, int, list, tuple, dict等，
# collections模块在这些内置数据类型的基础上，提供了几个额外的数据类型
# OrderedDict: 有序字典
from collections import OrderedDict
from torch.nn import init
import math


# 普通带BN与Relu的conv
def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)  # 进行原地操作,直接操作节省运算
    )


# 1*1带BN与Relu的conv
def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def channel_shuffle(x, groups):
    # 如果 x 是一个 Variable，那么 x.data 则是一个 Tensor
    # 解析维度信息
    batchsize, num_channels, height, width = x.data.size()
    # 整除
    channels_per_group = num_channels // groups

    # reshape，view函数旨在reshape张量形状
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    # transpose操作2D矩阵的转置，即只能转换两个维度
    # contiguous一般与transpose，permute,view搭配使用，使之变连续
    # 即使用transpose或permute进行维度变换后，调用contiguous，然后方可使用view对维度进行变形
    # 1，2维度互换
    # Size(batchsize, channels_per_group, groups, height, width)
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


# 构建ShuffleNetV2 block
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, benchmodel):
        super(InvertedResidual, self).__init__()
        self.benchmodel = benchmodel
        self.stride = stride
        assert stride in [1, 2]

        oup_inc = oup // 2

        if self.benchmodel == 1:  # 对应block  c，有通道拆分
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
        else:  # 对应block  d，无通道拆分
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

    # python staticmethod 返回函数的静态方法
    @staticmethod
    def _concat(x, out):
        # concatenate along channel axis
        # torch.cat((A,B),dim)，第二维度拼接，通道数维度
        return torch.cat((x, out), 1)

    def forward(self, x):
        if 1 == self.benchmodel:
            x1 = x[:, :(x.shape[1] // 2), :, :]
            x2 = x[:, (x.shape[1] // 2):, :, :]
            out = self._concat(x1, self.banch2(x2))
        elif 2 == self.benchmodel:
            out = self._concat(self.banch1(x), self.banch2(x))

        return channel_shuffle(out, 2)  # 两路通道混合


# different feature size sum to detection
class FeatureSumModule(nn.Module):
    def __init__(self, *feature_infos):
        """
        :param feature_infos: element 0 means sizes((32, 32), (16, 16), ...), element 1 means channels
        """
        super(FeatureSumModule, self).__init__()
        print('Feature fusion: size: ', feature_infos[0], '  chn: ', feature_infos[1])
        assert len(feature_infos[0]) == len(feature_infos[1])
        self.feature_infos = feature_infos
        self.feature_num = len(feature_infos[0])
        feature_size = [x[0] for x in self.feature_infos[0]]
        for i in range(self.feature_num - 1):
            assert feature_size[i] > feature_size[i + 1]
        self.targetC = self.feature_infos[1][1]
        self.targetH, self.targetW = self.feature_infos[0][1]
        self.layers = []
        for i in range(self.feature_num):
            if i == 1:
                continue
            else:
                self.layers.append(conv_1x1_bn(self.feature_infos[1][i], self.targetC))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x_list):
        x_add = x_list[1]
        assert len(x_list) == self.feature_num
        count = 0
        for i, x in enumerate(x_list):
            if i < 1:
                # interpolate before conv, smaller calculation
                x = F.interpolate(x, size=(self.targetH, self.targetW), mode='bilinear', align_corners=True)
                x = self.layers[count](x)
                count += 1
            elif i == 1:
                continue
            else:
                # interpolate after conv, smaller calculation
                x = self.layers[count](x)
                x = F.interpolate(x, size=(self.targetH, self.targetW), mode='bilinear', align_corners=True)
                count += 1
            x_add = torch.add(x_add, x)
        return x_add


class ShuffleNetV2(nn.Module):
    def __init__(self, input_size=512, width_mult=1.):
        # 继承父类初始化
        super(ShuffleNetV2, self).__init__()

        assert input_size % 32 == 0

        self.stage_repeats = [4, 8, 4]
        self.repeat_sum = [4, 12, 16]
        # index 0 is invalid and should never be called.
        # only used for indexing convenience.
        # 对应不同尺寸的网络（Table 5）
        if width_mult == 0.5:
            self.stage_out_channels = [-1, 24, 48, 96, 192, 1024]
        elif width_mult == 1.0:
            self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif width_mult == 1.5:
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif width_mult == 2.0:
            self.stage_out_channels = [-1, 24, 224, 488, 976, 2048]
        else:  # 引发一个异常
            raise ValueError(
                """{} groups is not supported for 
                1x1 Grouped Convolutions""".format(width_mult))

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.conv1 = conv_bn(3, input_channel, 2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.features = []
        # building inverted residual blocks
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]  # 分别取4,8,4
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

        # building last several layers
        sizes = [(input_size // (2 ** i), input_size // (2 ** i)) for i in range(3, 6)]
        channels = self.stage_out_channels[2: -1]
        self.sumlayers = FeatureSumModule(sizes, channels)
        self.out_channel = self.stage_out_channels[3]

    def forward(self, x):
        x = self.conv1(x)
        x0 = self.maxpool(x)
        x1 = self.features[:self.repeat_sum[0]](x0)
        x2 = self.features[self.repeat_sum[0]:self.repeat_sum[1]](x1)
        x3 = self.features[self.repeat_sum[1]:self.repeat_sum[2]](x2)
        # print(x1.shape, x2.shape, x3.shape)
        x_pred = self.sumlayers([x1, x2, x3])
        return x0, x1, x_pred


def shufflenetv2(width_mult=1.5):
    model = ShuffleNetV2(width_mult=width_mult)
    return model


if __name__ == "__main__":
    """Testing
    """
    x = torch.rand(1, 3, 512, 512)
    t0 = time.time()
    net = shufflenetv2().eval()
    for i in range(100):
        y = net(x)
        # for yy in y:
        # print(yy.shape)
    t1 = time.time()
    print((t1 - t0) / 100)