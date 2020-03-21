import torch
import torch.nn as nn
import torch.nn.functional as F
import time


#####################################################################################################
#                                   particularly for detection part
#####################################################################################################
# ==========================[1] public common component ============================
# common 3x3 conv with BN and ReLU
def conv_bn(inp, oup, stride, if_bias=False):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=if_bias),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


# common 1x1 conv with BN and ReLU
def conv_1x1_bn(inp, oup, if_bias=False):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=if_bias),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


# ==========================[2] public function module ============================
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


# ==========================[3] different blocks ============================
# a residual block with depthwise separable convolution
class BasicResidualBlock(nn.Module):
    def __init__(self, chn_in, chn_out=None, expansion_ratio=1.0):
        super(BasicResidualBlock, self).__init__()
        self.mode = 1 if chn_out is None else 2
        chn_expan = int(chn_in * expansion_ratio)
        if self.mode == 1:  # without channel changes
            self.main_branch = nn.Sequential(
                conv_1x1_bn(inp=chn_in, oup=chn_expan),
                nn.Conv2d(chn_expan, chn_expan, kernel_size=3, stride=1,
                          padding=1, groups=chn_expan, bias=False),
                conv_1x1_bn(inp=chn_expan, oup=chn_in)
            )
        if self.mode == 2:  # change channels, multi scales
            self.main_branch = nn.Sequential(
                conv_1x1_bn(inp=chn_in, oup=chn_expan),
                nn.Conv2d(chn_expan, chn_expan, kernel_size=3, stride=1,
                          padding=1, groups=chn_expan, bias=False),
                conv_1x1_bn(inp=chn_expan, oup=chn_out)
            )
            self.conv_1x1 = conv_1x1_bn(inp=chn_in, oup=chn_out)

    def forward(self, x):
        if self.mode == 1:
            x = x + self.main_branch(x)
        else:
            x = self.conv_1x1(x) + self.main_branch(x)
        return x


# for detection
# a composite primary downsampler from Pelee, particularly for detection part
class StemBlock(nn.Module):
    def __init__(self, chn_in, chn_out):
        super(StemBlock, self).__init__()
        self.down_layer1 = conv_bn(inp=chn_in, oup=chn_out, stride=2, if_bias=True)
        self.branch1 = nn.Sequential(
            conv_1x1_bn(inp=chn_out, oup=chn_out // 2, if_bias=False),
            conv_bn(inp=chn_out // 2, oup=chn_out, stride=2, if_bias=False),
        )
        self.branch2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.last_conv = conv_1x1_bn(inp=chn_out * 2, oup=chn_out, if_bias=False)

    def forward(self, x):
        x = self.down_layer1(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.last_conv(x)
        return x


# for recognition
# This module also can change the channels!
class ParallelDownBlock(nn.Module):
    def __init__(self, chn_in, chn_out, mode='max', stride=(2, 2)):
        super(ParallelDownBlock, self).__init__()
        assert mode in ('max', 'mean')
        assert stride in ((2, 2), (1, 2), (1, 4))
        if stride == (2, 2):
            chn_mid = chn_in // 2
            self.branch1 = nn.Sequential(
                conv_1x1_bn(inp=chn_in, oup=chn_mid, if_bias=False),
                conv_bn(inp=chn_mid, oup=chn_in, stride=2, if_bias=False),
            )
            if mode == 'max':
                self.branch2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            else:
                self.branch2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
            self.last_conv = conv_1x1_bn(inp=chn_in * 2, oup=chn_out, if_bias=False)
        elif stride == (1, 2):
            chn_mid = chn_in // 4
            self.branch1 = nn.Sequential(
                conv_1x1_bn(inp=chn_in, oup=chn_mid, if_bias=False),
                nn.Conv2d(chn_mid, chn_in, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2), bias=False),
                nn.BatchNorm2d(chn_in),
                nn.ReLU(inplace=True)
            )
            if mode == 'max':
                self.branch2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 2), padding=(1, 1))
            else:
                self.branch2 = nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 2), padding=(1, 1))
            self.last_conv = conv_1x1_bn(inp=chn_in * 2, oup=chn_out, if_bias=False)
        elif stride == (1, 4):
            chn_mid = chn_in // 4
            self.branch1 = nn.Sequential(
                conv_1x1_bn(inp=chn_in, oup=chn_mid, if_bias=False),
                nn.Conv2d(chn_mid, chn_in, kernel_size=(1, 7), stride=(1, 4), padding=(0, 3), bias=False),
                nn.BatchNorm2d(chn_in),
                nn.ReLU(inplace=True)
            )
            if mode == 'max':
                self.branch2 = nn.MaxPool2d(kernel_size=(3, 5), stride=(1, 4), padding=(1, 2))
            else:
                self.branch2 = nn.AvgPool2d(kernel_size=(3, 5), stride=(1, 4), padding=(1, 2))
            self.last_conv = conv_1x1_bn(inp=chn_in * 2, oup=chn_out, if_bias=False)

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.last_conv(x)
        return x


# original ShuffleNetV2 Block, together with stride=2 and stride=1 modules
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, benchmodel):
        super(InvertedResidual, self).__init__()
        assert benchmodel in (1, 2)
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
        elif self.benchmodel == 2:  # 对应block  d，无通道拆分
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


# ==========================[4] enhance modules =============================
# GC module, from paper: GCNet, aimed at enhancing global context
class GCModule(nn.Module):
    def __init__(self, chn_in, shrink_ratio):
        super(GCModule, self).__init__()
        self.conv_mask = nn.Conv2d(chn_in, 1, kernel_size=1, stride=1)
        self.softmax = nn.Softmax(dim=2)
        self.chn_mid = int(chn_in * shrink_ratio)
        self.channel_add_conv = nn.Sequential(
            nn.Conv2d(chn_in, self.chn_mid, kernel_size=1, stride=1),
            nn.LayerNorm([self.chn_mid, 1, 1]),
            nn.ReLU(inplace=True),  # yapf: disable
            nn.Conv2d(self.chn_mid, chn_in, kernel_size=1, stride=1))

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        input_x = x
        # [N, C, H * W]
        input_x = input_x.view(batch, channel, height * width)
        # [N, 1, C, H * W]
        input_x = input_x.unsqueeze(1)
        # [N, 1, H, W]
        context_mask = self.conv_mask(x)
        # [N, 1, H * W]
        context_mask = context_mask.view(batch, 1, height * width)
        # [N, 1, H * W]
        context_mask = self.softmax(context_mask)
        # [N, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(-1)
        # [N, 1, C, 1]
        context = torch.matmul(input_x, context_mask)
        # [N, C, 1, 1]
        context = context.view(batch, channel, 1, 1)
        return context

    def forward(self, x):
        # size(N, C, 1, 1)
        context = self.spatial_pool(x)
        out = x
        # [N, C, 1, 1]
        channel_add_term = self.channel_add_conv(context)
        out = out + channel_add_term
        return out  # size same as input x


class FeatureSumModule(nn.Module):
    def __init__(self, *feature_infos):
        """
        :param feature_infos: element 0 means sizes((32, 32), (16, 16), ...), element 1 means channels
        """
        super(FeatureSumModule, self).__init__()
        # print('Feature fusion: size: ', feature_infos[0], '  chn: ', feature_infos[1])
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


# for recognition
class GlobalAvgContextEnhanceBlock(nn.Module):
    def __init__(self, chn_in, size_hw_in, chn_shrink_ratio=6):
        """
        :param chn_in:
        :param size_hw_in: (h, w) or [h, w]
        :param chn_shrink_ratio:
        """
        super(GlobalAvgContextEnhanceBlock, self).__init__()
        self.H, self.W = size_hw_in
        chn_out = chn_in // chn_shrink_ratio
        self.layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(chn_in, chn_out, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LayerNorm([chn_out, 1, 1]),
            # nn.ReLU(inplace=True)
        )
        self.BN = nn.BatchNorm2d(chn_out)

    def forward(self, x):
        x_avg = self.layers(x)  # size(B, chn_out)
        x_avg = x_avg.repeat((1, 1, self.H, self.W))  # size(B, chn_out, H, W)
        x_avg = self.BN(x_avg)
        return torch.cat([x, x_avg], dim=1)


# for recognition
# only to generate global context
class GlobalAvgContextEmbeddingBlock(nn.Module):
    def __init__(self, chn_in, size_hw_in, chn_shrink_ratio=6):
        """
        :param chn_in: a number or a list (tuple)
        :param size_hw_in: (h, w) or [h, w]
        :param chn_shrink_ratio:
        """
        super(GlobalAvgContextEmbeddingBlock, self).__init__()
        if isinstance(chn_in, int):
            self.mode = 1
        elif isinstance(chn_in, list) or isinstance(chn_in, tuple):
            self.mode = 2
        else:
            raise ValueError("Wrong argument 'chn_in'! Should be int, list or tuple.")
        self.H, self.W = size_hw_in
        if self.mode == 1:
            chn_out = chn_in // chn_shrink_ratio
            self.layers = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(chn_in, chn_out, kernel_size=1, stride=1, padding=0, bias=False),
                nn.LayerNorm([chn_out, 1, 1]),
                # nn.ReLU(inplace=True)
            )
            self.BN = nn.BatchNorm2d(chn_out)
        else:  # self.mode == 2
            self.feature_num = len(chn_in)
            self.avg_layers = []
            chn_concat = 0
            for i, chn in enumerate(chn_in):
                self.avg_layers.append(nn.AdaptiveAvgPool2d(1))
                chn_concat += chn
            chn_out = chn_concat // chn_shrink_ratio
            self.other_layers = nn.Sequential(
                nn.Conv2d(chn_concat, chn_out, kernel_size=1, stride=1, padding=0, bias=False),
                nn.LayerNorm([chn_out, 1, 1]),
            )
            self.BN = nn.BatchNorm2d(chn_out)

    def forward(self, x):
        # x: a tensor or list(tuple)
        if self.mode == 1:
            x_avg = self.layers(x)  # size(B, chn_out)
            x_avg = x_avg.repeat((1, 1, self.H, self.W))  # size(B, chn_out, H, W)
            x_avg = self.BN(x_avg)
        else:
            x_avg = []
            for i in range(self.feature_num):
                x_avg.append(self.avg_layers[i](x[i]))
            x_avg = torch.cat(x_avg, dim=1)
            x_avg = self.other_layers(x_avg)
            x_avg = x_avg.repeat((1, 1, self.H, self.W))  # size(B, chn_out, H, W)
            x_avg = self.BN(x_avg)
        return x_avg


if __name__ == '__main__':
    """
    x = torch.rand(1, 3, 512, 512)
    t0 = time.time()
    net = SSNetDet().eval()
    for i in range(100):
        y = net(x)
        #for yy in y:
            #print(yy.shape)
    t1 = time.time()
    print((t1 - t0) / 100)
    """
    x1 = torch.rand(1, 32, 6, 18)
    x2 = torch.rand(1, 80, 3, 9)
    t0 = time.time()
    net = GlobalAvgContextEmbeddingBlock(chn_in=(32, 80), size_hw_in=(24, 18), chn_shrink_ratio=4).eval()
    for i in range(1):
        y = net([x1, x2])
        print(y.shape)
    t1 = time.time()
    print((t1 - t0) / 1)