import time
from model.basic_modules import *
from utils.detection_head import coord_trans2real_batch


# create detection stage block
class StageBlock(nn.Module):
    def __init__(self, chn_in, mode, repeat_num, down_chn=None, down_ratio=None):
        """
        :param chn_in:
        :param mode: choose in (0, 1, 2, 3), 0:skip concatenate connection; 1:skip concatenate & add connection;
                                              2:skip add connection; 3: no connection
        :param repeat_num:
        :param down_chn: output channel after down sampler
        :param down_ratio: first down sampler operation change channel to channel * down_ratio
        down_chn and down_ratio only use one
        """
        super(StageBlock, self).__init__()
        assert mode in (0, 1, 2, 3), "Choose model in (0, 1, 2, 3)"
        self.mode = mode
        assert chn_in % 2 == 0
        if down_chn is not None:
            self.down_chn = down_chn
        elif down_ratio is not None:
            self.down_chn = int(chn_in * down_ratio)
        else:
            assert down_chn is not None or down_ratio is not None, "down_chn or down_ratio, one of the args " \
                                                                   "needed to be assignment"
        if mode == 1:
            assert repeat_num > 3, "To create stage block of mode 1, the number of basic block must more than 3"
        self.chn_in = chn_in
        self.repeat_num = repeat_num
        self.features = []

        # concatenate skip connection
        if 1 == self.mode:
            input_channel = chn_in
            output_channel = self.down_chn
            # skip concatenate connection
            for i in range(3):
                if i == 0:
                    self.features.append(InvertedResidual(input_channel, output_channel, 2, 2))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, 1))
                input_channel = output_channel
            # skip add connection
            input_channel = output_channel * 2
            output_channel = input_channel
            for i in range(repeat_num - 3):
                # channel unchangeable block
                self.features.append(InvertedResidual(input_channel, output_channel, 1, 1))
                input_channel = output_channel

        else:  # self.mode in (0, 2, 3)
            input_channel = chn_in
            output_channel = self.down_chn
            for i in range(repeat_num):
                if i == 0:
                    self.features.append(InvertedResidual(input_channel, output_channel, 2, 2))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, 1))
                input_channel = output_channel
        self.layers = nn.Sequential(*self.features)
        # GCModule
        if self.mode in (0, 1):
            self.enhance_block = GCModule(chn_in=input_channel * 2, shrink_ratio=0.25) if self.mode == 0 else \
                                 GCModule(chn_in=input_channel, shrink_ratio=0.25)

    def forward(self, x):
        if 0 == self.mode:
            x = self.layers[0](x)  # down /2
            x_cat = 0
            for i in range(1, self.repeat_num):
                x = self.layers[i](x)
                if i == 1:
                    x_cat = x
                elif i == self.repeat_num - 1:
                    x_cat = torch.cat([x_cat, x], dim=1)
            x_out = self.enhance_block(x_cat)  # GCModule

        elif 1 == self.mode:
            x_cat1 = self.layers[0](x)  # down /2
            x_cat2 = self.layers[1:3](x_cat1)
            # concat x0 and x2
            x = torch.cat([x_cat1, x_cat2], dim=1)
            x = self.enhance_block(x)  # GCModule
            x_original = x
            # remaining layers, skip connection
            remaining_num = self.repeat_num - 3
            for i in range(remaining_num):
                if i % 2 == 0:
                    x_add0 = x_original
                else:
                    x_add1 = x_original
                x = self.layers[3 + i](x)
                x_original = x
                if i >= 1:
                    x = torch.add(x, x_add1) if i % 2 == 0 else torch.add(x, x_add0)
            x_out = x

        elif 2 == self.mode:
            x_0 = self.layers[0](x)
            x = x_0
            x_original = x
            for i in range(self.repeat_num-1):
                if i % 2 == 0:
                    x_add0 = x_original
                else:
                    x_add1 = x_original
                x = self.layers[i + 1](x)
                x_original = x
                if i >= 1:
                    x = torch.add(x, x_add1) if i % 2 == 0 else torch.add(x, x_add0)
            x_out = x

        else:  # 3 == self.mode
            x_out = self.layers(x)
        return x_out


# ===================================== Total Network 1: detection ============================================
class SSNetDet(nn.Module):
    def __init__(self, input_size=512):
        # 继承父类初始化
        super(SSNetDet, self).__init__()
        assert input_size in (512, 640)
        assert input_size % 32 == 0
        # ================== detection layers ================
        self.stage_repeats = (4, 7, 4)
        self.stageblock_inp = (32, 80, 192)
        self.down_channel = (40, 96, 224)
        self.stageblock_oup = (80, 192, 224)
        self.mode = (0, 1, 2)
        # self.down_ratio = (1.0, 1.0, 1.2, 1.0)
        self.stemblock = StemBlock(3, 32)
        self.stages = []
        for i in range(len(self.stage_repeats)):
            self.stages.append(StageBlock(self.stageblock_inp[i], mode=self.mode[i],
                                          repeat_num=self.stage_repeats[i],
                                          down_chn=self.down_channel[i],
                                          down_ratio=None))
        self.stages = nn.Sequential(*self.stages)
        sizes = [(input_size // (2**i), input_size // (2**i)) for i in range(3, 6)]
        channels = [80, 192, 224]
        self.sumlayers = FeatureSumModule(sizes, channels)
        self.GC_enhance_module = GCModule(192, shrink_ratio=0.25)

        # detection head layers
        self.post_layers = nn.Sequential(
            BasicResidualBlock(chn_in=192, chn_out=96, expansion_ratio=1.0),
            BasicResidualBlock(chn_in=96, chn_out=48, expansion_ratio=1.0),
            conv_bn(inp=48, oup=48, stride=1),
            nn.Conv2d(48, 12, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x):
        # detection backbone
        x0 = self.stemblock(x)  # 128
        x1 = self.stages[0](x0)  # 64
        x2 = self.stages[1](x1)  # 32
        x3 = self.stages[2](x2)  # 16
        x_add = self.sumlayers([x1, x2, x3])
        x_mul = self.GC_enhance_module(x_add)  # total: 0.196s
        x_pred = self.post_layers(x_mul)  # 0.36s
        x_pred = x_pred.permute(0, 2, 3, 1).contiguous()
        x_pred1 = x_pred[..., :4]
        x_pred1 = torch.sigmoid(x_pred1)
        x_pred2 = x_pred[..., 4:]
        x_pred2 = coord_trans2real_batch(x_pred2)
        x_pred = torch.cat([x_pred1, x_pred2], dim=-1)
        return x0, x1, x_pred


# ===================================== Total Network 2: recognition ============================================
class SSNetRegOriginal(nn.Module):  # SSNetRegOriginal
    def __init__(self, class_num):
        super(SSNetRegOriginal, self).__init__()
        self.stage0 = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, stride=1, padding=1, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            InvertedResidual(24, 24, 1, 1)
        )
        # stage 1
        self.down_layer1 = ParallelDownBlock(chn_in=24, chn_out=40, mode='max', stride=(2, 2))  # (72, 24)
        self.stage1 = nn.Sequential(
            InvertedResidual(40, 40, 1, 1),
            InvertedResidual(40, 40, 1, 1)
        )
        self.avg_context1 = ParallelDownBlock(chn_in=40, chn_out=40, mode='mean', stride=(1, 4))

        # stage 2
        self.down_layer2 = ParallelDownBlock(chn_in=40, chn_out=64, mode='max', stride=(1, 2))  # (36, 24)
        self.stage2 = nn.Sequential(
            InvertedResidual(64, 64, 1, 1),
            InvertedResidual(64, 64, 1, 1),
        )

        # fc enhance, out channel == 72, (64 + 64 // 8)
        self.enhance2 = GlobalAvgContextEnhanceBlock(chn_in=64, size_hw_in=(24, 36), chn_shrink_ratio=8)
        self.avg_context2 = ParallelDownBlock(chn_in=72, chn_out=72, mode='mean', stride=(1, 2))

        # stage 3
        self.down_layer3 = ParallelDownBlock(chn_in=72, chn_out=80, mode='max', stride=(1, 2))  # (18, 24)
        self.stage3 = nn.Sequential(
            InvertedResidual(80, 80, 1, 1),
            InvertedResidual(80, 80, 1, 1),
            InvertedResidual(80, 80, 1, 1),
        )

        # chn_in = 80 + 72 + 40, chn_out = 192 + 192 // 8 = 216
        self.enhance_last = GlobalAvgContextEnhanceBlock(chn_in=192, size_hw_in=(24, 18), chn_shrink_ratio=8)

        fc_out_chn = 216
        post_in_chn = fc_out_chn
        post_out_chn = post_in_chn // 2
        self.postprocessor = nn.Sequential(
            nn.Conv2d(in_channels=post_in_chn, out_channels=post_out_chn, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2)),
            nn.BatchNorm2d(post_out_chn),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=post_out_chn, out_channels=post_out_chn, kernel_size=(13, 1), stride=(1, 1), padding=(6, 0)),
            nn.BatchNorm2d(post_out_chn),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=(4, 3), stride=(4, 1), padding=(0, 1))
        )
        self.container = nn.Sequential(
            nn.Conv2d(in_channels=post_out_chn, out_channels=class_num, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x0 = self.stage0(x)  # up to 0.015

        x1 = self.down_layer1(x0)
        x1 = x1 + self.stage1(x1)  # up to 0.037

        x2 = self.down_layer2(x1)
        x2 = x2 + self.stage2(x2)
        x2 = self.enhance2(x2)

        x3_1 = self.down_layer3(x2)
        x3_2 = self.stage3[0](x3_1)
        x3_3 = self.stage3[1](x3_2)
        x3_4 = self.stage3[2](x3_1 + x3_3)
        x3 = x3_2 + x3_4

        x_concat = torch.cat([self.avg_context1(x1), self.avg_context2(x2), x3], dim=1)
        # print(x_cat.shape)
        x = self.enhance_last(x_concat)  # up to 0.066, out size(1, 216, 24, 24)
        x = self.container(self.postprocessor(x))  # size(1, class_num, 6, 12)
        logits = torch.mean(x, dim=2)  # size(1, class_num, 12)""""""
        return logits


if __name__ == '__main__':

    x = torch.rand(1, 3, 512, 512)
    t0 = time.time()
    net = SSNetDet().eval()
    for i in range(10):
        y = net(x)
        #for yy in y:
            #print(yy.shape)
    t1 = time.time()
    print((t1 - t0) / 10)
    """
    x = torch.rand(1, 3, 48, 144)
    x1 = torch.rand(1, 32, 6, 18)
    x2 = torch.rand(1, 80, 3, 9)
    t0 = time.time()
    net = SSNetRegv2(class_num=68).eval()
    for i in range(100):
        y = net(x, x1)
        # print(y.shape)
    t1 = time.time()
    print((t1 - t0) / 100)"""