import time
import torch
import torch.nn as nn
from model.basic_modules import *
from model.other_models.shufflenet import ShuffleNetV2
from module.det_part.detection_head import coord_trans2real_batch


# similar to mobilenetv2: Inverted residuals
class BasicResidualsBlock(nn.Module):
    def __init__(self, chn_in, chn_out=None, expansion_ratio=1.0):
        super(BasicResidualsBlock, self).__init__()
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


class OtherDetectionNet(nn.Module):
    def __init__(self, model='shufflenet', mode='small'):
        super(OtherDetectionNet, self).__init__()
        assert model in ('shufflenet', 'mobilenet')
        assert mode in ('small', 'large')
        if model == 'shufflenet':
            if mode == 'small':
                self.backbone = ShuffleNetV2(width_mult=1.0)
            else:
                self.backbone = ShuffleNetV2(width_mult=1.5)
        elif model == 'mobilenet':
            if mode == 'small':
                # self.backbone = MobileNetV3_Small()
                pass
            else:
                #self.backbone = MobileNetV3_Large()
                pass
        out_channel = self.backbone.out_channel
        self.post_layers = nn.Sequential(
            BasicResidualsBlock(chn_in=out_channel, chn_out=out_channel // 2, expansion_ratio=1.0),
            BasicResidualsBlock(chn_in=out_channel // 2, chn_out=out_channel // 4, expansion_ratio=1.0),
            conv_bn(inp=out_channel // 4, oup=out_channel // 4, stride=1),
            nn.Conv2d(out_channel // 4, 12, kernel_size=1, stride=1, padding=0, bias=True)
        )
        """
        self.post_layers = nn.Sequential(
            conv_bn(inp=out_channel, oup=out_channel // 2, stride=1),
            conv_1x1_bn(inp=out_channel // 2, oup=out_channel // 2),
            conv_bn(inp=out_channel // 2, oup=out_channel // 4, stride=1),
            nn.Conv2d(out_channel // 4, 12, kernel_size=1, stride=1, padding=0, bias=True)
        )"""

    def forward(self, x):
        x0, x1, x_add = self.backbone(x)
        x_pred = self.post_layers(x_add)  # 0.36s
        x_pred = x_pred.permute(0, 2, 3, 1).contiguous()
        x_pred1 = x_pred[..., :4]
        x_pred1 = torch.sigmoid(x_pred1)
        x_pred2 = x_pred[..., 4:]
        x_pred2 = coord_trans2real_batch(x_pred2)
        x_pred = torch.cat([x_pred1, x_pred2], dim=-1)
        return x0, x1, x_pred


if __name__ == '__main__':
    x = torch.rand(1, 3, 512, 512)
    t0 = time.time()
    # net = OtherDetectionNet(model='mobilenet', mode='large').eval()
    net = SSDetectionNet()
    for i in range(100):
        y = net(x)
        #print(y.shape)
        #for yy in y:
           #print(yy.shape)
    t1 = time.time()
    print((t1 - t0) / 100)