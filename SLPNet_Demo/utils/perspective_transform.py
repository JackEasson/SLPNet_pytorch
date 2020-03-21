from cv2 import getPerspectiveTransform
import numpy as np
from torch import meshgrid, arange, stack, from_numpy, tensor
import torch.nn as nn
import torch.nn.functional as F
from utils.GTProcessing import corner2bbox_int


# 生成透视对应关系：变换后图每个点对应到原图的归一化位置，
# 对应affine_grid函数功能来写

def perspective_grid(perspectiveMatrix, target_size):
    """
    :param perspectiveMatrix: from cv2.getPerspectiveTransform
    :param target_size: (w, h)
    :return:
    """
    W, H = target_size
    W_float = float(W)
    H_float = float(H)
    pers_grid = np.zeros((1, H, W, 2), np.float32)
    m = perspectiveMatrix
    for X, Y in [(X, Y) for X in range(W) for Y in range(H)]:
        # calculate Z
        Z = m[2][0] * X + m[2][1] * Y + m[2][2]
        x = (m[0][0] * X + m[0][1] * Y + m[0][2]) / Z
        y = (m[1][0] * X + m[1][1] * Y + m[1][2]) / Z
        x = (x - (W_float - 1) / 2) / ((W_float - 1) / 2)
        y = (y - (H_float - 1) / 2) / ((H_float - 1) / 2)
        pers_grid[0, Y, X, :] = [x, y]
    return pers_grid


# tensor version
def perspective_grid_tensor(perspectiveMatrix, target_size, device):
    """
    :param perspectiveMatrix: from cv2.getPerspectiveTransform, tensor
    :param target_size: (w, h)
    :return:
    """
    W, H = target_size
    # pers_grid = torch.zeros((1, H, W, 2), dtype=torch.float)
    m = perspectiveMatrix
    Y_grid, X_grid = meshgrid([arange(0, H), arange(0, W)])
    Y_grid = Y_grid.float().to(device)
    X_grid = X_grid.float().to(device)
    # calculate Z
    Z = m[2][0] * X_grid + m[2][1] * Y_grid + m[2][2]
    x = (m[0][0] * X_grid + m[0][1] * Y_grid + m[0][2]) / Z
    y = (m[1][0] * X_grid + m[1][1] * Y_grid + m[1][2]) / Z
    x = (x - (W - 1.0) / 2.0) / ((W - 1.0) / 2.0)
    y = (y - (H - 1.0) / 2.0) / ((H - 1.0) / 2.0)
    pers_grid = stack([x, y], dim=-1)  # size(H, W, 2)
    return pers_grid


class PerspectiveTrans(nn.Module):
    def __init__(self):
        super(PerspectiveTrans, self).__init__()

    def forward(self, fea_maps, corners_tensor, target_size):
        """
        :param fea_maps: feature maps to Perspective Transform, size(C, H, W)
        :param corners_tensor: size(N, 8), N corner pairs number
        :param target_size: (w, h)
        :return:
        """
        device = fea_maps.device
        t_W, t_H = target_size
        bbox = corner2bbox_int(corners_tensor)
        img_idx = 1
        pers_maps_list = []
        for b, c in zip(bbox, corners_tensor):
            # print(b)
            # 防止出现0
            b[3] = b[3] if b[3] - b[1] > 0 else b[3] + 1
            b[2] = b[2] if b[2] - b[0] > 0 else b[2] + 1
            H = b[3] - b[1]
            W = b[2] - b[0]
            wrap_maps = fea_maps[:, b[1]:b[3], b[0]:b[2]]
            # print('wrap_maps', wrap_maps.shape)
            # cv2.imshow('wrap', wrap_maps.data.numpy().transpose(1, 2, 0))
            # 透视变换
            # 原点位置
            srcpoints = (c.reshape(4, 2).cpu() - tensor([b[0], b[1]]).float()).numpy()
            # 原点顺序 左上 左下 右下 右上
            srcpoints = np.array([srcpoints[0], srcpoints[1], srcpoints[2], srcpoints[3]])
            # print('srcpoints', srcpoints)
            # 变换后位置
            canvaspoints = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
            # print('canvaspoints', canvaspoints)
            # 计算转换矩阵
            perspectiveMatrix = getPerspectiveTransform(np.array(canvaspoints), np.array(srcpoints))
            perspectiveMatrix = from_numpy(perspectiveMatrix)
            pers_grid = perspective_grid_tensor(perspectiveMatrix, (W, H), device)
            # print(wrap_maps.shape)
            pers_maps = F.grid_sample(wrap_maps.unsqueeze(0), pers_grid.unsqueeze(0))  # torch.from_numpy(pers_grid)
            pers_maps = F.interpolate(pers_maps, size=(t_H, t_W), mode='bilinear', align_corners=False)
            pers_maps_list.append(pers_maps)
            #cv2.imshow('pers%d' % img_idx, pers_maps.data.squeeze().detach().numpy().transpose(1, 2, 0))
            #img_idx += 1
            #cv2.waitKey()
        return pers_maps_list