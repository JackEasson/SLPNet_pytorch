import torch
import numpy as np
import time


# calculate the center of arbitrary quadrangle
def calc_centers(corners):
    """
    :param corners: tensor(N, 8), eight corner coordinate, clockwise, or size(B, H, W, 8)
    :return: centers tensor(N, 2)
    """
    device = corners[0].device
    if len(corners.shape) == 2:
        center_x = (corners[:, 0] + corners[:, 2] + corners[:, 4] + corners[:, 6]) / 4
        center_y = (corners[:, 1] + corners[:, 3] + corners[:, 5] + corners[:, 7]) / 4
        return torch.stack([center_x, center_y], dim=-1).to(device)

    elif len(corners.shape) == 1:
        corner = corners
        center_x = (corner[0] + corner[2] + corner[4] + corner[6]) / 4
        center_y = (corner[1] + corner[3] + corner[5] + corner[7]) / 4
        return torch.stack([center_x, center_y], dim=-1).to(device)

    # aim at total map, out size(B, H, W, 2)
    elif len(corners.shape) > 2:
        center_x = (corners[..., 0] + corners[..., 2] + corners[..., 4] + corners[..., 6]) / 4
        center_y = (corners[..., 1] + corners[..., 3] + corners[..., 5] + corners[..., 7]) / 4
        return torch.stack([center_x, center_y], dim=-1).to(device)


# use cross product to distinguish a point if in a quadrangle
def distinguish_point_pos(corners_list, point):
    """
    :param corners_list: tensor(8), eight corner coordinate, clockwise
    :param point: tensor(N, 2), to be distinguished
    :return: bool tensor(N)
    """
    assert corners_list.shape[0] == 8
    A = corners_list[0:2]
    B = corners_list[2:4]
    C = corners_list[4:6]
    D = corners_list[6:8]
    P = point
    AB = B - A
    AP = P - A
    # ABXAP = (b.x - a.x, b.y - a.y) x (p.x - a.x, p.y - a.y)
    # = (b.x -a.y)(p.y - a.y) -(b.y - a.y)(p.x - a.x)
    ABXAP = (AB[0] * AP[:, 1]) - (AB[1] * AP[:, 0])  # size(N)
    # print(ABXAP)
    BC = C - B
    BP = P - B
    BCXBP = (BC[0] * BP[:, 1]) - (BC[1] * BP[:, 0])
    CD = D - C
    CP = P - C
    CDXCP = (CD[0] * CP[:, 1]) - (CD[1] * CP[:, 0])
    DA = A - D
    DP = P - D
    DAXDP = (DA[0] * DP[:, 1]) - (DA[1] * DP[:, 0])
    """
    if (ABXAP >= 0 and BCXBP >= 0 and CDXCP >= 0 and DAXDP >= 0) or \
        (ABXAP < 0 and BCXBP < 0 and CDXCP < 0 and DAXDP < 0):
        return True
    else:
        return False"""
    distin_list = torch.zeros_like(ABXAP).byte()
    # t0 = time.time()
    idx1 = (ABXAP >= 0) * (BCXBP >= 0) * (CDXCP >= 0) * (DAXDP >= 0)
    idx2 = (ABXAP < 0) * (BCXBP < 0) * (CDXCP < 0) * (DAXDP < 0)
    distin_list[idx1] = True
    distin_list[idx2] = True
    # t1 = time.time()
    # print('找点for循环时间', t1-t0)
    return distin_list


def corner2bbox_int(corners_list):
    """
    :param corners_list: [tensor, tensor ...]
    :return: list of box, int, not tensor
    """
    bbox_list = []
    for corners in corners_list:
        corners = corners.view(-1)
        left, _ = torch.min(corners[::2], dim=-1)
        top, _ = torch.min(corners[1::2], dim=-1)
        right, _ = torch.max(corners[::2], dim=-1)
        bottom, _ = torch.max(corners[1::2], dim=-1)
        bbox = torch.stack([left, top, right, bottom], dim=-1)
        bbox_list.append(bbox.int())
    return bbox_list


def corner2bbox(corners_list):
    """
    :param corners_list: [tensor, tensor ...]
    :return: list of box, float, not tensor
    """
    bbox_list = []
    for corners in corners_list:
        corners = corners.view(-1)
        """
        left = torch.min(corners[::2]).int().item()
        top = torch.min(corners[1::2]).int().item()
        right = torch.max(corners[::2]).int().item()
        bottom = torch.max(corners[1::2]).int().item()
        bbox_list.append([left, top, right, bottom])"""
        left, _ = torch.min(corners[::2], dim=-1)
        top, _ = torch.min(corners[1::2], dim=-1)
        right, _ = torch.max(corners[::2], dim=-1)
        bottom, _ = torch.max(corners[1::2], dim=-1)
        bbox = torch.stack([left, top, right, bottom], dim=-1)
        bbox_list.append(bbox)
    return bbox_list


def corner2bboxSingle(corners):
    """
    :param corners: size(8)
    :return: size(4)
    """
    left, _ = torch.min(corners[::2], dim=-1)
    top, _ = torch.min(corners[1::2], dim=-1)
    right, _ = torch.max(corners[::2], dim=-1)
    bottom, _ = torch.max(corners[1::2], dim=-1)
    bbox = torch.stack([left, top, right, bottom], dim=-1)
    return bbox


def corner2bboxMulti(corners):
    """
    :param corners: size(N, 8)
    :return: size(N, 4)
    """
    left, _ = torch.min(corners[:, ::2], dim=-1)
    top, _ = torch.min(corners[:, 1::2], dim=-1)
    right, _ = torch.max(corners[:, ::2], dim=-1)
    bottom, _ = torch.max(corners[:, 1::2], dim=-1)
    bbox = torch.stack([left, top, right, bottom], dim=-1)
    return bbox


def corner2bboxHW(corners_maps):
    """
    :param corners_maps: tensor size(B, H, W, 8)
    :return: size(B, H, W, 4)
    """
    left, _ = torch.min(corners_maps[..., ::2], dim=-1)
    top, _ = torch.min(corners_maps[..., 1::2], dim=-1)
    right, _ = torch.max(corners_maps[..., ::2], dim=-1)
    bottom, _ = torch.max(corners_maps[..., 1::2], dim=-1)
    bbox = torch.stack([left, top, right, bottom], dim=-1)
    return bbox


def scale_distribute(corners, splitValue=(3670, 10780)):
    """
    :param corners: tensor, size(8) from the same image, different areas. In the scope of input size (512).
    split criterion: form K_Means, 5365-11666
    :return:
    """
    assert len(splitValue) == 2
    assert corners.shape[-1] == 8
    assert len(corners.shape) == 1
    # N = corners.shape[0]
    b_boxes = corner2bboxSingle(corners)
    b_sizes = (b_boxes[2] - b_boxes[0]) * (b_boxes[3] - b_boxes[1])
    # print(b_sizes)
    if b_sizes <= splitValue[0]:
        corner_distr = 0
    elif splitValue[0] <= b_sizes <= splitValue[1]:
        corner_distr = 1
    else:
        corner_distr = 2
    return corner_distr


def corner_scale_extend(corner, effective_ratio, ignore_ratio, max_size=(512, 512)):
    """
    :param corner: tensor size(8)
    :param effective_ratio:
    :param ignore_ratio:
    :param max_size
    :return: as input 'corner'
    """
    center = calc_centers(corner)
    effective_corner = ((corner.reshape(4, 2) - center) * effective_ratio + center).reshape(-1)
    effective_corner[0::2] = torch.clamp(effective_corner[0::2], min=0, max=max_size[0])
    effective_corner[1::2] = torch.clamp(effective_corner[1::2], min=0, max=max_size[1])
    ignore_corner = ((corner.reshape(4, 2) - center) * ignore_ratio + center).reshape(-1)
    ignore_corner[0::2] = torch.clamp(ignore_corner[0::2], min=0, max=max_size[0])
    ignore_corner[1::2] = torch.clamp(ignore_corner[1::2], min=0, max=max_size[1])
    return effective_corner, ignore_corner


def dilate_3x3(bool_grid_maps):
    H, W = bool_grid_maps.shape[:2]
    # print('==', H, W)
    dilate_idx = torch.zeros((H, W)).byte()
    for y, x in [[m, n] for m in range(H) for n in range(W)]:
        if bool_grid_maps[y][x]:
            tl_x = x - 1 if x - 1 >= 0 else 0
            tl_y = y - 1 if y - 1 >= 0 else 0
            br_x = x + 1 if x + 1 <= W - 1 else W - 1
            br_y = y + 1 if y + 1 <= H - 1 else H - 1
            dilate_idx[tl_y:br_y+1, tl_x:br_x+1] = True
    return dilate_idx


if __name__ == '__main__':
    """
    corners = torch.tensor([[0.0, 0.0, 100.0, 10.0, 110.0, 60.0, 10.0, 50.0],
                            [0.0, 0.0, 20.0, 5.0, 25.0, 25.0, 1.0, 20.0],
                            [0.0, 0.0, 200.0, 10.0, 220.0, 160.0, 10.0, 160.0]])"""
    corners = torch.tensor([0.0, 0.0, 50.0, 10.0, 60.0, 60.0, 10.0, 50.0])
    print(torch.min(corners[::2]))
    #point = torch.tensor([[50.0, 140.0], [50.0, 40.0]])
    #print('result', distinguish_point_pos(corners, point).float())
    corner_distr_list = scale_distribute(corners)
    print(corner_distr_list)
    # bbox = corner2bboxSingle(corners)
    # print(bbox)