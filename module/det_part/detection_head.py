import numpy as np
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from module.det_part import GTProcessing as gtP
import train_config as tra_cfg


# network total reg part transform to real coordinate
def coord_trans2real_batch(reg_map, stage_lvl=4, S0=16):
    """
    :param reg_map: size(B, 32, 32, 8)
    :param stage_lvl: scale ratio
    :param S0: 16 from fovea box
    :return:
    """
    device = reg_map.device
    zeta = (4 ** stage_lvl * S0) ** 0.5
    # gen grid_center
    B, H, W, _ = reg_map.shape
    y, x = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])
    grid = torch.stack([x, y], dim=-1).float()
    grid_center = (grid + 0.5).unsqueeze(2).unsqueeze(0).repeat((B, 1, 1, 1, 1)).to(device)
    # print('1', grid_center)
    reg_map = reg_map.reshape(B, H, W, 4, 2)
    real_coord_map = reg_map ** 3 * zeta + grid_center * (2 ** stage_lvl)
    real_coord_map = real_coord_map.reshape(B, H, W, -1)
    return real_coord_map


# reg part transform to real coordinate
def coord_trans2real(single_reg_map, stage_lvl=4, S0=16):
    """
    :param single_reg_map: size(32, 32, 8)
    :param stage_lvl: scale ratio
    :param S0: 16 from fovea box
    :return:
    """
    zeta = (4 ** stage_lvl * S0) ** 0.5
    # gen grid_center
    H, W, _ = single_reg_map.shape
    y, x = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])
    grid = torch.stack([x, y], dim=-1).float()
    grid_center = (grid + 0.5).unsqueeze(2)
    single_reg_map = single_reg_map.reshape(H, W, 4, 2)
    real_coord_map = single_reg_map ** 3 * zeta + grid_center * (2 ** stage_lvl)
    real_coord_map = real_coord_map.reshape(H, W, -1)
    return real_coord_map


# get effective and ignore idx maps
def get_spatial_idx(corner_xy, W, H, scale_idx, device):
    """
    :param corner_xy: torch size(8)
    :param W: detection map size
    :param H: as above
    :param scale_idx: corner size to distribute
    :param device:
    :return:
    """
    device = corner_xy.device
    y, x = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])
    grid = torch.stack([x, y], dim=-1).float()
    grid_center = (grid + 0.5).reshape(-1, 2).to(device)
    # zero-tensor w/ (H,W)
    e_spatial_idx = torch.zeros((H, W)).bool()
    i_spatial_idx = torch.zeros((H, W)).bool()

    # calculate corner coordinate
    effective_corner, ignore_corner = gtP.corner_scale_extend(corner_xy,
                                                              tra_cfg.K_Means_args['effective_ratio'][scale_idx],
                                                              tra_cfg.K_Means_args['ignore_ratio'][scale_idx])
    # print(effective_corner, ignore_corner)
    # effective
    eff_bool_grid = gtP.distinguish_point_pos(effective_corner, grid_center)
    eff_bool_grid = eff_bool_grid.reshape(H, W)
    e_spatial_idx[eff_bool_grid] = 1

    # ignore
    ign_bool_grid = gtP.distinguish_point_pos(ignore_corner, grid_center)
    ign_bool_grid = ign_bool_grid.reshape(H, W)
    # dilate process, make it at least has a surrounding ignore areas
    dilate_bool_grid = gtP.dilate_3x3(eff_bool_grid)
    i_spatial_idx[dilate_bool_grid] = 1
    i_spatial_idx[ign_bool_grid] = 1
    i_spatial_idx[eff_bool_grid] = 0

    return e_spatial_idx.to(device), i_spatial_idx.to(device)


# use net's output generate target maps for detection
def detection_target(output_maps, corners_list_512, stage_lvl=4):
    """
    :param output_maps: net output size(B, H, W, C) have transform to real coordinate (0-512) NOTICE !!!!!
    :param corners_list_512: list(B), tensor(N, 8), here B is batch_size, N is obj number in one image, 512 scale
    :param stage_lvl: which level the corners project to
    :return: sample_area_target, coordinate_target
    """
    device = output_maps.device
    num_imgs = len(corners_list_512)
    corners_list_32 = [single_corners / (2 ** stage_lvl) for single_corners in corners_list_512]
    B, H, W, C = output_maps.shape
    # ======================target initial==========================
    # positive and negative area maps
    sample_area_target = torch.zeros((B, H, W)).long().to(device)
    # coordinate target maps, the corresponding gt coordinate
    coordinate_target = torch.zeros((B, H, W, 8)).float().to(device)
    """
    其他的在loss函数里面计算吧
    # distance target maps, ratio of distance
    distance_target = torch.zeros((B, H, W)).float()
    # size target maps, ratio of w / h, arctan
    size_target = torch.zeros((B, H, W)).float()
    # Discrete degree target maps, Mean square error
    discrete_target = torch.zeros((B, H, W)).float()
    """
    # no grad
    # output_maps_detach = output_maps.detach()
    # each image in batch
    for img in range(num_imgs):
        # single_map_detach = output_maps_detach[img]  # size(12, 32, 32)
        single_corners_32 = corners_list_32[img]  # size(N, 8)
        single_corners_512 = corners_list_512[img]
        # calculate pos & neg areas
        obj_num = single_corners_32.shape[0]
        for obj in range(obj_num):
            # which to distribute
            dist_idx = gtP.scale_distribute(single_corners_512[obj], tra_cfg.K_Means_args['split_value'])
            # print('distribute', dist_idx)
            # e_spatial_map, i_spatial_map are all bool tensors
            e_spatial_map, i_spatial_map = get_spatial_idx(single_corners_32[obj], W, H, dist_idx, device)
            # print(i_spatial_map[i_spatial_map == 1].size())
            # print(i_spatial_map.shape)
            coordinate_target[img, e_spatial_map == 1] = single_corners_512[obj]
            # sample_area_target 非背景部分乘以新的i_spatial_map得到相交部分，i_spatial_map除去相交点再赋值
            i_spatial_map = (i_spatial_map.byte() - ((sample_area_target[img] != 0) * i_spatial_map).byte()).bool()
            # print(i_spatial_map[i_spatial_map == 1].size())
            # print(i_spatial_map.shape)
            sample_area_target[img, e_spatial_map == 1] = 1
            sample_area_target[img, i_spatial_map == 1] = -1
    return sample_area_target, coordinate_target


# ------------------------------------------------------------------
# ------------------ [1] Our MG-loss in SLPNet ---------------------
# ------------------------------------------------------------------
class MultiConstraintsGaussDistanceLoss(nn.Module):
    def __init__(self):
        super(MultiConstraintsGaussDistanceLoss, self).__init__()

    @staticmethod
    def gauss_2d(x1, x2, u1, u2, d1, d2):
        m = torch.pow((x1 - u1) / d1, 2)
        n = torch.pow((x2 - u2) / d2, 2)
        return torch.exp(-0.5 * (m + n))

    @staticmethod
    def gen_distance_maps(out_centers, target_centers, out_bbox, target_bbox):
        """
        :param out_centers: size(B, H, W, 2)
        :param target_centers: size(B, H, W, 2)
        :param out_bbox: size(B, H, W, 4)
        :param target_bbox: size(B, H, W, 4)
        :return:
        """
        center_distance = (out_centers[..., 0] - target_centers[..., 0]) ** 2 + \
                          (out_centers[..., 1] - target_centers[..., 1]) ** 2
        union_bbox = gtP.corner2bboxHW(torch.cat([out_bbox, target_bbox], dim=-1))
        corner_distance = (union_bbox[..., 2] - union_bbox[..., 0]) ** 2 + \
                          (union_bbox[..., 3] - union_bbox[..., 1]) ** 2
        distance_maps = center_distance / corner_distance
        return distance_maps

    @staticmethod
    def gen_whwh_maps(coordinate_maps, delta=1e-6):
        # coordinate transform to w,h,w,h, two bbox size
        """
        :param coordinate_maps: size(B, H, W, 8)
        :param delta
        :return:
        """
        w1 = torch.abs(coordinate_maps[..., 4] - coordinate_maps[..., 0])
        w1 = torch.clamp(w1, min=delta)
        h1 = torch.abs(coordinate_maps[..., 5] - coordinate_maps[..., 1])
        h1 = torch.clamp(h1, min=delta)
        w2 = torch.abs(coordinate_maps[..., 2] - coordinate_maps[..., 6])
        w2 = torch.clamp(w2, min=delta)
        h2 = torch.abs(coordinate_maps[..., 7] - coordinate_maps[..., 3])
        h2 = torch.clamp(h2, min=delta)
        whwh_maps = torch.stack([w1, h1, w2, h2], dim=-1)
        return whwh_maps

    @staticmethod
    def focal_for_gauss(sample_target, gauss_out, gauss_target, gamma=2, alpha=0.25, delta=1e-6):
        """
        :param sample_target: positive and negative sample, size(B, H, W)
        :param gauss_out: score, size(B, H, W, 4)
        :param gauss_target: score, size(B, H, W, 4)
        :param gamma for focal loss
        :param alpha for focal loss
        :param delta for log
        :return:
        """
        # positive
        difference_score = torch.abs(gauss_out - gauss_target)  # size(B, H, W, 4)
        y_pos = torch.clamp(difference_score, max=1.0 - delta)
        pos_focal_loss = -alpha * y_pos ** gamma * torch.log(torch.ones_like(y_pos).float() - y_pos)
        pos_focal_loss = torch.sum(pos_focal_loss[sample_target == 1]) / torch.sum(sample_target == 1)
        # print(pos_focal_loss)
        # negative
        y_neg = torch.clamp(gauss_out, max=1.0 - delta)
        neg_focal_loss = -(1.0 - alpha) * y_neg ** gamma * torch.log(torch.ones_like(y_neg).float() - y_neg)
        neg_focal_loss = torch.sum(neg_focal_loss[sample_target == 0]) / torch.sum(sample_target == 0)
        # total focal loss
        return pos_focal_loss + neg_focal_loss

    def gen_loss_target(self, output_maps, coordinate_target, delta_ratio=0.2):
        """
        :param output_maps: net output size(B, H, W, C) have transform to real coordinate (0-512) NOTICE !!!!!
        :param coordinate_target: from fun: detection_target, size(B, H, W, 8)
        :param delta_ratio: trans w / h to delta in 2D-gauss
        :return:
        """
        # device = output_maps.device
        gauss_out = output_maps[..., :4]
        coordinate_out = output_maps[..., 4:]
        """
        # gaussian score
        gaussian_score = torch.zeros((B, H, W, 4)).float()
        # distance target maps, ratio of distance
        distance_target = torch.zeros((B, H, W)).float()
        # size target maps, ratio of w / h, arctan
        size_target = torch.zeros((B, H, W)).float()
        # Discrete degree target maps, Mean square error
        discrete_target = torch.zeros((B, H, W)).float()
        """
        # ====================【1】Gaussian Score ======================
        target_bbox = gtP.corner2bboxHW(coordinate_target)
        target_size_w = (target_bbox[..., 2] - target_bbox[..., 0]).unsqueeze(-1)  # size(B, H, W, 1)
        target_size_h = (target_bbox[..., 3] - target_bbox[..., 1]).unsqueeze(-1)  # size(B, H, W, 1)
        target_size_w = torch.clamp(target_size_w, min=1e-6)
        target_size_h = torch.clamp(target_size_h, min=1e-6)
        gaussian_score_target = self.gauss_2d(coordinate_out[..., 0::2], coordinate_out[..., 1::2],
                                              coordinate_target[..., 0::2], coordinate_target[..., 1::2],
                                              delta_ratio * target_size_w, delta_ratio * target_size_h)
        # =====================【2】discrete loss =======================
        score_means = (gaussian_score_target[..., 0] + gaussian_score_target[..., 1] +
                       gaussian_score_target[..., 2] + gaussian_score_target[..., 3]) / 4
        # 均方根
        discrete_target = torch.sqrt(((gaussian_score_target[..., 0] - score_means) ** 2 +
                                     (gaussian_score_target[..., 1] - score_means) ** 2 +
                                     (gaussian_score_target[..., 2] - score_means) ** 2 +
                                     (gaussian_score_target[..., 3] - score_means) ** 2) / 4 + 1e-8)
        # =====================【3】Distance loss =======================
        out_bbox = gtP.corner2bboxHW(coordinate_out)
        # centers of out and target
        target_centers = gtP.calc_centers(coordinate_target)
        out_centers = gtP.calc_centers(coordinate_out)
        distance_target = self.gen_distance_maps(out_centers, target_centers, out_bbox, target_bbox)
        # =======================【4】Size loss =========================
        whwh_out = self.gen_whwh_maps(coordinate_out)
        whwh_target = self.gen_whwh_maps(coordinate_target)
        size_target = (torch.atan(whwh_out[..., 0] / whwh_out[..., 1]) -
                       torch.atan(whwh_target[..., 0] / whwh_target[..., 1])) ** 2 + \
                      (torch.atan(whwh_out[..., 2] / whwh_out[..., 3]) -
                       torch.atan(whwh_target[..., 2] / whwh_target[..., 3])) ** 2
        size_target = size_target * 2 / (math.pi ** 2)
        return gaussian_score_target, distance_target, size_target, discrete_target

    def forward(self, output_maps, corners_list_512):
        sample_area_target, coordinate_target = detection_target(output_maps, corners_list_512, stage_lvl=4)
        # print(sample_area_target.device, coordinate_target.device)
        # ===================== generate loss targets ====================
        gauss_target, distance_target, size_target, discrete_target = self.gen_loss_target(output_maps,
                                                                                           coordinate_target,
                                                                                           delta_ratio=0.3)
        # print(gauss_target.device, distance_target.device, size_target.device, discrete_target.device)
        # ===================== calculate loss part1 ====================
        gauss_out = output_maps[..., :4]
        coordinate_out = output_maps[..., 4:]
        # 【1】gauss loss
        loss_gauss = torch.sum(torch.ones_like(gauss_target).float() - gauss_target, dim=-1) / 4
        # 【2】distance
        loss_distance = distance_target
        # alpha * V
        loss_v = size_target
        alpha = loss_v / (loss_gauss + loss_v)
        loss_size = alpha * loss_v
        # beta * D
        loss_d = discrete_target
        beta = loss_d / (loss_gauss + loss_d)
        loss_discrete = beta * loss_d

        # ===================== calculate loss part2 ====================
        sample_focal_loss = self.focal_for_gauss(sample_area_target, gauss_out, gauss_target)
        # ===================== total loss =======================
        coord_loss = loss_gauss + loss_distance + loss_size + loss_discrete
        coord_loss = torch.sum(coord_loss[sample_area_target == 1]) / torch.sum(sample_area_target == 1)
        score_loss = sample_focal_loss
        detection_loss = score_loss + coord_loss
        # print(detection_loss, coord_loss, score_loss)
        return detection_loss, coord_loss, score_loss


# ------------------------------------------------------------------
# ------------------ Only Gauss-loss in SLPNet ---------------------
# ------------------------------------------------------------------
class GaussLoss(nn.Module):
    def __init__(self):
        super(GaussLoss, self).__init__()

    @staticmethod
    def gauss_2d(x1, x2, u1, u2, d1, d2):
        m = torch.pow((x1 - u1) / d1, 2)
        n = torch.pow((x2 - u2) / d2, 2)
        return torch.exp(-0.5 * (m + n))

    @staticmethod
    def gen_distance_maps(out_centers, target_centers, out_bbox, target_bbox):
        """
        :param out_centers: size(B, H, W, 2)
        :param target_centers: size(B, H, W, 2)
        :param out_bbox: size(B, H, W, 4)
        :param target_bbox: size(B, H, W, 4)
        :return:
        """
        center_distance = (out_centers[..., 0] - target_centers[..., 0]) ** 2 + \
                          (out_centers[..., 1] - target_centers[..., 1]) ** 2
        union_bbox = gtP.corner2bboxHW(torch.cat([out_bbox, target_bbox], dim=-1))
        corner_distance = (union_bbox[..., 2] - union_bbox[..., 0]) ** 2 + \
                          (union_bbox[..., 3] - union_bbox[..., 1]) ** 2
        distance_maps = center_distance / corner_distance
        return distance_maps

    @staticmethod
    def gen_whwh_maps(coordinate_maps, delta=1e-6):
        # coordinate transform to w,h,w,h, two bbox size
        """
        :param coordinate_maps: size(B, H, W, 8)
        :param delta
        :return:
        """
        w1 = torch.abs(coordinate_maps[..., 4] - coordinate_maps[..., 0])
        w1 = torch.clamp(w1, min=delta)
        h1 = torch.abs(coordinate_maps[..., 5] - coordinate_maps[..., 1])
        h1 = torch.clamp(h1, min=delta)
        w2 = torch.abs(coordinate_maps[..., 2] - coordinate_maps[..., 6])
        w2 = torch.clamp(w2, min=delta)
        h2 = torch.abs(coordinate_maps[..., 7] - coordinate_maps[..., 3])
        h2 = torch.clamp(h2, min=delta)
        whwh_maps = torch.stack([w1, h1, w2, h2], dim=-1)
        return whwh_maps

    @staticmethod
    def focal_for_gauss(sample_target, gauss_out, gauss_target, gamma=2, alpha=0.25, delta=1e-6):
        """
        :param sample_target: positive and negative sample, size(B, H, W)
        :param gauss_out: score, size(B, H, W, 4)
        :param gauss_target: score, size(B, H, W, 4)
        :param gamma for focal loss
        :param alpha for focal loss
        :param delta for log
        :return:
        """
        # positive
        difference_score = torch.abs(gauss_out - gauss_target)  # size(B, H, W, 4)
        y_pos = torch.clamp(difference_score, max=1.0 - delta)
        pos_focal_loss = -alpha * y_pos ** gamma * torch.log(torch.ones_like(y_pos).float() - y_pos)
        pos_focal_loss = torch.sum(pos_focal_loss[sample_target == 1]) / torch.sum(sample_target == 1)
        # print(pos_focal_loss)
        # negative
        y_neg = torch.clamp(gauss_out, max=1.0 - delta)
        neg_focal_loss = -(1.0 - alpha) * y_neg ** gamma * torch.log(torch.ones_like(y_neg).float() - y_neg)
        neg_focal_loss = torch.sum(neg_focal_loss[sample_target == 0]) / torch.sum(sample_target == 0)
        # total focal loss
        return pos_focal_loss + neg_focal_loss

    def gen_loss_target(self, output_maps, coordinate_target, delta_ratio=0.2):
        """
        :param output_maps: net output size(B, H, W, C) have transform to real coordinate (0-512) NOTICE !!!!!
        :param coordinate_target: from fun: detection_target, size(B, H, W, 8)
        :param delta_ratio: trans w / h to delta in 2D-gauss
        :return:
        """
        # device = output_maps.device
        gauss_out = output_maps[..., :4]
        coordinate_out = output_maps[..., 4:]
        """
        # gaussian score
        gaussian_score = torch.zeros((B, H, W, 4)).float()
        # distance target maps, ratio of distance
        distance_target = torch.zeros((B, H, W)).float()
        # size target maps, ratio of w / h, arctan
        size_target = torch.zeros((B, H, W)).float()
        # Discrete degree target maps, Mean square error
        discrete_target = torch.zeros((B, H, W)).float()
        """
        # ====================【1】Gaussian Score ======================
        target_bbox = gtP.corner2bboxHW(coordinate_target)
        target_size_w = (target_bbox[..., 2] - target_bbox[..., 0]).unsqueeze(-1)  # size(B, H, W, 1)
        target_size_h = (target_bbox[..., 3] - target_bbox[..., 1]).unsqueeze(-1)  # size(B, H, W, 1)
        target_size_w = torch.clamp(target_size_w, min=1e-6)
        target_size_h = torch.clamp(target_size_h, min=1e-6)
        gaussian_score_target = self.gauss_2d(coordinate_out[..., 0::2], coordinate_out[..., 1::2],
                                              coordinate_target[..., 0::2], coordinate_target[..., 1::2],
                                              delta_ratio * target_size_w, delta_ratio * target_size_h)
        return gaussian_score_target

    def forward(self, output_maps, corners_list_512):
        sample_area_target, coordinate_target = detection_target(output_maps, corners_list_512, stage_lvl=4)
        # print(sample_area_target.device, coordinate_target.device)
        # ===================== generate loss targets ====================
        gauss_target = self.gen_loss_target(output_maps, coordinate_target, delta_ratio=0.3)
        # print(gauss_target.device, distance_target.device, size_target.device, discrete_target.device)
        # ===================== calculate loss part1 ====================
        gauss_out = output_maps[..., :4]
        coordinate_out = output_maps[..., 4:]
        # 【1】gauss loss
        loss_gauss = torch.sum(torch.ones_like(gauss_target).float() - gauss_target, dim=-1) / 4
        """
        # 【2】distance
        loss_distance = distance_target
        # alpha * V
        loss_v = size_target
        alpha = loss_v / (loss_gauss + loss_v)
        loss_size = alpha * loss_v
        # beta * D
        loss_d = discrete_target
        beta = loss_d / (loss_gauss + loss_d)
        loss_discrete = beta * loss_d
        """
        # ===================== calculate loss part2 ====================
        sample_focal_loss = self.focal_for_gauss(sample_area_target, gauss_out, gauss_target)
        # ===================== total loss =======================
        coord_loss = loss_gauss  # + loss_distance + loss_size + loss_discrete
        coord_loss = torch.sum(coord_loss[sample_area_target == 1]) / torch.sum(sample_area_target == 1)
        score_loss = sample_focal_loss
        detection_loss = score_loss + coord_loss
        # print(detection_loss, coord_loss, score_loss)
        return detection_loss, coord_loss, score_loss


# ------------------------------------------------------------------
# --------------- smooth-L1 loss with focal loss -------------------
# ------------------------------------------------------------------
class SmoothL1Loss(nn.Module):
    def __init__(self):
        super(SmoothL1Loss, self).__init__()

    @staticmethod
    def focal_for_gauss(sample_target, gauss_out, gauss_target, gamma=2, alpha=0.25, delta=1e-6):
        """
        :param sample_target: positive and negative sample, size(B, H, W)
        :param gauss_out: score, size(B, H, W, 4)
        :param gauss_target: score, size(B, H, W, 4)
        :param gamma for focal loss
        :param alpha for focal loss
        :param delta for log
        :return:
        """
        # positive
        difference_score = torch.abs(gauss_out - gauss_target)  # size(B, H, W, 4)
        y_pos = torch.clamp(difference_score, max=1.0 - delta)
        pos_focal_loss = -alpha * y_pos ** gamma * torch.log(torch.ones_like(y_pos).float() - y_pos)
        pos_focal_loss = torch.sum(pos_focal_loss[sample_target == 1]) / torch.sum(sample_target == 1)
        # print(pos_focal_loss)
        # negative
        y_neg = torch.clamp(gauss_out, max=1.0 - delta)
        neg_focal_loss = -(1.0 - alpha) * y_neg ** gamma * torch.log(torch.ones_like(y_neg).float() - y_neg)
        neg_focal_loss = torch.sum(neg_focal_loss[sample_target == 0]) / torch.sum(sample_target == 0)
        # total focal loss
        return pos_focal_loss + neg_focal_loss

    def gen_loss_target(self, output_maps, coordinate_target, delta_ratio=0.2):
        """
        :param output_maps: net output size(B, H, W, C) have transform to real coordinate (0-512) NOTICE !!!!!
        :param coordinate_target: from fun: detection_target, size(B, H, W, 8)
        :param delta_ratio: trans w / h to delta in 2D-gauss
        :return:
        """
        # device = output_maps.device
        gauss_out = output_maps[..., :4]
        coordinate_out = output_maps[..., 4:]
        """
        # gaussian score
        gaussian_score = torch.zeros((B, H, W, 4)).float()
        # distance target maps, ratio of distance
        distance_target = torch.zeros((B, H, W)).float()
        # size target maps, ratio of w / h, arctan
        size_target = torch.zeros((B, H, W)).float()
        # Discrete degree target maps, Mean square error
        discrete_target = torch.zeros((B, H, W)).float()
        """
        # ====================【1】Gaussian Score ======================
        target_bbox = gtP.corner2bboxHW(coordinate_target)
        target_size_w = (target_bbox[..., 2] - target_bbox[..., 0]).unsqueeze(-1)  # size(B, H, W, 1)
        target_size_h = (target_bbox[..., 3] - target_bbox[..., 1]).unsqueeze(-1)  # size(B, H, W, 1)
        target_size_w = torch.clamp(target_size_w, min=1e-6)
        target_size_h = torch.clamp(target_size_h, min=1e-6)
        gaussian_score_target = self.gauss_2d(coordinate_out[..., 0::2], coordinate_out[..., 1::2],
                                              coordinate_target[..., 0::2], coordinate_target[..., 1::2],
                                              delta_ratio * target_size_w, delta_ratio * target_size_h)
        return gaussian_score_target

    def forward(self, output_maps, corners_list_512):
        sample_area_target, coordinate_target = detection_target(output_maps, corners_list_512, stage_lvl=4)
        # print(sample_area_target.device, coordinate_target.device)
        # ===================== generate loss targets ====================
        gauss_target = self.gen_loss_target(output_maps, coordinate_target, delta_ratio=0.3)
        # print(gauss_target.device, distance_target.device, size_target.device, discrete_target.device)
        # ===================== calculate loss part1 ====================
        gauss_out = output_maps[..., :4]
        coordinate_out = output_maps[..., 4:]
        # 【1】gauss loss
        loss_gauss = torch.sum(torch.ones_like(gauss_target).float() - gauss_target, dim=-1) / 4
        # reduce=False: keep dims of output just as the input
        smooth_L1_map = F.smooth_l1_loss(coordinate_out, coordinate_target, reduction='none')
        """
        # 【2】distance
        loss_distance = distance_target
        # alpha * V
        loss_v = size_target
        alpha = loss_v / (loss_gauss + loss_v)
        loss_size = alpha * loss_v
        # beta * D
        loss_d = discrete_target
        beta = loss_d / (loss_gauss + loss_d)
        loss_discrete = beta * loss_d
        """
        # ===================== calculate loss part2 ====================
        sample_focal_loss = self.focal_for_gauss(sample_area_target, gauss_out, gauss_target)
        # ===================== total loss =======================
        coord_loss = smooth_L1_map
        coord_loss = torch.sum(coord_loss[sample_area_target == 1]) / torch.sum(sample_area_target == 1)
        score_loss = sample_focal_loss
        detection_loss = score_loss + coord_loss
        # print(detection_loss, coord_loss, score_loss)
        return detection_loss, coord_loss, score_loss


if __name__ == '__main__':
    corners = torch.tensor([[100.0, 100.0, 170.0, 120.0, 170.0, 135.0, 100.0, 115.0]])
    distr_idx = gtP.scale_distribute(corners, splitValue=tra_cfg.K_Means_args['split_value'])
    distr_idx = distr_idx[0]
    print("idx", distr_idx)
    corners = corners / 16
    print('corner / 16', corners)
    e, i = get_spatial_idx(corners, 32, 32, distr_idx, corners.device)
    with open('1.txt', 'w') as f:
        for y in range(e.shape[0]):
            for x in range(e.shape[1]):
                f.write(str(e[y][x].item()))
                f.write(' ')
            f.write('\n')
        f.write('\n=======================\n')
        for y in range(i.shape[0]):
            for x in range(i.shape[1]):
                f.write(str(i[y][x].item()))
                f.write(' ')
            f.write('\n')
    print(e[e == 1].size())
    """
    outmap = torch.rand((1, 32, 32, 12))
    corners = [torch.tensor([[100.0, 100.0, 130.0, 105.0, 130.0, 115.0, 100.0, 105.0],
                             [135.0, 118.0, 200.0, 120.0, 200.0, 135.0, 135.0, 135.0]])]
    GDLoss = GaussDistanceLoss()
    out = GDLoss(outmap, corners)
    for i in range(len(out)):
        print(i, out[i])
    """