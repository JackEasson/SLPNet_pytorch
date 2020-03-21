import torch
import utils.GTProcessing as gtP
import cv2
import numpy as np

SAVE_COUNT = 0

# 读取图像，解决imread不能读取中文路径的问题
def cv_imread(filepath):
    cv_img = cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), -1)
    return cv_img


def get_image_wh(image_name):
    img = cv_imread(image_name)  # 解决中文路径问题
    return img.shape[0], img.shape[1]  # image's h and w


def detection_analysis(outputs, top_k=50, threshold=0.5, out_size=512.0):
    """
    :param outputs: from detection network, size(B, H, W, C)
    :param top_k: select top k corners
    :param threshold:
    :param out_size: detection outputs size, to limit bound (0~512)
    :return:
    """
    outputs_list = []
    outputs = outputs.data
    B = outputs.shape[0]
    gauss_scores = (torch.sum(outputs[..., :4], dim=-1) / 4).reshape(B, -1)  # size(B, H*W)
    coord_outputs = outputs[..., 4:].reshape(B, -1, 8)  # size(B, H*W, 8)
    for i in range(B):
        _, order = gauss_scores[i].sort(dim=0, descending=True)  # order size(B, H*W)
        gauss_score = gauss_scores[i, order[:top_k]]  # size(top_k)
        coord_output = coord_outputs[i, order[:top_k]]  # size(top_k, 8)
        # 下面保证所得的四边形外接矩形长宽不为0
        bbox_output = gtP.corner2bboxHW(coord_output)  # size(top_k, 4)
        w_output = bbox_output[..., 2] - bbox_output[..., 0]  # size(top_k)
        h_output = bbox_output[..., 3] - bbox_output[..., 1]  # size(top_k)
        match_w = (w_output > 0)
        match_h = (h_output > 0)
        nonzero_match = (match_w * match_h).nonzero().squeeze()
        gauss_score = gauss_score[nonzero_match]
        coord_output = coord_output[nonzero_match]
        idx = (gauss_score >= threshold).nonzero().squeeze()
        gauss_score = gauss_score[idx]
        coord_output = coord_output[idx]
        coord_output = coord_output.clamp(min=0.0, max=out_size)
        if len(coord_output.shape) != 2:
            gauss_score = gauss_score.unsqueeze(0)
            coord_output = coord_output.unsqueeze(0)
        outputs_list.append({
            'score': gauss_score,  # tensor size(obj)
            'coord': coord_output  # tensor size(obj, 8)
        })
    return outputs_list


# return size(..., 4)
def gauss_2d(x1, x2, u1, u2, d1, d2):
    m = torch.pow((x1 - u1) / d1, 2)
    n = torch.pow((x2 - u2) / d2, 2)
    return torch.exp(-0.5 * (m + n))


# corners[N,8]，scores维度为[N], 均为tensor
def nms_gauss(corners, scores, threshold=0.2, delta_ratio=0.2):
    # 降序排列，order下标排序
    _, order = scores.sort(0, descending=True)
    # print(_)
    keep = []
    while order.numel() > 0:  # torch.numel()返回张量元素个数
        if order.numel() == 1:  # 保留框只剩一个
            i = order.item()
            keep.append(i)
            break
        else:
            i = order[0].item()  # 保留scores最大的那个框corner[i]
            keep.append(i)
        # 计算corners[i]与其余各框的Gauss Score
        target_bbox = gtP.corner2bboxSingle(corners[i])  # size(4)
        target_size_w = (target_bbox[2] - target_bbox[0]).unsqueeze(-1)  # size(B, H, W, 1)
        target_size_h = (target_bbox[3] - target_bbox[1]).unsqueeze(-1)  # size(B, H, W, 1)
        target_size_w = torch.clamp(target_size_w, min=1e-6)
        target_size_h = torch.clamp(target_size_h, min=1e-6)
        gaussian_score = gauss_2d(corners[order[1:], 0::2], corners[order[1:], 1::2],
                                  corners[order[0], 0::2], corners[order[0], 1::2],
                                  delta_ratio * target_size_w, delta_ratio * target_size_h)  # size(N-1, 4)
        gaussian_score = torch.sum(gaussian_score, dim=-1) / 4
        # print(gaussian_score)
        idx = (gaussian_score <= threshold).nonzero().squeeze()  # 注意此时idx为[N-1,] 而order为[N,]
        if idx.numel() == 0:
            break
        order = order[idx + 1]  # 修补索引之间的差值
    return torch.LongTensor(keep)  # Pytorch的索引值为LongTensor


def det_show(img_path, save_path, out_corner, out_score, label_corner, out_scale=(512, 512)):
    """
    :param img_path: full image path for evaluation
    :param save_path: to save the eval image
    :param out_corner: corner after nms of outputs
    :param out_score:
    :param label_corner:
    :param out_scale:
    :return:
    """
    global SAVE_COUNT
    img = cv_imread(img_path)
    img1 = img.copy()
    img_h, img_w = img1.shape[:2]
    # print(img_h, img_w)
    ratio_h = img_h / out_scale[1]
    ratio_w = img_w / out_scale[0]
    ratio_size = torch.tensor([ratio_w, ratio_h]).float()
    # print(ratio_size)
    img2 = img1.copy()
    point_size = 1
    point_color1 = (0, 255, 0)  # BGR
    point_color2 = (0, 0, 255)  # BGR
    thickness = 6  # 可以为 0 、4、8
    # part1 out
    obj_num = out_corner.shape[0]
    for obj in range(obj_num):
        points = out_corner[obj].reshape(4, 2)
        points = points * ratio_size
        for i, point in enumerate(points):
            if i == 0:
                # print(out_score[obj].item(), (int(point[0].item()), int(point[1].item())))
                cv2.putText(img1, "GaussScore: %.4f" % out_score[obj].item(), (int(point[0].item()), int(point[1].item())-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            cv2.circle(img1, (int(point[0].item()), int(point[1].item())), point_size, point_color1, thickness)
        # lines
        for i in range(3):
            cv2.line(img1, (int(points[i][0].item()), int(points[i][1].item())),
                     (int(points[i+1][0].item()), int(points[i+1][1].item())), (0, 255, 0), 2, cv2.LINE_AA)
        cv2.line(img1, (int(points[3][0].item()), int(points[3][1].item())),
                 (int(points[0][0].item()), int(points[0][1].item())), (0, 255, 0), 2, cv2.LINE_AA)
    """
    # part2 label
    obj_num = label_corner.shape[0]
    for obj in range(obj_num):
        points = label_corner[obj].reshape(4, 2)
        for point in points:
            point = point * ratio_size
            cv2.circle(img2, (int(point[0].item()), int(point[1].item())), point_size, point_color2, thickness)
    """

    cv2.imshow('output', img1)

    # show perspective image
    out_corner = out_corner.reshape(-1, 4, 2) * ratio_size
    corners_list = [x for x in out_corner]
    lp_img_list = wrap_perspective(img, corners_list, (144, 48))
    # cv2.imshow('label', img2)
    """
    for image in lp_img_list:
        cv2.imwrite(save_path + "/lp_%s.jpg" % SAVE_COUNT, image)
        SAVE_COUNT += 1
        cv2.waitKey(200)
    return False
    """
    cc = cv2.waitKey(0)
    if cc == 115:  # press 's' to save
        cv2.imwrite(save_path + "/save_%s.jpg" % SAVE_COUNT, img1)
        SAVE_COUNT += 1
        return False
    elif cc == 27:  # ESC to quit
        return True
    else:
        return False


# ====================== perspective transform =======================
def corner2bbox(corners_list):
    """
    :param corners_list: [tensor, tensor ...]
    :return: list of box, float, not tensor
    """
    bbox_list = []
    for corners in corners_list:
        corners = corners.view(-1)
        left = torch.min(corners[::2]).int().item()
        top = torch.min(corners[1::2]).int().item()
        right = torch.max(corners[::2]).int().item()
        bottom = torch.max(corners[1::2]).int().item()
        bbox_list.append([left, top, right, bottom])
    return bbox_list


def wrap_perspective(src_img, corners_list, out_size):
    """
    :param src_img: 用于wrap的原图
    :param corners_list: [tensor, tensor ...]，用于corner2bbox函数
    :param out_size: 生成的透视后图像尺寸  w*h ??
    :return: image after perspective
    """
    # print('corners_list', corners_list)
    bbox = corner2bbox(corners_list)
    # print('bbox', bbox)
    img_idx = 1
    img_list =[]
    for b, c in zip(bbox, corners_list):
        wrap_area = src_img[b[1]:b[3], b[0]:b[2]]
        # 透视变换
        # 原点位置
        srcpoints = (c.long() - torch.tensor([b[0], b[1]])).float().numpy()
        # 原点顺序 左上 左下 右下 右上
        srcpoints = np.array([srcpoints[0], srcpoints[3], srcpoints[2], srcpoints[1]])
        # print('srcpoints', srcpoints)
        # 变换后位置
        canvaspoints = np.float32([[0, 0], [0, out_size[1]], [out_size[0], out_size[1]], [out_size[0], 0]])
        # print('canvaspoints', canvaspoints)
        # 计算转换矩阵
        perspectiveMatrix = cv2.getPerspectiveTransform(np.array(srcpoints), np.array(canvaspoints))
        perspectiveImg = cv2.warpPerspective(wrap_area, perspectiveMatrix, out_size)
        img_list.append(perspectiveImg)
        cv2.imshow("%d" % img_idx, perspectiveImg)
        img_idx += 1
    return img_list


# ======================== 用于计算模型在测试集上的实际高斯总得分 ================================
def clac_gauss_score_eval(coordinate_out, coordinate_target, delta_ratio=0.2):
    """
    :param coordinate_out: size(8)
    :param coordinate_target: same as above
    :param delta_ratio
    :return:
    """
    target_bbox = gtP.corner2bboxSingle(coordinate_target)  # out size(4)
    target_size_w = (target_bbox[2] - target_bbox[0]).unsqueeze(-1)  # size(1)
    target_size_h = (target_bbox[3] - target_bbox[1]).unsqueeze(-1)  # size(1)
    target_size_w = torch.clamp(target_size_w, min=1e-6)
    target_size_h = torch.clamp(target_size_h, min=1e-6)
    gaussian_score4 = gauss_2d(coordinate_out[0::2], coordinate_out[1::2],
                               coordinate_target[0::2], coordinate_target[1::2],
                               delta_ratio * target_size_w, delta_ratio * target_size_h)  # size(4)
    gauss_score = (torch.sum(gaussian_score4) / 4)  # size(1)
    return gauss_score


def clac_gauss_score_multi(coordinate_out, coordinate_target, delta_ratio=0.2):
    """
    :param coordinate_out: size(8)
    :param coordinate_target: size(N, 8)
    :param delta_ratio
    :return:
    """
    target_bbox = gtP.corner2bboxMulti(coordinate_target)  # out size(N, 4)
    target_size_w = (target_bbox[:, 2] - target_bbox[:, 0]).unsqueeze(-1)  # size(N, 1)
    target_size_h = (target_bbox[:, 3] - target_bbox[:, 1]).unsqueeze(-1)  # size(N, 1)
    target_size_w = torch.clamp(target_size_w, min=1e-6)
    target_size_h = torch.clamp(target_size_h, min=1e-6)
    gaussian_score4 = gauss_2d(coordinate_out[0::2], coordinate_out[1::2],
                               coordinate_target[:, 0::2], coordinate_target[:, 1::2],
                               delta_ratio * target_size_w, delta_ratio * target_size_h)  # size(N, 4)
    gauss_score = (torch.sum(gaussian_score4, dim=-1) / 4)  # size(N)
    return gauss_score


def gaussian_eval(corner_preds, corner_targets, gauss_threshold=0.5):
    """
    :param corner_preds: from function: detection_analysis, tensor size(obj1, 8)
    :param corner_targets: tensor, size(obj2, 8)
    :param gauss_threshold
    :return:
    """
    Tp = 0
    Fp = 0
    Fn = 0
    gauss_list2return = list()
    len_pred = corner_preds.shape[0]
    len_target = corner_targets.shape[0]
    match_array_p = np.zeros(len_pred, dtype=np.byte)
    match_array_t = np.zeros(len_target, dtype=np.byte)
    for obj_p in range(len_pred):
        score_list = list()
        for obj_t in range(len_target):
            score_list.append(clac_gauss_score_eval(corner_preds[obj_p], corner_targets[obj_t]))
        score_tensor = torch.tensor(score_list)
        _, order = score_tensor.sort(dim=0, descending=True)
        match_target_idx = list(order.numpy()).index(0)
        score_value = score_tensor[match_target_idx].item()
        if score_value > gauss_threshold:  # 匹配成功
            match_array_t[match_target_idx] = 1
            match_array_p[obj_p] = 1
            Tp += 1
            gauss_list2return.append(score_value)
    for idx in match_array_p:
        if idx == 0:
            Fn += 1
    for idx in match_array_t:
        if idx == 0:
            Fp += 1
            gauss_list2return.append(0.0)
    return Tp, Fn, Fp, gauss_list2return


if __name__ == '__main__':

    corners_target = torch.tensor([[0.0, 0.0, 100.0, 10.0, 110.0, 60.0, 10.0, 50.0],
                                  [0.0, 0.0, 20.0, 5.0, 25.0, 25.0, 1.0, 20.0],
                                  [0.0, 0.0, 200.0, 10.0, 220.0, 160.0, 10.0, 160.0]])
    corners = torch.tensor([0.0, 0.0, 100.0, 10.0, 110.0, 60.0, 10.0, 50.0])
    score_tensor = clac_gauss_score_multi(corners, corners_target)
    print(score_tensor)
    _, order = score_tensor.sort(dim=0, descending=True)
    order = order.cuda()
    print(order[0])
    """
    corners = torch.tensor([[0.0, 0.0, 100.0, 10.0, 110.0, 60.0, 10.0, 50.0],
                            [0.0, 0.0, 80.0, 10.0, 80.0, 60.0, 10.0, 50.0],
                            [100.0, 100.0, 180.0, 100.0, 180.0, 160.0, 100.0, 150.0],
                            [100.0, 100.0, 160.0, 100.0, 160.0, 130.0, 100.0, 130.0]])

    scores = torch.tensor([0.9, 0.8, 0.75, 0.7])
    keep_idx = nms_gauss(corners, scores)
    print(keep_idx)
    print(corners[keep_idx], scores[keep_idx])    """
    """
    x = torch.rand((2, 4, 4, 12))
    y = detection_analysis(x, top_k=5)
    print(y)
    keep_idx = nms_gauss(y[0]['coord'], y[0]['score'])
    print(keep_idx)
    print(y[0]['coord'][keep_idx], y[0]['score'][keep_idx])"""