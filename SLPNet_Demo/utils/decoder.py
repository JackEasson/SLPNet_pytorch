import torch
import numpy as np
from utils.PostProcessing import detection_analysis, nms_gauss


CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'I', 'O', '-'
         ]


# 对一批进行操作
def detection_decoder(predict_maps, top_k=50, value_threshold=0.6, nms_threshold=0.2):
    """
    :param predict_maps: outputs from detection network, size(B, 32, 32, 12)
    :param top_k:
    :param value_threshold:
    :param nms_threshold:
    :return:
    """
    # NMS to get detection results
    outputs_list = detection_analysis(predict_maps, top_k=top_k, threshold=value_threshold)  # 就算没有检测结果，输出列表中仍有空张量！
    coordinates_list = []
    scores_list = []
    obj_num_list = []  # each image in batch contain how many objects
    for batch_idx in range(len(outputs_list)):
        if outputs_list[batch_idx]['score'].shape == torch.Size([0]):
            obj_num_list.append(0)
        else:
            keep_idx = nms_gauss(outputs_list[batch_idx]['coord'], outputs_list[batch_idx]['score'],
                                 threshold=nms_threshold, delta_ratio=0.2)
            # print(keep_idx)
            # print(outputs_list[0]['coord'][keep_idx], outputs_list[0]['score'][keep_idx])
            out_corners = outputs_list[batch_idx]['coord'][keep_idx]
            coordinates_list.append(out_corners)
            # print(out_corners.shape)
            out_scores = outputs_list[batch_idx]['score'][keep_idx]
            scores_list.append(out_scores)
            obj_num_list.append(out_scores.shape[0])
    if sum(obj_num_list) == 0:  # all empty
        coordinates_tensor = None
        scores_tensor = None
    else:
        coordinates_tensor = torch.cat(coordinates_list, dim=0)
        scores_tensor = torch.cat(scores_list, dim=0)
    return obj_num_list, scores_tensor, coordinates_tensor


# recognition decoder
def greedy_decoder(prebs):
    """
    :param prebs: outputs from lp recognition network
    :return:
    """
    # greedy decode
    prebs = prebs.cpu().detach().numpy()
    preb_labels = list()
    for i in range(prebs.shape[0]):
        preb = prebs[i, :, :]
        preb_label = list()
        for j in range(preb.shape[1]):
            preb_label.append(np.argmax(preb[:, j], axis=0))
        no_repeat_blank_label = list()
        pre_c = preb_label[0]
        if pre_c != len(CHARS) - 1:
            no_repeat_blank_label.append(pre_c)
        for c in preb_label:  # dropout repeate label and blank label
            if (pre_c == c) or (c == len(CHARS) - 1):
                if c == len(CHARS) - 1:
                    pre_c = c
                continue
            no_repeat_blank_label.append(c)
            pre_c = c
        preb_labels.append(no_repeat_blank_label)
    return preb_labels

