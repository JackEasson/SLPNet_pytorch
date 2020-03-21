from model.SSNet_modules import SSNetDet, SSNetRegOriginal
from utils.PostProcessing import clac_gauss_score_multi
import utils.decoder as decoder
from model.basic_modules import *
from utils.perspective_transform import PerspectiveTrans


class DetectionRecognitionPipeline(nn.Module):
    def __init__(self, input_size=(1024, 1024), det_size=(512, 512), reg_size=(144, 48), class_num=68):
        super(DetectionRecognitionPipeline, self).__init__()
        self.input_size = input_size
        self.det_size = det_size
        self.reg_size = reg_size
        self.model_det = SSNetDet(input_size=512)
        self.model_reg = SSNetRegOriginal(class_num=class_num)
        self.perspective_trans_function = PerspectiveTrans()

    def forward(self, x, mode1='det_reg', mode2='eval'):
        assert mode1 in ('det_only', 'det_reg') and mode2 in ('train', 'eval')
        device = x.device
        if mode1 == 'det_only':
            # x's size is respected to be 1024 * 1024
            x_det = F.interpolate(x, self.det_size, mode='bilinear', align_corners=True)
            # output size: x1 (64 * 64), x2 (32 * 32)
            x1, x2, x_pred = self.model_det(x_det)
            if mode2 == 'train':
                return x_pred  # size(B, 12, 32, 32)

            else:  # mode2 == 'eval', detection decoder: Top_K and NMS
                obj_num_list, scores_tensor, coordinates_tensor = decoder.detection_decoder(x_pred)
                return  obj_num_list, scores_tensor, coordinates_tensor

        else:  # mode1 == 'det_reg'
            x_det = F.interpolate(x, self.det_size, mode='bilinear', align_corners=True)
            # output size: x1 (64 * 64), x2 (32 * 32)
            x1, x2, x_pred = self.model_det(x_det)
            obj_num_list, scores_tensor, coordinates_tensor = decoder.detection_decoder(x_pred)
            # print("!!!")

            # 获取识别所需的输入图与共享特征图
            w_ratio1, h_ratio1 = (self.input_size[0] * 1.0 / self.det_size[0],
                                  self.input_size[1] * 1.0 / self.det_size[1])
            lp_img_tensor_list = []
            start_idx = 0
            for batch_idx in range(len(obj_num_list)):  # each image in batch
                obj_num = obj_num_list[batch_idx]
                if obj_num != 0:
                    single_img_corners = coordinates_tensor[start_idx: start_idx + obj_num]
                    start_idx = start_idx + obj_num
                    lp_corners = (single_img_corners.reshape(-1, 4, 2) * torch.tensor([w_ratio1, h_ratio1]).to(device)).\
                        reshape(-1, 8)
                    lp_img_tensor_list.extend(self.perspective_trans_function(fea_maps=x[batch_idx],
                                                                              corners_tensor=lp_corners,
                                                                              target_size=self.reg_size))
            reg_batch = len(lp_img_tensor_list)
            if reg_batch != 0:  # 有lp图，可以进行识别
                lp_imgs_tensor = torch.cat(lp_img_tensor_list, dim=0)
                logits = self.model_reg(lp_imgs_tensor)
                if mode2 == 'train':
                    return obj_num_list, scores_tensor, coordinates_tensor, x_pred, logits
                else:  # 'eval'
                    # 识别部分解码
                    predict_sequence = decoder.greedy_decoder(logits)  # 预测的标签序列，0-67, list(reg_batch)
                    return obj_num_list, scores_tensor, coordinates_tensor, predict_sequence
            else:  # 无lp图
                if mode2 == 'train':
                    return obj_num_list, scores_tensor, coordinates_tensor, x_pred, None
                else:  # mode2 == 'eval'
                    return obj_num_list, scores_tensor, coordinates_tensor, None


def online_distribute_ctc_targets(obj_num_list, coordinates_tensor, coord_labels, lp_labels, length_labels, gauss_threshold=0.3):
    """
    :param obj_num_list: list(B), each image in batch, how many lps have detected.
    :param coordinates_tensor: tensor size(lps_num, 8)
    :param coord_labels: list(B), each element a tensor, size(obj_num, 8), From DataLoader
    :param lp_labels: list(B), each element is list(N), and each element is tensor int32, From DataLoader, such as
                      [[tensor, tensor], [tensor], [tensor, tensor]]
    :param length_labels: corresponding to each lp's length, list(B), each element is list(N), and each element is
                          a number int32, From DataLoader, such as [[7, 7], [7], [8, 7]]
    :param gauss_threshold: if gauss score less than it, it will be recognized as [all blank]
    :return:
    """
    device = coordinates_tensor.device
    start_idx_pred = 0
    lp_labels_clean = []
    length_labels_clean = []
    keep_pred_list = []
    for batch_idx, obj_num_pred in enumerate(obj_num_list):
        if obj_num_pred != 0:
            # tensor size(obj_num_pred, 8)
            single_img_coord_preds = coordinates_tensor[start_idx_pred: start_idx_pred + obj_num_pred]
            # print('single_img_coord_preds', single_img_coord_preds)
            start_idx_pred = start_idx_pred + obj_num_pred
            single_img_coord_labels = coord_labels[batch_idx]
            # 真实lp个数
            single_img_length_label_list = length_labels[batch_idx]  # a list, list(obj_true), each element a number
            single_img_lp_labels_list = lp_labels[batch_idx]  # a list, list(obj_true), each element tensor(7) or tensor(8)
            match_tensor = torch.zeros(obj_num_pred, dtype=torch.long).to(device) - 1  # 存放对应label的序号, initial -1
            for obj_idx in range(obj_num_pred):  # 对每个图的每个检测的lp对应label进行匹配
                match_scores = clac_gauss_score_multi(single_img_coord_preds[obj_idx], single_img_coord_labels,
                                                      delta_ratio=0.2)  # out: tensor size(obj_num_label)
                # print('match_scores', match_scores)
                _, order = match_scores.sort(dim=0, descending=True)
                if match_scores[order[0]] > gauss_threshold:
                    match_tensor[obj_idx] = order[0]
            keep_pred_list.append(match_tensor != -1)  # [tensor, tensor, ...], each tensor is bool type
            match_tensor_clean = match_tensor[match_tensor != -1]
            lp_labels_clean.extend([single_img_lp_labels_list[i.item()] for i in match_tensor_clean])
            length_labels_clean.extend([single_img_length_label_list[i.item()] for i in match_tensor_clean])
    # print('lp_labels_clean', lp_labels_clean, length_labels_clean)
    keep_pred_tensor = torch.cat(keep_pred_list, dim=0)
    if lp_labels_clean:  # not empty
        lp_labels_clean = torch.cat(lp_labels_clean, dim=0)
    # 上面lp_labels_clean == [], 则一定有length_labels_clean == []
    else:
        lp_labels_clean = None
        length_labels_clean = None
    return keep_pred_list, keep_pred_tensor, lp_labels_clean, length_labels_clean


if __name__ == '__main__':

    coord_label = [torch.tensor([[0.0, 0.0, 100.0, 0.0, 100.0, 50.0, 0.0, 50.0]]),
                   torch.tensor([[0.0, 0.0, 20.0, 5.0, 25.0, 25.0, 1.0, 20.0],
                                [0.0, 0.0, 200.0, 10.0, 220.0, 160.0, 10.0, 160.0]])]
    coordinates_tensor = torch.tensor([[0.0, 0.0, 20.0, 5.0, 25.0, 25.0, 1.0, 20.0],
                                       [0.0, 0.0, 150.0, 20.0, 200.0, 160.0, 10.0, 160.0]])
    obj_num_list = [0, 2]
    lp_labels = [[torch.tensor([0, 1, 2, 3, 4, 5, 6])],
                 [torch.tensor([0, 0, 0, 0, 0, 0, 0]), torch.tensor([2, 1, 2, 1, 2, 1, 2, 1])]]
    length_labels = [[7], [7, 8]]
    output = online_distribute_ctc_targets(obj_num_list, coordinates_tensor, coord_label, lp_labels, length_labels,
                                           gauss_threshold=0.6)
    for i in output:
        print(i)






