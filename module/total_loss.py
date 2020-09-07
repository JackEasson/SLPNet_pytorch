import torch
import torch.nn as nn
from load_data import CHARS
from module.det_part.detection_head import GaussDistanceLoss


class TotalLoss(nn.Module):
    def __init__(self):
        super(TotalLoss, self).__init__()
        self.GDLoss = GaussDistanceLoss()  # 'mean'
        self.CTCLoss = nn.CTCLoss(blank=len(CHARS)-1, reduction='mean')

    def forward(self, det_maps, det_labels, log_probs, reg_labels, input_lengths, target_lengths, mode='det_reg'):
        assert mode in ('det_only', 'det_reg')
        device = det_maps.device()
        if mode == 'det_only':
            # det_loss = score_loss + coord_loss
            det_loss, coord_loss, score_loss = self.GDLoss(output_maps=det_maps, corners_list_512=det_labels)
            return det_loss, coord_loss, score_loss
        else:  # mode == 'det_reg'
            # 识别部分要先根据输出情况分有lp、无lp进行处理，无则都是torch.tensor(-1)，只有det_loss
            det_loss, coord_loss, score_loss = self.GDLoss(output_maps=det_maps, corners_list_512=det_labels)
            if log_probs == torch.tensor(-1).to(device):
                total_loss = det_loss
                reg_loss = torch.tensor(0.).to(device)
                return total_loss, det_loss, reg_loss, coord_loss, score_loss
            else:  # 有lp识别结果
                reg_loss = self.CTCLoss(log_probs, reg_labels, input_lengths=input_lengths,
                                        target_lengths=target_lengths)
                total_loss = det_loss + reg_loss
                return total_loss, det_loss, reg_loss, coord_loss, score_loss


