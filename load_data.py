# coding=utf-8
import cv2
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import train_config as cfg

# ==========================【1】Global Variable ===========================
CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'I', 'O', '-'
         ]

CHARS_DICT = {char: i for i, char in enumerate(CHARS)}


EXTENSIONS = ('.jpg', '.png')


# =========================== some transform function ===========================
def trans_lp_str2label(lp_str):
    """
    :param lp_str: like '皖A1J877'
    :return:
    """
    return [CHARS_DICT[c] for c in lp_str]


def trans2cuda_list(tensor_list):
    return [t.cuda() for t in tensor_list]


def trans2requires_grad_list(tensor_list):
    return [t.requires_grad_() for t in tensor_list]


# ============================= about dataset =================================
# operate Chinese path problem
def cv_imread(filepath):
    cv_img = cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), -1)
    return cv_img


def read_filenames(file_path, extension):
    assert type(extension) in (list, tuple)
    names_list = []
    for parent, dirnames, filenames in os.walk(file_path):
        for filename in filenames:
            # 后缀判断
            (name, exten) = os.path.splitext(filename)
            if exten in extension:
                names_list.append(filename)  # 得到有后缀的名字列表
    return names_list


# transform the suffix
def trans_extension(filename, target_extension):
    """
    :param filename: just a name without extension, not full path
    :param target_extension:
    :return:
    """
    assert type(target_extension) is str
    none_extension_filename = os.path.splitext(filename)[0]
    target_filename = none_extension_filename + target_extension
    return target_filename


def get_image_wh(image_name):
    img = cv_imread(image_name)
    return img.shape[0], img.shape[1]  # image's h and w


def base_image_loader(img_path, img_name, input_size):
    """
    返回image的tensor
    """
    width, height = input_size
    full_img_path = os.path.join(img_path, img_name)
    # print(full_img_path)
    inputs = cv_imread(full_img_path)
    # print(inputs.shape)  # (H, W, C)
    # 800*800
    inputs = cv2.resize(inputs, (width, height))
    inputs = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB).astype(np.float32)  # bgr to rgb
    inputs = (inputs / 255.0) * 2.0 - 1.0  # to -1.0 ~ 1.0
    img_tensor = transforms.ToTensor()(inputs)  # (H, W, C) -> (C, H, W), scale also -1.0 ~ 1.0
    return img_tensor


def read_txt_info(img_name, txt_parent_path):
    """
    :param img_name: one name, not full path, with extension
    :param txt_parent_path
    :return: LP label and polygon coordinate (tl, tr, br, bl), type: float; image w, h (int type)
    """
    txt_name = trans_extension(img_name, '.txt')
    txt_path = os.path.join(txt_parent_path, txt_name)
    cornerpoints = []
    lpchar_list = []
    with open(txt_path, "r", encoding='utf-8') as f_txt:
        lines = f_txt.readlines()  # 读取全部内容 ，并以列表方式返回
        for line in lines:
            # 循环处理一个obj的4组坐标(8个)
            # print(line)
            line = line.strip()
            split_infos = line.split(',')
            lp_str = split_infos[-1]
            lpchar_list.append(lp_str)
            coord_list = split_infos[:4]
            coord_list = [c.split(' ') for c in coord_list]
            coord_list = [float(v) for c in coord_list for v in c]
            cornerpoints.append(coord_list)
    single_info = {'Name': img_name,
                   'Points': cornerpoints,
                   'Lpchar': lpchar_list}
    return single_info


class LPDataSet(Dataset):
    def __init__(self, img_path, txt_path, image_loader=base_image_loader, label_loader=read_txt_info):
        super(LPDataSet, self).__init__()  # 调用父类初始化
        self.img_path = img_path
        self.txt_path = txt_path
        self.image_loader = image_loader
        self.label_loader = label_loader
        self.img_names = read_filenames(img_path, EXTENSIONS)
        self.gt_labels = self._data_init()

    def _data_init(self):
        gt_labels = []
        img_num = len(self.img_names)
        for i in range(img_num):
            gt_labels.append(self.label_loader(self.img_names[i], self.txt_path))
        return gt_labels

    def __getitem__(self, index):
        """
        :param index:
        :return: img_tensor size(3, 512, 512); corners_tensor size(N, 8);
                 lp_label_tensor list(obj_num), each size(7 or 8);
                 lp_label_length list(obj_num)
        """
        img_name = self.gt_labels[index]['Name']
        # print("name", img_name)
        global global_name
        global_name = img_name
        # 读取、预处理、生成tensor
        img_tensor = self.image_loader(self.img_path, img_name, cfg.INPUT_SIZE)  # size(1024, 1024)
        # print("coord", coord_labl)
        # 输出也是tensor
        points_list = self.gt_labels[index]['Points']
        # print("coord", lp_label)
        image_h, image_w = get_image_wh(os.path.join(self.img_path, img_name))
        size_tensor = torch.tensor([image_w, image_h]).float()
        corners_tensor = torch.tensor(points_list).reshape(-1, 4, 2) / size_tensor * \
                         torch.tensor(cfg.DETECTION_SIZE).float()
        corners_tensor = corners_tensor.reshape(-1, 8)
        # like [tensor([12, 41, 32, 49, 39, 38, 38], dtype=torch.int32),
        #       tensor([12, 41, 32, 49, 39, 38, 38, 40, 39], dtype=torch.int32)]
        lp_label_list = [torch.tensor(trans_lp_str2label(lp_str), dtype=torch.int32) for lp_str in self.gt_labels[index]['Lpchar']]
        lp_length_list = [len(lp_str) for lp_str in self.gt_labels[index]['Lpchar']]
        return img_tensor, corners_tensor, lp_label_list, lp_length_list, [img_name]

    def __len__(self):
        return len(self.gt_labels)


# and we set the collate method
def base_lp_collate(batch):
    img_data = [item[0] for item in batch]  # 输出list(batch), element size(3, img_size, img_size)
    # 将batch个3维list扩成一个4维tensor
    img_data = torch.stack(img_data, dim=0)
    point_data = [item[1] for item in batch]
    label_data = [item[2] for item in batch]
    length_data = [item[3] for item in batch]
    img_name = [item[4] for item in batch]
    # return size
    # img_data size(B, 3, 512, 512)
    # point_data list(B), each element size(obj_num, 8), if only one obj, size(1, 8)
    # label_data list(B) -> list(obj_num) -> tensor size(7 or 8)
    # length_data list(B) -> list(obj_num) -> 7 or 8
    return [img_data, point_data, label_data, length_data, img_name]


if __name__ == '__main__':
    lp_dataset = LPDataSet(cfg.val_img_folder_path, cfg.val_txt_folder_path)
    print(lp_dataset.__len__())
    trainloader = DataLoader(lp_dataset, batch_size=2, num_workers=4, shuffle=True,
                             drop_last=True, collate_fn=base_lp_collate)  # , collate_fn=train_data.my_collate
    for i, (img_tensor, corners_tensor, lp_label_list, lp_length_list, name_list) in enumerate(trainloader):
        print(img_tensor.shape)
        print(corners_tensor)
        print('!', lp_label_list)
        print(lp_length_list)
        print(name_list)
        break