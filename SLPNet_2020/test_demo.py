import os
import cv2
import numpy as np
import time
import torch
from torch.autograd import Variable
import module.det_part.GTProcessing as gtP
import module.det_part.PostProcessing as postP
from SSNet_framework import OtherDetectionNet
from argparse import ArgumentParser
from torchvision import transforms
from model.detection_recognition_pipeline import DetectionRecognitionPipeline
from load_data import CHARS, CHARS_DICT


networks = {
    '0': 'SSNet',
    '1': 'ShuffleNetV2',
    '2': 'MobileNetV3',
}

EXTENSIONS = ('.jpg', '.png')

# 读取图像，解决imread不能读取中文路径的问题
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


def default_image_loader(img_path, img_name, input_size):
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


def test_detection(args):
    # =================== create save dir ====================
    savedir = os.path.join(args.save_parent_folder, str(args.savedir))
    print("The save file path is: " + savedir)
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    # ==================== define model ===================
    if 0 == args.model:
        # model = SSDetectionNet()
        pass
    elif 1 == args.model:
        # model = OtherDetectionNet(model='shufflenet', mode='large')
        model = DetectionRecognitionPipeline()
    else:
        raise ValueError("No network matches the args.model!")
    if args.cuda:
        model = model.cuda()
    model = model.eval()
    # ==================== load the weight =====================
    assert args.pretrained is not None
    print("Load weight from pretrained model ...")
    pretrained_dict = torch.load(args.pretrained)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # print(pretrained_dict)
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print("=> Load weight successfully.")
    test_name_list = read_filenames(args.testdir, EXTENSIONS)
    # print(test_name_list)
    if 0 == args.mode:  # single image
        for index, img_name in enumerate(test_name_list):
            stop_flag = False
            print('index: %d -- %s' % (index, img_name))
            img_tensor = default_image_loader(args.testdir, img_name, (1024, 1024)).unsqueeze(0)
            t0 = time.time()
            if args.cuda:
                img_tensor = img_tensor.cuda()

            image = Variable(img_tensor, requires_grad=False)
            # x0, x1, outputs = model(image)
            obj_num_list, scores_tensor, coordinates_tensor, predict_sequence = model(image, mode1='det_reg', mode2='eval')
            t1 = time.time()
            print("Time: %.4f /img" % (t1 - t0))
            print(obj_num_list, scores_tensor, coordinates_tensor, predict_sequence)
            start_idx_pred = 0
            for batch_idx, obj_num_pred in enumerate(obj_num_list):
                if obj_num_pred != 0:
                    # tensor size(obj_num_pred, 8)
                    single_img_coord_preds = coordinates_tensor[start_idx_pred: start_idx_pred + obj_num_pred]
                    single_img_scores = scores_tensor[start_idx_pred: start_idx_pred + obj_num_pred]
                    # print(single_img_coord_preds, single_img_scores)
                    for lp_list in predict_sequence:
                        # print(lp_list)
                        lp_char = ''
                        for c in lp_list:
                            # print(CHARS)
                            lp_char += CHARS[c]
                        print("The predict lp: %s" % lp_char)
                    stop_flag = postP.det_show(os.path.join(args.testdir, img_name),
                                               savedir, single_img_coord_preds, single_img_scores, None)
                    start_idx_pred = start_idx_pred + obj_num_pred
            if stop_flag:
                break

    else:
        pass


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cuda', action='store_true', default=False)
    # model: 0 means SSNet, otherwise means other nets; all these according to a list
    parser.add_argument('--model', type=int, default=1)
    # mode: (0, 1), 0 means one image once, with imshow; 1 used to calculate mAP, now only mode 0 is accepted
    parser.add_argument('--mode', type=int, default=0)
    parser.add_argument('--testdir', default="./data/test")
    parser.add_argument('--pretrained', default="./weight/weight3_8/model_best.pth")  # ../Pretrained/rgcnet/model_best.pth"
    # used to save output image
    parser.add_argument('--save_parent_folder', default='ImgSave')
    parser.add_argument('--savedir', default="SLPNet")
    test_detection(parser.parse_args())
