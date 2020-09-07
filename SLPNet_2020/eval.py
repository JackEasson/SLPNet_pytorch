import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import math
from argparse import ArgumentParser
from torch.optim import SGD, Adam
from torch.autograd import Variable
from load_data import *
from module.det_part.detection_head import GaussDistanceLoss
import module.det_part.PostProcessing as postP
import train_config as train_cfg
from model.detection_recognition_pipeline import DetectionRecognitionPipeline, online_distribute_ctc_targets


def sparse_tuple_for_ctc(T_length, lengths):
    input_lengths = []
    target_lengths = []

    for ch in lengths:
        input_lengths.append(T_length)
        target_lengths.append(ch)

    return tuple(input_lengths), tuple(target_lengths)


def eval(args, model):
    # ===============================【1】DataSet=================================
    # val dataset loader
    dataset_val = LPDataSet(img_path=train_cfg.val_img_folder_path, txt_path=train_cfg.val_txt_folder_path)
    print("=>Val dataset total images: % d" % dataset_val.__len__())
    loader_val = DataLoader(dataset_val, num_workers=args.num_workers, batch_size=args.batch_size,
                            drop_last=False, shuffle=False, collate_fn=base_lp_collate)
    if args.pretrained is not None:
        print("Load weight from pretrained model ...")
        pretrained_dict = torch.load(args.pretrained)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print("=> Load weight successfully.")
    else:
        print("Please input the model weight!")
        raise ValueError("The args pretrained shouldn't be None!")
    model.eval()
    if args.mode == 0:  # detection only
        print("The validation mode is: detection only.")
        Tp_all = 0
        Fn_all = 0
        Fp_all = 0
        gauss_all = []
        for step, (images, point_label_list, lpchar_label_list, lpchar_length_list, name_list) in enumerate(loader_val):
            step += 1
            start_time = time.time()
            if args.cuda:
                images = images.cuda()
                point_label_list = [label.cuda() for label in point_label_list]
                # lpchar_label_list = [trans2cuda_list(l) for l in lpchar_label_list]
            obj_num_list, scores_tensor, coordinates_tensor = model(images, mode1='det_only', mode2='eval')

            # ======================= test accuracy of val =====================
            start_idx_pred = 0
            for batch_idx, obj_num_pred in enumerate(obj_num_list):
                if obj_num_pred != 0:
                    # tensor size(obj_num_pred, 8)
                    single_img_coord_preds = coordinates_tensor[start_idx_pred: start_idx_pred + obj_num_pred]
                    print(single_img_coord_preds)
                    start_idx_pred = start_idx_pred + obj_num_pred
                    Tp, Fn, Fp, gauss_list = postP.gaussian_eval(single_img_coord_preds, point_label_list[batch_idx])
                    # print(Tp, Fn, Fp, gauss_list)
                    Tp_all += Tp
                    Fn_all += Fn
                    Fp_all += Fp
                    gauss_all.extend(gauss_list)
        if Tp_all == 0:
            precision = 0.0
            recall = 0.0
            mGauss = 0.0
        else:
            precision = Tp_all * 1.0 / (Tp_all + Fp_all)
            recall = Tp_all * 1.0 / (Tp_all + Fn_all)
            mGauss = sum(gauss_all) / len(gauss_all)

        # ============================= Total Epoch Print =========================
        print("=> Precision: ", precision)
        print("=> Recall: ", recall)
        print("=> mGauss: %.3f" % (mGauss * 100))
    else:
        print("The validation mode is: detection and recognition.")
        Tp = 0
        Tn_1 = 0
        Tn_2 = 0
        for step, (images, point_label_list, lpchar_label_list, lpchar_length_list, name_list) in enumerate(loader_val):
            step += 1
            start_time = time.time()
            if args.cuda:
                images = images.cuda()
                point_label_list = [label.cuda() for label in point_label_list]
                lpchar_label_list = [trans2cuda_list(l) for l in lpchar_label_list]
            # print('val lpcharlist', lpchar_label_list)
            obj_num_list, scores_tensor, coordinates_tensor, predict_sequence = model(images, mode1='det_reg',
                                                                                      mode2='eval')
            # print('obj_num_list', obj_num_list)
            # print('predict_sequence', predict_sequence)
            if sum(obj_num_list) == 0:
                continue
            keep_pred_list, keep_pred_tensor, lp_labels_clean, length_labels_clean = online_distribute_ctc_targets(
                obj_num_list, coordinates_tensor, point_label_list, lpchar_label_list, lpchar_length_list)
            # print('sequence_probs', sequence_probs)
            # print('val', keep_pred_list, keep_pred_tensor, lp_labels_clean, length_labels_clean)
            pred_labels = []
            for batch_lp_idx in range(len(keep_pred_list)):
                keep_tensor = keep_pred_list[batch_lp_idx]
                single_sequence = predict_sequence[batch_lp_idx]
                for obj in range(keep_tensor.shape[0]):
                    if keep_tensor[obj]:
                        pred_labels.append(single_sequence)
            start = 0
            targets = []
            for length in length_labels_clean:
                label = lp_labels_clean[start:start + length]
                targets.append(label.cpu())
                start += length
            targets_np = np.array([el.numpy() for el in targets])
            # print('target_np', targets_np)

            for i, pred_label in enumerate(pred_labels):
                # print("Target: \n", targets_np[i])
                # print("Preb: \n", pred_label)
                if len(pred_label) != len(targets_np[i]):
                    Tn_1 += 1
                    continue
                if (np.asarray(pred_label) == np.asarray(targets_np[i])).all():
                    Tp += 1
                else:
                    Tn_2 += 1
        if Tp == 0:
            average_epoch_accuracy_val = 0.0
        else:
            average_epoch_accuracy_val = Tp * 1.0 / (Tp + Tn_1 + Tn_2)
        print("The accuracy is: {:0.4f}".format(average_epoch_accuracy_val))


def main(args):
    # savedir = '../Modelsave/{}'.format(args.savedir)
    savedir = os.path.join(cfg.save_parent_folder, str(args.savedir))
    print("The save file path is: " + savedir)

    if not os.path.exists(savedir):
        os.makedirs(savedir)

    with open(savedir + '/opts.txt', "w") as myfile:
        myfile.write(str(args))

    # ============================== Load Model ===========================
    model = DetectionRecognitionPipeline(input_size=train_cfg.INPUT_SIZE,  # (1024, 1024)
                                         det_size=train_cfg.DETECTION_SIZE,   # (512, 512)
                                         reg_size=train_cfg.RECOGNITION_SIZE,   # (144, 48)
                                         class_num=len(CHARS))  # 68
    # =====================================================================
    if args.cuda:
        model = model.cuda()

    # train(args, model)
    """
    if (not args.decoder):
        print("========== ENCODER TRAINING ===========")
        model = train(args, model, True)  # Train encoder"""
    # CAREFUL: for some reason, after training encoder alone, the decoder gets weights=0.
    # We must reinit decoder weights or reload network passing only encoder in order to train decoder
    print("========== START TRAINING ===========")
    model = eval(args, model)  # Train decoder
    print("========== EVALUATE FINISHED ===========")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--mode', type=int, default=1)  # 0 or 1
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--steps_interval', type=int, default=50, help='show loss every how many steps')
    parser.add_argument('--savedir', default="ssnetv2_total_2_11")
    parser.add_argument('--pretrained', default="./weight/ssnet_3_8/weight3_8/model_best.pth")  # "./weight/pretrained_original/model_best.pth"

    main(parser.parse_args())