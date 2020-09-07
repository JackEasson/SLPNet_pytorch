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
from module.det_part.detection_head import MultiConstraintsGaussDistanceLoss, GaussLoss, SmoothL1Loss
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


def xavier(param):
    nn.init.xavier_uniform_(param)


def weights_init(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 0.0, 0.01)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


# 学习率调整
def adjust_lr(optimizer, gamma):
    """
    调整学习率的值
    """
    # param_groups管理optimizer参数
    for param_group in optimizer.param_groups:
        param_group['lr'] *= gamma


def train(args, model):
    # ===============================【0】Argument Estimate ========================
    assert args.grad_accumulation_steps is None or args.grad_accumulation_steps > 0
    if args.reg_epochs is None:
        assert args.acc_threshold is not None
    else:
        assert args.acc_threshold is None
        assert args.reg_epochs < args.num_epochs
    # ===============================【1】DataSet=================================
    # 1）train dataset loader； 2）val dataset loader
    dataset_train = LPDataSet(img_path=train_cfg.train_img_folder_path, txt_path=train_cfg.train_txt_folder_path)
    dataset_val = LPDataSet(img_path=train_cfg.val_img_folder_path, txt_path=train_cfg.val_txt_folder_path)
    print("=>Train dataset total images: % d" % dataset_train.__len__())
    print("=>Val dataset total images: % d" % dataset_val.__len__())
    loader_train = DataLoader(dataset_train, num_workers=args.num_workers, batch_size=args.batch_size,
                              drop_last=True, shuffle=True, collate_fn=base_lp_collate)
    loader_val = DataLoader(dataset_val, num_workers=args.num_workers, batch_size=args.batch_size,
                            drop_last=False, shuffle=False, collate_fn=base_lp_collate)
    # ===============================【2】Loss Function=================================
    det_criterion = MultiConstraintsGaussDistanceLoss()
    ctc_criterion = nn.CTCLoss(blank=len(CHARS)-1, reduction='mean')
    print("Train loss function for detection: ", type(det_criterion))
    print("Train loss function for recognition: ", type(ctc_criterion))
    # ===============================【3】save train information and models=================================
    # save dir
    savedir = os.path.join(cfg.save_parent_folder, str(args.savedir))

    # model instruction
    modeltxtpath = savedir + "/model.txt"
    # loss txt
    losstxtpath = savedir + "/loss.txt"

    with open(modeltxtpath, "w") as f:
        f.write(str(model))

    if not os.path.exists(losstxtpath):  # dont add first line if it exists
        with open(losstxtpath, "a") as f:
            f.write("Epoch\t\tStep\t\tTotalStep\t\tDet-loss\t\tReg-loss\t\tTrainMode")

    # ===============================【4】optimizer =================================
    # 权重衰减等价于L2范数正则化(regularzation)。正则化通过模型损失函数添加惩罚项使学到的模型参数值较小，是应对过拟合的常用方法。
    optimizer = Adam(model.parameters(), args.learning_rate, (0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    # optimizer = SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    # ===============================【5】ckpt resume =================================
    start_epoch = 1
    step_total = 0
    best_acc_det = 0.0  # 仅训练检测部分时使用
    best_acc_reg = 0.0  # 联合训练时使用
    train_mode = 0
    if args.resume:  # 继续训练
        # Must load weights, optimizer, epoch and best value.
        filenameCheckpoint = savedir + '/checkpoint.pth'
        assert os.path.exists(
            filenameCheckpoint), "Error: resume option was used but checkpoint was not found in folder"
        checkpoint = torch.load(filenameCheckpoint)
        start_epoch = checkpoint['epoch']
        step_total = checkpoint['step_total']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_acc_reg = checkpoint['best_acc']
        train_mode = checkpoint['mode']
        print("=> Loaded checkpoint at epoch {}".format(checkpoint['epoch']))

    elif args.pretrained is not None:
        print("Load weight from pretrained model ...")
        pretrained_dict = torch.load(args.pretrained)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print("=> Load weight successfully.")

    else:
        print("Start weight initialize ...")
        weights_init(model)
        print("=> Load weight successfully.")
    # ===============================【6】begin train =================================
    optimizer.zero_grad()
    for epoch in range(start_epoch, args.num_epochs + 1):
        print("----- TRAINING - EPOCH", epoch, "-----")
        # 先进行模式判断： detection only or detection & recognition
        # train mode: 0 for detection and 1 for total
        if train_mode == 0:
            if args.reg_epochs is not None:
                if epoch > args.reg_epochs:
                    train_mode = 1

            else:
                if best_acc_det >= args.acc_threshold:
                    train_mode = 1
        # 调整学习率
        if epoch != 1:
            adjust_lr(optimizer, 0.99)

        epoch_loss_det = []
        epoch_loss_reg = []
        time_train = []

        usedLr = 0
        for param_group in optimizer.param_groups:
            print("LEARNING RATE: ", param_group['lr'])
            usedLr = float(param_group['lr'])

        model.train()
        # one epoch train process
        for step, (images, point_label_list, lpchar_label_list, lpchar_length_list, name_list) in enumerate(loader_train):
            # print("Now process ", name_list)
            # if step > 0:
                # break
            # step += 1
            step_total += 1
            # 调整学习率
            start_time = time.time()
            if args.cuda:
                images = images.cuda()
                point_label_list = [label.cuda() for label in point_label_list]
                lpchar_label_list = [trans2cuda_list(l) for l in lpchar_label_list]

            # replace variable, use requires_grad_()
            images = images.requires_grad_()
            point_label_list = [label.requires_grad_() for label in point_label_list]
            # lpchar_label_list = [trans2requires_grad_list(l) for l in lpchar_label_list]

            if train_mode == 0:  # detetcion only
                outputs = model(images, mode1='det_only', mode2='train')
                # here outputs is pred_maps
                det_loss, coord_loss, score_loss = det_criterion(output_maps=outputs, corners_list_512=point_label_list)
                # loss_data = loss.item()
                coord_loss_data = coord_loss.item()
                score_loss_data = score_loss.item()
                reg_loss = None
            else:
                outputs = model(images, mode1='det_reg', mode2='train')
                obj_num_list, scores_tensor, coordinates_tensor, pred_maps, logits = outputs
                # print(obj_num_list)
                # print(scores_tensor)
                # print(coordinates_tensor)
                # print(logits)
                det_loss, coord_loss, score_loss = det_criterion(output_maps=pred_maps, corners_list_512=point_label_list)
                # print('det_loss', det_loss)
                coord_loss_data = coord_loss.item()
                score_loss_data = score_loss.item()
                # 验证有没有检测到车牌用以识别
                if logits is None:  # also sum(obj_num_list) == 0
                    reg_loss = None
                else:  # 有车牌检测出并加以识别
                    # 动态NMS匹配检出的车牌与GT，匹配成功的计算reg_loss
                    # print('lpchar_label_list', lpchar_label_list)

                    keep_pred_list, keep_pred_tensor, lp_labels_clean, length_labels_clean = \
                        online_distribute_ctc_targets(obj_num_list, coordinates_tensor, point_label_list,
                                                      lpchar_label_list, lpchar_length_list)
                    # print("lp num: %d" % keep_pred_tensor.shape[0])
                    # print(keep_pred_list, keep_pred_tensor, lp_labels_clean, length_labels_clean)
                    # ============= 判断匹配后余下的是否为空 ==============
                    if lp_labels_clean is not None:  # 不为空
                        # 取出匹配成功的并通道转换
                        log_probs = logits[keep_pred_tensor].permute(2, 0, 1)  # for ctc loss: T x N x C
                        # print(labels.shape)
                        log_probs = torch.log_softmax(log_probs, dim=2)
                        # print(outputs[0][0], targets)
                        # ==================== calculate loss ===================
                        # get ctc parameters
                        input_lengths, target_lengths = sparse_tuple_for_ctc(cfg.T_LENGTH, length_labels_clean)
                        # print(log_probs.shape, lp_labels_clean, input_lengths, target_lengths)
                        reg_loss = ctc_criterion(log_probs, lp_labels_clean, input_lengths=input_lengths,
                                                 target_lengths=target_lengths)
                        # print('reg_loss', reg_loss)
                        if reg_loss.item() == np.inf or reg_loss.item() == np.nan:
                            reg_loss = None
                    else:
                        reg_loss = None

                    # print('loss', loss)
            # print(loss)
            # 加入累积梯度更新
            if args.grad_accumulation_steps is not None:
                det_loss_avg = det_loss / args.grad_accumulation_steps
                if reg_loss is not None:
                    reg_loss_avg = reg_loss / args.grad_accumulation_steps
                    loss_avg = det_loss_avg + 0.5 * reg_loss_avg
                    loss_avg.backward()
                    # det_loss_avg.backward(retain_graph=True)
                    # reg_loss_avg.backward()
                else:
                    det_loss_avg.backward()
                if step_total % args.grad_accumulation_steps == 0:
                    optimizer.step()  # 实际更新梯度
                    optimizer.zero_grad()  # 梯度清零
            else:
                if reg_loss is not None:
                    # det_loss.backward(retain_graph=True)
                    # reg_loss.backward()
                    loss = det_loss + 0.5 * reg_loss
                    loss.backward()
                else:
                    det_loss.backward()
                optimizer.step()  # 实际更新梯度
                optimizer.zero_grad()  # 梯度清零

            det_loss_data = det_loss.item()
            epoch_loss_det.append(det_loss_data)
            if reg_loss is not None:
                reg_loss_data = reg_loss.item()
                epoch_loss_reg.append(reg_loss_data)
            else:
                reg_loss_data = None
            time_train.append(time.time() - start_time)

            if args.steps_interval > 0 and step_total % args.steps_interval == 0:
                if reg_loss_data is not None:
                    print('mode:{}  det_loss: {:0.4f}  coord_loss: {:0.4f}  score_loss: {:0.4f}  reg_loss: {:0.4f}  '
                          '(epoch: {}, step: {})'.format(
                           train_mode, det_loss_data, coord_loss_data, score_loss_data, reg_loss_data, epoch, step+1),
                          "// Avg time/img: %.4f s" % (sum(time_train) / len(time_train) / args.batch_size))
                    # record to txt
                    # Epoch		Step		TotalStep		Train-loss
                    with open(losstxtpath, "a") as f:
                        f.write("\n%d\t\t%d\t\t%d\t\t%.4f\t\t%.4f\t\t%d" %
                                (epoch, step + 1, step_total, det_loss_data, reg_loss_data, train_mode))

                else:
                    print('mode:{}  det_loss: {:0.4f}  coord_loss: {:0.4f}  score_loss: {:0.4f}  reg_loss: {}  '
                          '(epoch: {}, step: {})'.format(
                           train_mode, det_loss_data, coord_loss_data, score_loss_data, reg_loss_data, epoch, step + 1),
                          "// Avg time/img: %.4f s" % (sum(time_train) / len(time_train) / args.batch_size))
                    # record to txt
                    # Epoch		Step		TotalStep		Train-loss
                    with open(losstxtpath, "a") as f:
                        f.write("\n%d\t\t%d\t\t%d\t\t%.4f\t\t{}\t\t%d".format(reg_loss_data) %
                                (epoch, step+1, step_total, det_loss_data, train_mode))

        average_epoch_loss_det = sum(epoch_loss_det) / len(epoch_loss_det)
        average_epoch_loss_reg = sum(epoch_loss_reg) / len(epoch_loss_reg) if epoch_loss_reg else None
        print("\t=>The average detection loss of epoch %d is %.4f" % (epoch, average_epoch_loss_det))
        if average_epoch_loss_reg is not None:
            print("\t=>The average recognition loss of epoch %d is %.4f" % (epoch, average_epoch_loss_reg))
        else:
            print("\t=>The average recognition loss of epoch %d is None, didn't train")

        # continue
        # ######################################################################################################
        # ================================== val accuracy ======================================
        print("----- VALIDATING - EPOCH", epoch, "-----")
        model.eval()
        if train_mode == 0:  # detection only
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
            current_acc_det = mGauss
            is_best = current_acc_det > best_acc_det  # best_acc initial value: 0.0
            best_acc_det = max(current_acc_det, best_acc_det)
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
                obj_num_list, scores_tensor, coordinates_tensor, predict_sequence = model(images, mode1='det_reg', mode2='eval')
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
            print("The accuracy of epoch {} is: {:0.4f}".format(epoch, average_epoch_accuracy_val))
            current_acc_reg = average_epoch_accuracy_val
            is_best = current_acc_reg > best_acc_reg
            best_acc_reg = max(current_acc_reg, best_acc_reg)

        filenameCheckpoint = savedir + '/checkpoint.pth'
        filenameBest = savedir + '/checkpoint_best.pth'

        # ================ Here, if best, we save it as the best model ================
        save_checkpoint({
            'epoch': epoch + 1,
            'step_total': step_total,
            'arch': str(model),
            'state_dict': model.state_dict(),
            'best_acc': best_acc_reg,
            'mode': train_mode,
            'optimizer': optimizer.state_dict(),
        }, is_best, filenameCheckpoint, filenameBest)

        # SAVE MODEL AFTER EPOCH
        filename = '{}/model-{}.pth'.format(savedir, epoch)
        filenamebest = '{}/model_best.pth'.format(savedir)
        # ===================固定epoch保存的仅有模型weight参数====================
        if args.epochs_save > 0 and epoch % args.epochs_save == 0 and epoch > 10:
            torch.save(model.state_dict(), filename)
            print('save: {} (epoch: {})'.format(filename, epoch))
        if is_best:
            torch.save(model.state_dict(), filenamebest)
            print('save: {} (epoch: {})'.format(filenamebest, epoch))
            with open(savedir + "/best.txt", "w") as f:
                f.write("Best epoch is %d, with Val-Accuracy= %.4f" % (epoch, best_acc_reg))
    return model  # return model (convenience for encoder-decoder training)


def save_checkpoint(state, is_best, filenameCheckpoint, filenameBest):
    torch.save(state, filenameCheckpoint)
    if is_best:
        print("Saving model as best")
        torch.save(state, filenameBest)


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

    if args.state:
        # if args.state is provided then load this state for training
        # Note: this only loads initialized weights. If you want to resume a training use "--resume" option!!
        """
        try:
            model.load_state_dict(torch.load(args.state))
        except AssertionError:
            model.load_state_dict(torch.load(args.state,
                map_location=lambda storage, loc: storage))
        #When model is saved as DataParallel it adds a model. to each key. To remove:
        #state_dict = {k.partition('model.')[2]: v for k,v in state_dict}
        #https://discuss.pytorch.org/t/prefix-parameter-names-in-saved-model-if-trained-by-multi-gpu/494
        """

        def load_my_state_dict(model, state_dict):  # custom function to load model when not all dict keys are there
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    continue
                own_state[name].copy_(param)
            return model

        # print(torch.load(args.state))
        model = load_my_state_dict(model, torch.load(args.state))

    # train(args, model)
    """
    if (not args.decoder):
        print("========== ENCODER TRAINING ===========")
        model = train(args, model, True)  # Train encoder"""
    # CAREFUL: for some reason, after training encoder alone, the decoder gets weights=0.
    # We must reinit decoder weights or reload network passing only encoder in order to train decoder
    print("========== START TRAINING ===========")
    model = train(args, model)  # Train decoder
    print("========== TRAINING FINISHED ===========")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--state')
    parser.add_argument('--learning_rate', type=int, default=0.005)
    parser.add_argument('--num_epochs', type=int, default=120, help='total epoch to train')
    # Argument 'reg_epochs' and 'acc_threshold' just need one! Recommend for using acc_threshold.
    parser.add_argument('--reg_epochs', type=int, default=None, help='start epoch to add recognition')
    # This argument 'acc_threshold' is used to control when to transform LP detection only
    # to LP detection and recognition.
    parser.add_argument('--acc_threshold', type=float, default=0.85, help='accuracy threshold to start recognize')

    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--grad_accumulation_steps', default=1, type=int,
                        help='Number of gradient accumulation steps')  # Here the real batch is (grad_accumulation_steps * batch_size)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--steps_interval', type=int, default=50, help='show loss every how many steps')
    parser.add_argument('--epochs_save', type=int, default=10)  # You can use this value to save model every X epochs
    parser.add_argument('--savedir', default="SLPNetSave") # The save path, under folder -- ./weight
    parser.add_argument('--resume', action='store_true', default=False)  # Use this flag to load last checkpoint for training
    parser.add_argument('--pretrained', default=None)  # "./weight/pretrained_original/model_best.pth"

    main(parser.parse_args())