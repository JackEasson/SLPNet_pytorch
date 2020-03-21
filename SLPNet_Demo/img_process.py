from cv2 import resize, cvtColor, COLOR_BGR2RGB, COLOR_RGB2BGR, line, circle, LINE_AA
import numpy as np
from torch import tensor, load
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import ToTensor
from model.detection_recognition_pipeline import DetectionRecognitionPipeline


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


def default_image_preprocess(image, inp_size=(1024, 1024)):
    """
    返回image的tensor
    """
    image = resize(image, inp_size)
    image = cvtColor(image, COLOR_BGR2RGB).astype(np.float32)  # bgr to rgb
    image = (image / 255.0) * 2.0 - 1.0  # to -1.0 ~ 1.0
    img_tensor = ToTensor()(image)  # (H, W, C) -> (C, H, W), scale also -1.0 ~ 1.0
    return img_tensor


def model_initial():
    # create model
    model = DetectionRecognitionPipeline()
    model = model.eval()
    # load weight
    pretrained_dict = load('./weight/model_best.pth')
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # print(pretrained_dict)
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print("Weights of SLPNet have been loaded.")
    return model


def image_det_reg_process(model, img_initial, img_resize_for_show):
    # get image tensor
    img_tensor = default_image_preprocess(img_initial)
    obj_num_list, scores_tensor, coordinates_tensor, predict_sequence = model(img_tensor.unsqueeze(0), mode1='det_reg', mode2='eval')
    # post process
    lp_char_list = []
    start_idx_pred = 0
    obj_num_pred = obj_num_list[0]
    if obj_num_pred != 0:
        single_img_coord_preds = coordinates_tensor[start_idx_pred: start_idx_pred + obj_num_pred]
        single_img_scores = scores_tensor[start_idx_pred: start_idx_pred + obj_num_pred]
        for lp_list in predict_sequence:
            lp_char = ''
            for c in lp_list:
                lp_char += CHARS[c]
            lp_char_list.append(lp_char)
        img_resize_for_show = result_show(img_resize_for_show, single_img_coord_preds, single_img_scores, lp_char_list)
    return img_resize_for_show


def result_show(img_for_show, per_img_coords, per_img_scores, lp_chars, det_size=(512, 512)):
    img_h, img_w = img_for_show.shape[:2]
    ratio_h = img_h / det_size[1]
    ratio_w = img_w / det_size[0]
    ratio_size = tensor([ratio_w, ratio_h]).float()
    point_size = 1
    point_color1 = (0, 0, 255)  # BGR
    point_color2 = (0, 0, 255)  # BGR
    thickness = 6  # 可以为 0 、4、8
    # part1 out
    obj_num = per_img_coords.shape[0]
    for obj in range(obj_num):
        points = per_img_coords[obj].reshape(4, 2)
        points = points * ratio_size
        lp_char = lp_chars[obj]
        print("The recognition result NO.%d: %s" % (obj+1, lp_char))
        # lines
        for i in range(3):
            line(img_for_show, (int(points[i][0].item()), int(points[i][1].item())),
                     (int(points[i + 1][0].item()), int(points[i + 1][1].item())), (0, 255, 0), 2, LINE_AA)
        line(img_for_show, (int(points[3][0].item()), int(points[3][1].item())),
                 (int(points[0][0].item()), int(points[0][1].item())), (0, 255, 0), 2, LINE_AA)
        for i, point in enumerate(points):
            circle(img_for_show, (int(point[0].item()), int(point[1].item())), point_size, point_color1, thickness)
        # confidence
        """
        cv2.putText(img_for_show, "conf: %.4f" % per_img_scores[obj].item(),
                    (int(points[3][0].item()-10), int(points[3][1].item())+10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)"""
        img_for_show = cv2ImgAddText(img_for_show, "score:%.3f" % per_img_scores[obj].item(),
                                     pos=(int(points[3][0].item() - 5), int(points[3][1].item()) + 3))
        # draw lp str
        img_for_show = cv2ImgAddText(img_for_show, lp_char, pos=(int(points[3][0].item()-5), int(points[3][1].item())+25))
    return img_for_show


def cv2ImgAddText(img, text, pos, textColor=(255, 0, 0), textSize=22):
    if isinstance(img, np.ndarray):  # detect opencv format or not
        img = Image.fromarray(cvtColor(img, COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype("./lib/ttc/NotoSansCJK-Regular.ttc", textSize, encoding="utf-8")
    draw.text(pos, text, textColor, font=fontText)
    return cvtColor(np.asarray(img), COLOR_RGB2BGR)
