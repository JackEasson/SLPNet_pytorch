import os
import cv2


provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁",
             "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
             'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']


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


def unique_validate():
    txt_path = "./train/txt"
    img_path = "./train/image"
    txt_files = read_filenames(txt_path, ['.txt'])
    remove_list = []
    for txt in txt_files:
        with open(os.path.join(txt_path, txt), "r") as f_txt:
            lines = f_txt.readlines()
            if len(lines) > 1:
                print(txt)
                remove_list.append(os.path.join(txt_path, txt))
    for file in remove_list:
        os.remove(file)
    img_files = read_filenames(img_path, ['.jpg', '.png'])
    for img in img_files:
        img2txt = os.path.splitext(img)[0] + '.txt'
        if img2txt not in txt_files:
            print(img2txt)
            img_full = os.path.join(img_path, img)
            os.remove(img_full)


def create_det_reg_txt():
    txt_path = "D:\\tf_related\\LicensePlateProjects20191123\\数据添加2_11\\test2\\test2_txt"
    save_txt_path = "D:\\tf_related\\LicensePlateProjects20191123\\数据添加2_11\\test2\\test2_txt2"
    txt_files = read_filenames(txt_path, ['.txt'])
    for txt in txt_files:
        # print(txt)
        just_name = os.path.splitext(txt)[0]
        split_infos = just_name.split('-')
        lp_list = split_infos[-3].split('_')
        # print(lp_list)
        lp_str = provinces[int(lp_list[0])] + alphabets[int(lp_list[1])] + ads[int(lp_list[2])] + \
                 ads[int(lp_list[3])] + ads[int(lp_list[4])] + ads[int(lp_list[5])] + ads[int(lp_list[6])]
        print(lp_str)

        with open(os.path.join(txt_path, txt), "r") as f_txt:
            lines = f_txt.readlines()
            with open(os.path.join(save_txt_path, txt), "w", encoding='utf-8') as f_save:
                save_line = lines[0].strip() + lp_str + '\n'
                f_save.write(save_line)
        # break


def trans_txt2utf8():
    txt_path = "D:\\tf_related\\LicensePlateProjects20191123\\数据添加2_11\\fn_txt"
    save_txt_path = "D:\\tf_related\\LicensePlateProjects20191123\\数据添加2_11\\fn_txt2"
    txt_files = read_filenames(txt_path, ['.txt'])
    for txt in txt_files:
        with open(os.path.join(txt_path, txt), "r") as f_txt:
            lines = f_txt.readlines()
            with open(os.path.join(save_txt_path, txt), "w", encoding='utf-8') as f_save:
                for line in lines:
                    f_save.write(line)


def alignment_txt_img():
    img_path = "D:\\网络模型与数据\\工具\\C++标注相关\\LPCornerLabels\\LPCornerLabels\\LPImages\\ccpd_fn"
    txt_path = "D:\\网络模型与数据\\工具\\C++标注相关\\LPCornerLabels\\LPCornerLabels\\output"
    img_files = read_filenames(img_path, ['.jpg', '.png'])
    print(len(img_files))
    txt_files = read_filenames(txt_path, ['.txt'])
    print(len(txt_files))
    count = 0
    for txt in txt_files:
        txt2img = txt.split('.')[0] + '.jpg'
        if txt2img not in img_files:
            os.remove(os.path.join(txt_path, txt))
            count += 1
    print('count: ', count)


if __name__ == '__main__':
    # unique_validate()
    create_det_reg_txt()