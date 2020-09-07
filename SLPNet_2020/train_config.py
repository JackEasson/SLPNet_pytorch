K_Means_args = {
    'split_value': (3670, 10780),
    'small_size512': (0, 3670),
    'middle_size512': (3670, 10780),
    'large_size512': (10780, 512*512),
    'effective_ratio': (1.0, 0.8, 0.6),  # (effective_ratio)
    'ignore_ratio': (1.5, 1.2, 0.9),
}

INPUT_SIZE = (1024, 1024)  # (W, H)
DETECTION_SIZE = (512, 512)  # (W, H)
RECOGNITION_SIZE = (144, 48)
T_LENGTH = 18


train_img_folder_path = "./data/train/image"
train_txt_folder_path = "./data/train/label"
val_img_folder_path = "./data/val/image"
val_txt_folder_path = "./data/val/label"
save_parent_folder = "./weight"
