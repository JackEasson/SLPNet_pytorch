"""
Use K Means cluster to split lp size into three parts
In my data, get size center 3670 and 10780
_______________________________________________________
class       small     |      middle      |     large
-------------------------------------------------------
size       --3670     |    3670-10780    |    10780--
-------------------------------------------------------
eff&ign   1.0 & 1.5   |    0.8 & 1.2    |    0.6 & 0.9
_______________________________________________________
2019/12/25
"""
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2


def read_filenames(label_path):
    names_list = []
    for parent, dirnames, filenames in os.walk(label_path):
        for filename in filenames:
            # 后缀判断
            (name, exten) = os.path.splitext(filename)
            if exten == '.txt':
                names_list.append(filename)  # 得到无后缀的名字列表
    return names_list


label_path = "D:/网络模型与数据/数据/车牌项目/车牌定位/yoloLP/yolo_txt"
# label_path = "D:/网络模型与数据/数据/车牌项目/车牌定位/yolo_new_11_22/yolo_new_11_22/label"
txt_list = read_filenames(label_path)
size_list = []
sizeMax = 0.0
sizeMin = 10000.0
for txt in txt_list:
    with open(os.path.join(label_path, txt), "r") as f:  # 设置文件对象
        for line in f:
            line = line.strip()
            data = line.split(' ')
            # print(data, float(data[3]), float(data[4]))
            size = float(data[3]) * 512 * float(data[4]) * 512
            # print(txt, size)
            if size > sizeMax:
                sizeMax = size
            if size < sizeMin:
                sizeMin = size

            size_list.append(size)
# print(size_list)
print('Max', sizeMax)
print('Min', sizeMin)

size_list_new = []
for size in size_list:
    if sizeMax * 0.9 >= size >= sizeMin * 0.9:
        size_list_new.append(size)

size_list = np.float32(size_list_new)
# plt.hist(size_list, 50), plt.show()

# K_Means
# Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# Set flags (Just to avoid line break in the code)
flags = cv2.KMEANS_RANDOM_CENTERS
# Apply KMeans
compactness, labels, centers = cv2.kmeans(size_list, 3, None, criteria, 10, flags)
# print(labels)
# plt show
A = []
B = []
C = []
for x, y in zip(size_list, labels):
    if y == 0:
        A.append(x)
    elif y == 1:
        B.append(x)
    else:
        C.append(x)
# Now plot 'A' in red, 'B' in blue, 'centers' in yellow
plt.hist(A, 100, color='r')
plt.hist(B, 100, color='g')
plt.hist(C, 100, color='y')
plt.hist(centers, 20, color='b')
plt.show()
print('centers: (1) %d,  (2) %d,  (3) %d' % (centers[0], centers[1], centers[2]))
print('split points: (1) %d,  (2) %d' % ((centers[0] + centers[1]) / 2, (centers[1] + centers[2]) / 2))
