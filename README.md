# SLPNet_pytorch
SLPNet: Towards End-to-End Car License Plates Detection and Recognition Using Lightweight CNN

# Update
The full source is available now! You can train your own LP detector and recognizer together with your data easiliy. We also provide the best model trained on CCPD dataset in weight/weight3_8/model_best.pth!

## Background
This is a model for Automatic License Plate Detection and Recognition, which is trained on CCPD. Hence, it only supports for Chinese Blue LPs now.

## Install
### 1. For demonstration
Now we just provide a demonstration for fun. But it's the best model we have trained and can show our SLPNet's performance.<br>
Just run the main.py and a interface (by PyQt5) will appear and it's easy to operate.<br>
`cd SLPNet_Demo`<br>
`python main.py`

### 2. For train and test
First, you should put your license plate data in 'data' folder, including three part: train, val and test. The train and test part need images together with their labels. Some example are put in the folder and you can refer to the labels' format. The format is as following:<br>
`303 510,472 510,474 554,307 556,皖AG1191`<br>
It represents 8 corners of a LP and the number. If there is more than one LP, then write the information in next line.<br>
The easiest usage:<br>
For train: `python train.py --savedir SLPNetweight`<br>
For test: `pytthon test_demo.py`<br>
More arguments can be adjust in the train_config.py, train.py and test_demo.py.

## Environment
Pytorch >= 1.1.0<br>
Opnecv<br>
numpy<br>
PyQt5<br>

## Results
We show some recognition results based on images from CCPD and shot personally.<br>
![picture1](https://github.com/JackEasson/SLPNet_pytorch/tree/master/example_pictures/example3.PNG)<br>
![picture2](https://github.com/JackEasson/SLPNet_pytorch/tree/master/example_pictures/example4.PNG)

## Notice
We open our SLPNet in other platform now. It's a stand-alone program on Windows system and you can just run it in a PC without pytorch or opencv! Here is the URL: 百度网盘 链接：https://pan.baidu.com/s/13Cfb-LOhfa9ZrQnVBotQ9w    提取码：pp2n
For successful operation, all the path (whether save path or input images path) of this project shouldn't contain Chinese characters!

## Other
Our paper has acceted by PRCV2020, we will open it soon!
