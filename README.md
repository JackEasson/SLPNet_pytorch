# SLPNet_pytorch
SLPNet: Towards End-to-End Car License Plates Detection and Recognition Using Lightweight CNN

## Background
This is a model for Automatic License Plate Detection and Recognition, which is trained on CCPD. Hence, it only supports for Chinese Blue LPs now.

## Install
Now we just provide a demonstration for fun. But it's the best model we have trained and can show our SLPNet's performance.<br>
Just run the main.py and a interface (by PyQt5) will appear and it's easy to operate.<br>
`python main.py`

## Environment
Pytorch >= 1.1.0<br>
Opnecv<br>
numpy<br>
PyQt5<br>

## Results
We show some recognition results based on images from CCPD and shot personally.<br>
![picture1](https://github.com/JackEasson/SLPNet_pytorch/tree/master/example_pictures/example3.PNG)<br>
![picture2](https://github.com/JackEasson/SLPNet_pytorch/tree/master/example_pictures/example4.PNG)

## Others
The entire codes including train.py and test.py will be available soon.

## Notice
We open our SLPNet in other platform now. It's a stand-alone program on Windows system and you can just run it in a PC without pytorch or opencv! Here is the URL: 百度网盘 链接：https://pan.baidu.com/s/13Cfb-LOhfa9ZrQnVBotQ9w    提取码：pp2n
For successful operation, all the path (whether save path or input images path) of this project should't contain Chinese characters!
