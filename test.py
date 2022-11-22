# -*- coding: utf-8 -*-
"""
-------------------------------------------------
Project Name: unet
File Name: test.py
Author: chenming
Create Date: 2022/2/7
Description：
-------------------------------------------------
"""
import os
from tqdm import tqdm
from utils.utils_metrics import compute_mIoU, show_results
import glob
import os
import cv2
from model.unet_model import UNet
import mindspore as ms
import numpy as np
from mindspore import Tensor
# import torch
from mindspore import dtype as mstype
# torch 改 mindspore

def cal_miou(test_dir="D:/demo/UNet/Q_Y_throat_data_patient/data/data_set/Test_Images",
             pred_dir="D:/demo/UNet/Q_Y_throat_data_patient/data/data_set/results_try", gt_dir="D:/demo/UNet/Q_Y_throat_data_patient/data/data_set/Test_Labels"):
    # ---------------------------------------------------------------------------#
    #   miou_mode用于指定该文件运行时计算的内容
    #   miou_mode为0代表整个miou计算流程，包括获得预测结果、计算miou。
    #   miou_mode为1代表仅仅获得预测结果。
    #   miou_mode为2代表仅仅计算miou。
    # ---------------------------------------------------------------------------#
    miou_mode = 0
    # ------------------------------#
    #   分类个数+1、如2+1
    # ------------------------------#
    num_classes = 2
    # --------------------------------------------#
    #   区分的种类，和json_to_dataset里面的一样
    # --------------------------------------------#
    name_classes = ["background", "nidus"]
    # name_classes    = ["_background_","cat","dog"]
    # -------------------------------------------------------#
    #   指向VOC数据集所在的文件夹
    #   默认指向根目录下的VOC数据集
    # -------------------------------------------------------#
    # 计算结果和gt的结果进行比对

    # 加载模型

    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)

        print("Load model.")
        """
        #？ 修改MindSpore
        # 俺也不知道，没找到torch.device对应的部分
        """
        device = ms.get_context("device_target") # "device_id"为GPU使用 # or GPU\Ascend
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 加载网络，图片单通道，分类为1。
        net = UNet(n_channels=1, n_classes=1)
        # 将网络拷贝到deivce中
        # 修改 俺把它注释掉了
        # net.to(device=device)
        # 加载模型参数
        # 修改？ 参考：https://www.mindspore.cn/docs/zh-CN/r1.8/api_python/mindspore/mindspore.load_checkpoint.html#mindspore-load-checkpoint
        # 没有找到torch.load中map_location对应的参数
        # 目前参考的是https://mindspore.cn/docs/zh-CN/r1.9/api_python/mindspore/mindspore.load_param_into_net.html#mindspore.load_param_into_net
        # ckpt_file_name = "./checkpoint/LeNet5-1_32.ckpt"
        # param_dict = ms.load_checkpoint(ckpt_file_name, filter_prefix="conv1")
        # param_not_load = ms.load_param_into_net(net, param_dict)
        # 报错显示好像.pth的格式不行，要，ckpt的才可以
        net = ms.load_param_into_net(net,ms.load_checkpoint('./best_model.pth'))
        # net.load_state_dict(torch.load('best_model.pth', map_location=device)) # todo
        # 测试模式
        net.eval()
        print("Load model done.")

        img_names = os.listdir(test_dir)
        image_ids = [image_name.split(".")[0] for image_name in img_names]

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path = os.path.join(test_dir, image_id + ".png")
            img = cv2.imread(image_path)
            origin_shape = img.shape
            # print(origin_shape)
            # 转为灰度图
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = cv2.resize(img, (512, 512))
            # 转为batch为1，通道为1，大小为512*512的数组
            img = img.reshape(1, 1, img.shape[0], img.shape[1])
            # 转为tensor
            # 修改 参考：https://mindspore.cn/docs/zh-CN/r1.8/api_python/mindspore/mindspore.Tensor.html?#mindspore.Tensor.from_numpy
            img_tensor = Tensor.from_numpy(img)
            # img_tensor = torch.from_numpy(img)
            # 将tensor拷贝到device中，只用cpu就是拷贝到cpu中，用cuda就是拷贝到cuda中。
            # 修改 参考https://www.mindspore.cn/docs/zh-CN/r1.8/api_python/mindspore/mindspore.dtype.html?highlight=dtype
            img_tensor = img_tensor.to(device=device, dtype=ms.float32)
            # img_tensor = img_tensor.to(device=device, dtype=torch.float32)
            # 预测
            pred = net(img_tensor)
            # 提取结果
            pred = np.array(pred.data.cpu()[0])[0]
            pred[pred >= 0.5] = 255
            pred[pred < 0.5] = 0
            pred = cv2.resize(pred, (origin_shape[1], origin_shape[0]), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(pred_dir, image_id + ".png"), pred)

        print("Get predict result done.")

    if miou_mode == 0 or miou_mode == 2:
        print("Get miou.")
        print(gt_dir)
        print(pred_dir)
        print(num_classes)
        print(name_classes)
        hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes,
                                                        name_classes)  # 执行计算mIoU的函数
        print("Get miou done.")
        miou_out_path = "results/"
        show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)

if __name__ == '__main__':
    cal_miou()