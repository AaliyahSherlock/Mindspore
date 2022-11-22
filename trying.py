""" Full assembly of the parts to form the complete network """
"""Refer https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py"""
import mindspore as ms
from mindspore import Tensor
import numpy as np
import mindspore.ops as ops
import os
import sys
from mindspore import ops
from mindspore import nn
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore as ms
from PIL import Image
from model.unet_model import UNet
import pandas as pd

"""ms.set_context train\predict\test里都有
欧克，解决
#有的地方改的不确定，比如这个ms.set_context"""
#https://blog.csdn.net/qq_43215538/article/details/126161578
# 有问题，torch里出来的torch type的CPU
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = ms.set_context(device_target="CPU")
# print(device)
# # 问了论坛，可用
# device = ms.get_context("device_target") # "device_id"为GPU使用
# print(device)
# print(type(device))

"""修改？predict，test里 参考：https://www.mindspore.cn/docs/zh-CN/r1.8/api_python/mindspore/mindspore.load_checkpoint.html#mindspore-load-checkpoint
    没有找到torch.load中map_location对应的参数
"""
#         # net.load_state_dict(ms.load_checkpoint('best_model.pth'))
# input_x1 = Tensor(np.array([[0, 1], [2, 1]]).astype(np.float32))
# input_x2 = Tensor(np.array([[0, 1], [2, 1]]).astype(np.float32))
# op = ops.Concat()
# output = op((input_x1, input_x2))
# print(output)

"""# # 修改？ 现在没有对应的算子,不知道这样可不可以，
发现哪里都没用到它，不管它也不会报错"""
# temph = max(inputs.size)  # 做一个等比例放大
# temp = max(inputs.size)  # 取一个最长边
# mask = Image.new('RGB', (temp, temp), (0, 0, 0))  # 制作一个mask，把原图贴上去，(temp,temp)正方形，(0,0,0)表示黑色
# mask.paste(inputs, (0, 0))  # 粘贴原图到mask，统一粘到左上角
# inputs = mask.resize((ht, wt))  # resize到标准size(同是正方形的等比例缩放)
# # inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)



# op = ops.Concat(1)
# output = op((input_x1, input_x2))
# print(output)

# x = Tensor(np.random.randn(3, 4,3).astype(np.float32))
# print(x)
# op = ops.ReduceSum(keep_dims=False)
# output = op(x, [0,1])
# print(output)
# print('second')
# output = op(output, 0)
# print(output)



# x = Tensor(np.array([[0, 1], [2, 1]]).astype(np.float32))
# print(x)
# softmax = ops.Softmax(axis=-1)
# output = softmax(x)
# print(output)

# path = r'D:\demo\UNet\Q_Y_throat_data_patient (2)\data\data_set\Training_Images'
# path = os.path.join(path,'qian_157.png')
# ht,wt = 256,256
# inputs = Image.open(path)
# # inputs.show()
# print(inputs.size)
# temp = max(inputs.size) # 取一个最长边
# mask = Image.new('RGB', (temp, temp), (0, 0, 0))  # 制作一个mask，把原图贴上去，(temp,temp)正方形，(0,0,0)表示黑色
# mask.paste(inputs, (0, 0))  # 粘贴原图到mask，统一粘到左上角
# mask = mask.resize((ht, wt))  # resize到标准size(同是正方形的等比例缩放)
# mask.show()
# inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)



# torch 转 mindspore


# x = Tensor([[[[1, 2, 3, 4], [5, 6, 7, 8]]]], ms.float32)
# resize_bilinear = nn.ResizeBilinear()
# result = resize_bilinear(x, size=(5,5))
# print(x)
# print(result)

file_name = 'a.xlsx'
a = pd.read_excel('C:/Users/Aaliy/Desktop/test3(2).xlsx')
# df = pd.DataFrame({
#    'FR': [-19.72, 60.52, 3],
#     'GR': [60.52, 1.7482, 1.8519],
#      'IT': [804.74, 810.01, 860.13]},
#      index=['A', 'B', 'C'])
#
# s = pd.DataFrame([-19.72, 60.52, 3])
print(a)
print(a['利润'].pct_change(periods=52))
print('\n')
# print(df['FR'].pct_change())