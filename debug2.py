import os
import numpy as np
import time
import random
import torch
import cv2

from backbone.transformer import RT1_Mod

model = RT1_Mod(
    
).cuda()


test = torch.rand([1, 3, 1, 224, 224]).cuda()

res = model(test)

print("over")

# from backbone.robotic_transformer_pytorch import MaxViT, RT1

# vit = MaxViT(
#     num_classes = 1000,
#     dim_conv_stem = 64,
#     dim = 96,
#     dim_head = 32,
#     depth = (2, 2, 5, 2),
#     window_size = 7,
#     mbconv_expansion_rate = 4,
#     mbconv_shrinkage_rate = 0.25,
#     dropout = 0.1
# ).cuda()

# RT1model = RT1(
#     vit = vit,
#     depth = 6,
#     heads = 8,
#     dim_head = 64,
#     cond_drop_prob = 0.2,
#     conditioner_kwargs = dict(
#         model_types = 't5',
#         model_names = r'/home/amax/Project/Pretrained/t5-v1_1-base',
#     )
# ).cuda()

# test = torch.rand([1, 3, 1, 224, 224]).cuda()

# res = RT1model(test, ['pick'])

# print("over")
