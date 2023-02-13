# import torch
# from backbone import *
# import os
# import numpy as np
# import torchvision
# import pickle
# import math
# import sys
# import logging
# import scipy
# import json


# similarity = torch.tensor([[2.0, 1, 1], [1, 2.0, 1], [1, 2.0, 1]])
# similarity_softmax = torch.softmax(similarity, dim=1)
# print(similarity_softmax)
#
# num_array = torch.arange(0, 12).reshape(3, 4).expand((2, 3, 4))
# print(num_array)
#
# print((1, 2) + (3, 4))
#
# backbone = MyInception_v3(transform_input=False)
# images_in_flat = torch.rand((8, 3, 80, 80), dtype=torch.float)
# outputs = backbone(images_in_flat)
# print(outputs[0].shape)
# features_multiscale = [torch.randint(0, 5, (3, 12, 12)) for i in range(8)]
# features_multiscale = torch.cat(features_multiscale, dim=1)  # B*T, D, OH, OW
# print(features_multiscale.shape)
#
# T = 10
# length = 15
# img_idx_list = np.arange(T)
# img_idx_list = img_idx_list.repeat(2)
# idx_list = np.linspace(0, T * 2 - 1, length).astype(np.int)
# image_frame_idx = [img_idx_list[idx_list[i]] for i in range(length)]
# print(image_frame_idx)
#
# print(torch.ones(5))
#
# a = torch.ones(4)
# b = 2 * torch.ones(4)
# c = torch.cat([a, b])
# print(c)
#
# device_list = [f'{i}' for i in range(16)]
# device_list = ','.join(device_list)
# print(device_list)
#
# final_score = torch.ones(5, 1).reshape(-1).numpy().tolist()
# print(final_score)
#
#
# # print(torchvision.models.inception_v3())
#
# def temporal_position_encoding(size):
#     bs = size[0]
#     max_len = size[1]
#     d_model = size[2]
#     pe = torch.zeros(max_len, d_model)
#     position = torch.arange(0, max_len).unsqueeze(1)
#     div_term = torch.exp(torch.arange(0, d_model, 2) *
#                          -(math.log(10000.0) / d_model))
#     pe[:, 0::2] = torch.sin(position * div_term)
#     pe[:, 1::2] = torch.cos(position * div_term)
#     pe = pe.unsqueeze(0)
#     pe_b = torch.cat([pe for i in range(bs)])
#     return pe_b
#
# pos_encoding = temporal_position_encoding((1, 4, 4))
# print(pos_encoding)
# #
# # bp_features = torch.tensor(np.load(r'C:\Users\张世乙\Desktop\Ball_001_s.npy'))
# # print(bp_features.shape)
# # #
# # formation_dict = pickle.load(open(r'C:\Users\张世乙\Desktop\formation_features_middle_1.pkl', 'rb'))
# # for key in formation_dict.keys():
# #     print(formation_dict[key].shape)
# rho_best = 0.1231423
# a = f'{rho_best:.4f}'
# print(a)
#
# def log_file():
#     log_file = 'testfun.log'
#     handler_control = logging.StreamHandler()    # stdout to console
#     # handler_control.setLevel('WARNING')             # 设置INFO级别
#
#     selfdef_fmt = '%(asctime)s - %(name)s - %(funcName)s - %(levelname)s - %(message)s'
#     formatter = logging.Formatter(selfdef_fmt)
#     handler_control.setFormatter(formatter)
#
#     logger = logging.getLogger('updateSecurity')
#     logger.setLevel('DEBUG')           #设置了这个才会把debug以上的输出到控制台
#
#     logger.addHandler(handler_control)
#     logger.info('info,一般的信息输出')
#     logger.debug('debug,blabla')
#     logger.warning('waring，用来用来打印警告信息')
#     logger.error('error，一般用来打印一些错误信息')
#     logger.critical('critical，用来打印一些致命的错误信息，等级最高')
#
# log_file()
#
# prefixs = [''] * 8
# print(prefixs)

def func():
    print(1)


print("This is using locals() : ", locals())
locals()['func']()
