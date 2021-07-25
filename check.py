
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import math
from PIL import Image
import cv2
# from pyntcloud import PyntCloud
from data_utils import *
import pickle
import torch
import open3d as o3d
from random import randint,random
# path = "data/augment/augm/train/labels/"
# pa = "data/augment/augm/train/images/"
dic = {}
# test = os.listdir(path)
imgs = {}
i = 0
# for item in test:
#     if i>10:
#         break
#     i+=1
#     # print(item)
#     with open(path+item) as f:
#         l,x,y,w,h = [float(x) for x in next(f).split()]
#     dic[item]=[l,x,y,w,h]
#     # print(dic[item])
#     im = cv2.imread(pa+item[:-3]+'jpg')
#     imgs[item] = im
#
#
# for item in dic.keys():
#     im = imgs[item]
#     l,x,y,w,h = dic[item]
#     print(item, dic[item],x-w/2,x+w/2,y-h/2,y+h/2)
#     for i in range(len(im)):
#         for j in range(len(im[0])):
#             if i/640>x-w/2 and i/640<x+w/2 and j/640>y-h/2 and j/640<y+h/2:
#                 im[i,j,2] = 100
#     cv2.imshow('image', im)
#     cv2.waitKey(0)

path_img = "./normals/augment/allData/"
path_lab = "./normals/augment/allData/"
#
# path_img = "data/normals/debug/"
# path_lab = "data/normals/debug/"

img = os.listdir(path_img)
lab = os.listdir(path_lab)


for item in img:
    if item[-4:]=='.jpg':
        name = item[:-4]
        im = cv2.imread(path_img + item)
        imgs[name] = im

for item in lab:
    if item[-4:] == '.txt':
        name = item[:-4]
        f  = open(path_lab+item, 'r')
        data = f.read()

        if len(data)>1:
            l,x,y,w,h = data.split()
            # dic[name]=[l,x,y,w,h]
            l, x, y, w, h = float(l), float(x), float(y), float(w), float(h)
            dic[name] = [l, x, y, w, h]

            # siz = 640
            # box_data_rot = "0 " + str(x / siz) + " " + str(x / siz) + " " + str(h / siz) + " " + str(w / siz)
            # print(l, x, y, w, h, data)
            # f.write(box_data_rot)
            f.close()

        else:
            dic[name]=[0,0,0,0,0]





k = 0
for item in imgs.keys():
    # if k>10:
    #     break
    k+=1
    name = item
    im = imgs[name]
    l, x, y, w, h = dic[name]

    # print(item, dic[name],x-w/2,x+w/2,y-h/2,y+h/2)
    for i in range(len(im)):
        for j in range(len(im[0])):
            if i/640>x-w/2 and i/640<x+w/2 and j/640>y-h/2 and j/640<y+h/2:
                im[j,i,2] = 100

    print(l, x, y, w, h, name)
    cv2.imshow('image', im)
    cv2.waitKey(0)
