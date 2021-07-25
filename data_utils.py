import copy
import numpy as np
import open3d as o3d
# import h5py
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
# from pyntcloud import PyntCloud
import pcl

from PIL import Image
import math
import random

def import_labels(path):
    df = pd.read_csv(path, header=None)
    labels = df[:][[0, 1, 2]]
    labels.columns = ['cam', 'num', 'label']
    labels['cam'] = [x[5:] for x in labels['cam']]
    return labels

def read_txt(file):
    load = np.fromfile(file, dtype=np.float32)
    # load_arr = np.asarray(load.points)
    return load

def image_to_point_cloud(depth):
    """Transform a depth image into a point cloud with one point for each
    pixel in the image, using the camera transform for a camera
    centred at cx, cy with field of view fx, fy.

    depth is a 2-D ndarray with shape (rows, cols) containing
    depths from 1 to 254 inclusive. The result is a 3-D array with
    shape (rows, cols, 3). Pixels with invalid depth in the input have
    NaN for the z-coordinate in the result.

    """
    cx, cy, fx, fy = 5.8818670481438744*100,  3.1076280589210484*100, 5.8724220649505514*100, 2.2887144980135292*100
    depth = depth[:,:,0]

    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    valid = (depth > 0) & (depth < 255)
    # print(valid)
    z = np.where(valid, depth / 256.0, np.nan)
    x = np.where(valid, z * (c - cx) / fx, 0)
    y = np.where(valid, z * (r - cy) / fy, 0)
    return np.dstack((x, y, z))

def show_point_cloud(pcd):
    # print(pcd)
    # print(np.asarray(pcd.points)[0][0])
    o3d.visualization.draw_geometries([pcd])
def rename_point_cloud(path,l,r):
    person_cloud = {}
    test = os.listdir(path)
    print("Load a ply point cloud, print it, and render it")
    for item in test:
        pcd = o3d.io.read_point_cloud(path + item)
        name = str(int(item[l:r])+1)
        name = (4-len(name))*'0'+name
        person_cloud[item[l:r]] = pcd

    return person_cloud


def load_list_point_cloud(path,l,r,person=False):
    person_cloud = {}
    test = os.listdir(path)
    print("Load a ply point cloud, print it, and render it")
    for item in test:
        pcd = o3d.io.read_point_cloud(path + item)
        name = item[l:r]
        if person and name!="":
            name = str(int(name) + 1)
            name = (4 - len(name)) * '0' + name
        person_cloud[name] = pcd
    return person_cloud


def scale_to_255(a, min, max, dtype=np.uint8):
    """ Scales an array of values from specified min, max range to 0-255
        Optionally specify the data type of the output (default is uint8)
    """
    return (((a - min) / float(max - min)) * 255).astype(dtype)

def get_bounding_boxes(point_clouds):
    bounding_boxes = {}

    for key, pcd in point_clouds.items():
        X,Y,Z = [],[],[]
        # show_point_cloud(pcd)2
        for point in pcd.points:
            x,y,z= point
            X.append(x)
            Y.append(y)
            Z.append(z)
        if len(X)!=0:
            X_max,Y_max,Z_max = max(X),max(Y),max(Z)
            X_min,Y_min,Z_min = min(X),min(Y),min(Z)
            bounding_boxes[key] = np.array([[X_max,Y_max,Z_max], [X_min,Y_min,Z_min]])
    return bounding_boxes

def get_bounding_box(pcd):

    X,Y,Z = [],[],[]
    for point in pcd.points:
        x,y,z = point
        X.append(x)
        Y.append(y)
        Z.append(z)
    X_max,Y_max,Z_max = max(X),max(Y),max(Z)
    X_min,Y_min,Z_min = min(X),min(Y),min(Z)
    np.array([[X_max,Y_max,Z_max], [X_min,Y_min,Z_min]])
    return np.array([[X_max,Y_max,Z_max], [X_min,Y_min,Z_min]])

def bounding_box_by_image(img_bound_box):
    img_bound_box_v = []
    for i in range(len(img_bound_box)):
        for j in range(len(img_bound_box[0])):
            if img_bound_box[i, j, 1] != 0:
                img_bound_box_v.append([i, j])
    xmi = img_bound_box_v[0][0]
    ymi = img_bound_box_v[0][1]
    xma = img_bound_box_v[1][0]
    yma = img_bound_box_v[1][1]
    return (xmi+xma)/2,(ymi+yma)/2, abs(xmi-xma),abs(ymi-yma)

def point_cloud_2_birdseye(points_,features_=None,
res=0.01,
side_range=(-5, 5), # left-most to right-most
fwd_range = (-5, 5), # back-most to forward-most
height_range=(-2., 2.), # bottom-most to upper-most
isArray = False, xm=0, ym=0,siz = 640
):
    # EXTRACT THE POINTS FOR EACH AXIS
    if not isArray:
        points = np.copy(points_)
    else:
        points = points_

    x_points = points[:, 0]
    y_points = points[:, 2]
    z_points = points[:, 1]

    # FILTER - To return only indices of points within desired cube
    # Three filters for: Front-to-back, side-to-side, and height ranges
    # Note left side is positive y axis in LIDAR coordinates
    f_filt = np.logical_and((x_points > fwd_range[0]), (x_points < fwd_range[1]))
    s_filt = np.logical_and((y_points > -side_range[1]), (y_points < -side_range[0]))
    filter = np.logical_and(f_filt, s_filt)
    indices = np.argwhere(filter).flatten()

    # KEEPERS
    x_points = x_points[indices]
    y_points = y_points[indices]
    z_points = z_points[indices]
    features = features_
    shp = 0
    if features_:
        features = np.copy(features_)
        features = features[indices, :]
        shp = features.shape[1]

    # PointCloud = np.array([x_points, y_points, z_points, features])
    # PointCloud = np.transpose(PointCloud)
    # indices = np.lexsort((-PointCloud[:, 2], PointCloud[:, 1], PointCloud[:, 0]))
    # PointCloud = PointCloud[indices]

    # CONVERT TO PIXEL POSITION VALUES - Based on resolution
    x_img = (-x_points / res).astype(np.int32)  # x axis is -y in LIDAR
    y_img = (-y_points / res).astype(np.int32)  # y axis is -x in LIDAR

    # SHIFT PIXELS TO HAVE MINIMUM BE (0,0)
    # floor & ceil used to prevent anything being rounded to below 0 after shift
    x_img -= int(np.floor(side_range[0] / res))
    y_img += int(np.ceil(fwd_range[1] / res))
    if xm+ ym==0:
        xm, ym = min(x_img), min(y_img)
    x_img -= xm
    y_img -= ym

    # CLIP HEIGHT VALUES - to between min and max heights
    pixel_values = np.clip(a=z_points,
                           a_min=height_range[0],
                           a_max=height_range[1])

    # RESCALE THE HEIGHT VALUES - to be between the range 0-255
    pixel_values = scale_to_255(pixel_values,
                                min=height_range[0],
                                max=height_range[1])


    # INITIALIZE EMPTY ARRAY - of the dimensions we want
    x_max = siz
    y_max = siz

    zg = np.zeros((y_max, x_max))
    zb = np.zeros((y_max, x_max))
    zr = np.zeros((y_max, x_max))

    for x,y,z in zip(x_img,y_img,pixel_values):
        if x<y_max and y<x_max:
            if zg[x,y]<z:
                zg[x,y]=z/255
            zr[x,y]+=1
    zr[:,:] = np.minimum(1.0, np.log(zr[:,:] + 1) / np.log(64))



    # im = np.zeros((y_max, x_max, shp+2))
    #
    # im[:, :, 0] = zr
    # im[:, :, 1] = zg
    #
    # if features_:
    #     for x, y, f in zip(x_img,y_img,features):
    #         im[x, y, -features.shape[1]:] =  f[:]

    im = np.zeros((y_max, x_max, 5))
    if features_:
        for x, y, f in zip(x_img,y_img,features):
            if x < y_max and y < x_max:
                im[x, y, -features.shape[1]:] = f[:]
    im[:, :, 0] = zr
    im[:, :, 1] = zg


    return im, xm, ym
