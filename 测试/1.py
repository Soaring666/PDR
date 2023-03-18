import numpy as np
import random
import torch
import torch
import torchvision
import open3d as o3d
import os
import h5py

#可视化
num_points = 3072
path = os.getcwd()
data_dir = 'mvp_dataloader/data/mvp_dataset'
mirror_save_dir = 'mirror_and_concated_partial'
os.makedirs(os.path.join(path, data_dir, mirror_save_dir), exist_ok=True)
save_file = 'mvp_train_input_mirror_and_concat_%dpts.h5' % num_points
save_file = os.path.join(path, data_dir, mirror_save_dir, save_file)

f = h5py.File(save_file, 'r')
data = f['data']
print(data.shape)

pcd = o3d.geometry.PointCloud()
pcd_vector = data[30000][:, :3]
print(type(pcd_vector))
print(pcd_vector.shape)
pcd.points = o3d.utility.Vector3dVector(pcd_vector)
o3d.visualization.draw_geometries([pcd])

