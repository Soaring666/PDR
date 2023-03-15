import h5py
import numpy as np

from h5py import Dataset, Group, File

with File('E:/A-CODE/3D点云/PDR/pointnet2/mvp_dataloader/data/mvp_dataset/mvp_test_input.h5', 'r') as f:
    for key in f.keys():
        print(f[key])
        print(f[key], key, f[key].name)
        print(f[key])
        print(key)
        print(f[key].name)
a = 12
print("{}adas{}".format(a-3, a+4))

