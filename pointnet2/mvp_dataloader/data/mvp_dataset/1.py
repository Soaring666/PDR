import h5py
import datasets
import numpy as np

from h5py import Dataset, Group, File

with File('mvp_test_gt_2048pts.h5', 'r') as f:
    for key in f.keys():
        print(f[key])
        print(f[key], key, f[key].name)
        print(f[key])
        print(key)
        print(f[key].name)
a = 12
print("{}adas{}".format(a-3, a+4))

