import torch
import h5py
import numpy as np
import wandb

with h5py.File('pointnet2/mvp_generated_data_2048pts_epoch120.h5', 'r') as f:
    data = np.array(f['data'][()])
    print(data.shape)

result = torch.randn(10, 8, 2048, 3)
result = np.array(result)
wandb.login()
wandb.init(project='show point cloud')
for i in range(data.shape[1]):
    wandb.log({"point_cloud": wandb.Object3D(result[0][i])})

