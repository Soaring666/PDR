import torch
import numpy as np
import torch.utils.data as data
import h5py
import os
import random
import copy
import sys
import warnings

sys.path.insert(0, os.path.dirname(__file__))
from mvp_dataset import ShapeNetH5
from mirror_partial import mirror_and_concat

if __name__ == '__main__':
    train = True
    batch_size = 128
    num_points = 3072

    path = os.getcwd()
    data_dir = 'mvp_dataloader/data/mvp_dataset'
    mirror_save_dir = 'mirror_and_concated_partial'
    os.makedirs(os.path.join(path, data_dir, mirror_save_dir), exist_ok=True)
    save_file = 'mvp_train_input_mirror_and_concat_%dpts.h5' % num_points
    save_file = os.path.join(path, data_dir, mirror_save_dir, save_file)

    dataset = ShapeNetH5(data_dir, train=train,  npoints=2048, novel_input=True, novel_input_only=False,
                            scale=0.5, random_subsample=False, num_samples=100000)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    mirror_concat = None
    save_interval = 20

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            label, partial, complete = data['label'], data['partial'], data['complete']
            print('index %d label %s partail shape %s [%.3f, %.3f] complete shape %s [%.3f, %.3f]' % (
                i, label.shape, partial.shape, partial.min(), partial.max(), complete.shape, complete.min(), complete.max(),))
            concat = mirror_and_concat(partial, axis=2, num_points=[2048, 3072])
            concat = [con.detach().cpu().numpy() for con in concat]
            if mirror_concat is None:
                mirror_concat = concat
            else:
                for idx in range(len(mirror_concat)):
                    mirror_concat[idx] = np.concatenate([mirror_concat[idx], concat[idx]], axis=0)
            if i % save_interval == 0 or i==len(dataloader)-1:
                if i % save_interval == 0 and i>0:
                    if i == len(dataloader)-1:
                        j = i % save_interval
                        save_data = mirror_concat[2]
                        dataset[(i-j)*batch_size: i*batch_size] = save_data
                    else:
                        hf = h5py.File(save_file, 'a')
                        dataset = hf['data']
                        save_data = mirror_concat[2]
                        dataset[(i-20)*batch_size: i*batch_size] = save_data
                elif i == 0:
                    hf = h5py.File(save_file, 'w')
                    save_data = mirror_concat[2]
                    dataset = hf.create_dataset('data', [62400, 3072, 4])
                    dataset[0: batch_size] = save_data
                    hf.close()
                mirror_concat = None
        hf.close()

    f = h5py.File(save_file, 'r')
    data = f['data']
    print(data.shape)
    print('generated mirror partials have been saved to', save_file)