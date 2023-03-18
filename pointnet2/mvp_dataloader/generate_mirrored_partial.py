import torch
import numpy as np
import torch.utils.data as data
import h5py
import os
import sys
from data_utils.mirror_partial import mirror_and_concat
sys.path.insert(0, os.path.dirname(__file__))
from mvp_dataset import ShapeNetH5


if __name__ == '__main__':
    train = True
    path = os.getcwd()
    data_dir = 'mvp_dataloader/data/mvp_dataset'
    mirror_save_dir = 'mirror_and_concated_partial'
    os.makedirs(os.path.join(path, data_dir, mirror_save_dir), exist_ok=True)
    # generated_sample_path='generated_samples/T1000_betaT0.02_shape_completion_no_class_condition_scale_1_no_random_replace_partail_with_complete/ckpt_1403999'
    dataset = ShapeNetH5(data_dir, train=train, npoints=2048, novel_input=True, novel_input_only=False,
                            random_replace_partial_with_complete_prob=0, augmentation=False, scale=0.5,
                            random_subsample=False, num_samples=100000,
                            include_generated_samples=False, generated_sample_path=None)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)
    mirror_concat = None
    save_interval = 20
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            # label, partial, complete = data
            label, partial, complete = data
            print('index %d label %s partail shape %s [%.3f, %.3f] complete shape %s [%.3f, %.3f]' % (
                i, label.shape, partial.shape, partial.min(), partial.max(), complete.shape, complete.min(), complete.max(),))
            concat = mirror_and_concat(partial, axis=2, num_points=[2048, 3072])  #([B, 4096, 4], [B, 2048, 4], [B, 3072, 4])
            concat = [con.detach().cpu().numpy() for con in concat]
            if mirror_concat is None:
                mirror_concat = concat
            else:
                for idx in range(len(mirror_concat)):
                    mirror_concat[idx] = np.concatenate([mirror_concat[idx], concat[idx]], axis=0)
            if i % save_interval == 0 or i==len(dataloader)-1:
                for idx in range(len(mirror_concat)):
                    num_points = mirror_concat[idx].shape[1]
                    if benchmark:
                        save_file = 'mvp_benchmark_mirror_and_concat_%dpts.h5' % num_points
                    else:
                        if train:
                            save_file = 'mvp_train_input_mirror_and_concat_%dpts.h5' % num_points
                        else:
                            save_file = 'mvp_test_input_mirror_and_concat_%dpts.h5' % num_points

                    save_file = os.path.join(data_dir, mirror_save_dir, save_file)
                    if i>0:
                        previous_saved_file = h5py.File(save_file, 'r')
                        previous_saved = np.array(previous_saved_file['data'])
                        previous_saved_file.close()
                        save_data = np.concatenate([previous_saved, mirror_concat[idx]], axis=0)
                    else:
                        save_data = mirror_concat[idx]

                    hf = h5py.File(save_file, 'w')
                    hf.create_dataset('data', data=save_data)
                    hf.close()
                mirror_concat = None

    print('generated mirror partials have been saved to', save_file)