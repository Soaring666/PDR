import torch
import numpy as np
import torch.utils.data as data
import h5py
import os
import random
import copy
import sys
import warnings
import pickle


sys.path.insert(0, os.path.dirname(__file__))
from mvp_data_utils import augment_cloud

class ShapeNetH5(data.Dataset):
    def __init__(self, data_dir, train=True, npoints=2048, novel_input=True, novel_input_only=False,
                        scale=1, random_sample_nums=None,
                        augmentation=False, return_augmentation_params=False,
                        include_generated_samples=False, generated_sample_path=None,
                        randomly_select_generated_samples=False, # randomly select a trial from multi trial generations
                        use_mirrored_partial_input=False,
                        load_pre_computed_XT=False, T_step=100, XT_folder=None,
                        ):
        """
        params:
            novel_input_only: 只使用novel points
            novel_input: 把本来的和novel的cat起来使用,如果为False,则既不用mirror也不加novel
            use_mirrored_partial_input: 需要设置novel_input为True, novel_input_only为False
        """
        self.return_augmentation_params = return_augmentation_params
        self.use_mirrored_partial_input = use_mirrored_partial_input
        self.npoints = npoints
        self.train = train # controls the trainset and testset
        self.augmentation = augmentation # augmentation could be a dict or False
        self.random_sample_nums = random_sample_nums
        
        if self.train:
            self.mirrored_input_path = ('%s/mirror_and_concated_partial/mvp_train_input_mirror_and_concat_3072pts.h5' % 
                                data_dir) if use_mirrored_partial_input else None 
            self.input_path = '%s/mvp_train_input.h5' % data_dir
            self.gt_path = '%s/mvp_train_gt_%dpts.h5' % (data_dir, npoints)
        else:
            self.input_path = '%s/mvp_test_input.h5' % data_dir
            self.gt_path = '%s/mvp_test_gt_%dpts.h5' % (data_dir, npoints)

        # load partial point clouds and their labels
        print(self.input_path)
        input_file = h5py.File(self.input_path, 'r')
        self.input_data = np.array((input_file['incomplete_pcds'][()]))
        self.labels = np.array((input_file['labels'][()]))
        self.novel_input_data = np.array((input_file['novel_incomplete_pcds'][()]))
        self.novel_labels = np.array((input_file['novel_labels'][()]))
        input_file.close()

        # load gt complete point cloud
        gt_file = h5py.File(self.gt_path, 'r')
        self.gt_data = np.array((gt_file['complete_pcds'][()]))
        self.gt_labels = np.array((gt_file['labels'][()]))
        self.novel_gt_data = np.array((gt_file['novel_complete_pcds'][()]))
        self.novel_gt_labels = np.array((gt_file['novel_labels'][()]))
        gt_file.close()

        # load XT generated from a trained DDPM
        self.load_pre_computed_XT = load_pre_computed_XT
        if load_pre_computed_XT:
            if train:
                XT_folder = os.path.join(XT_folder, 'train')
            else:
                XT_folder = os.path.join(XT_folder, 'test')
            self.T_step = T_step
            XT_file = os.path.join(XT_folder, 'mvp_generated_data_2048pts_T%d.h5' % T_step)
            self.XT_file = XT_file
            generated_XT_file = h5py.File(XT_file, 'r')
            self.generated_XT = np.array(generated_XT_file['data'])
            generated_XT_file.close()

        # load X0 generated from a trained DDPM
        self.include_generated_samples = include_generated_samples
        self.generated_sample_path = generated_sample_path
        self.randomly_select_generated_samples = randomly_select_generated_samples
        if include_generated_samples:
            # generated_samples/T1000_betaT0.02_shape_completion_no_class_condition_scale_1_no_random_replace_partail_with_complete/ckpt_1403999/
            generated_samples_file = os.path.join(data_dir, generated_sample_path)
            if randomly_select_generated_samples:
                files = os.listdir(generated_samples_file)
                files = [f for f in files if f.startswith('trial')]
                files = [os.path.join(generated_samples_file, f) for f in files]
                files = [generated_samples_file] + files
                generated_samples_file = random.choice(files)
                print('Randomly select file %s for generated samples from %d files' % (generated_samples_file, len(files)))

            if train:
                generated_samples_file = os.path.join(generated_samples_file, 'train')
            else:
                generated_samples_file = os.path.join(generated_samples_file, 'test')
            generated_samples_file = os.path.join(generated_samples_file, 'mvp_generated_data_2048pts.h5')

            generated_file = h5py.File(generated_samples_file, 'r')
            self.generated_sample = np.array(generated_file['data'])
            generated_file.close()
            # generated_sample should have the same number of shapes as input_data, 
            # because we generate one complete point cloud using the trained DDPM for each partial point cloud
            # however, they may have different number of points for each shape
        
        # combine normal input and novel input
        if novel_input_only:
            self.input_data = self.novel_input_data
            self.gt_data = self.novel_gt_data
            self.labels = self.novel_labels
        elif novel_input:
            #mirrored_input_data是全部的点云，所以不需要cat
            if use_mirrored_partial_input:
                print(self.mirrored_input_path)
                mirrored_file = h5py.File(self.mirrored_input_path, 'r')
                self.input_data = np.array(mirrored_file['data'])
                mirrored_file.close()
            else:
                self.input_data = np.concatenate((self.input_data, self.novel_input_data), axis=0)
            
            #只取训练集的前1000张飞机图进行训练看效果
            self.input_data = self.input_data[0:2000, :, :]
            self.gt_data = np.concatenate((self.gt_data, self.novel_gt_data), axis=0)
            self.labels = np.concatenate((self.labels, self.novel_labels), axis=0)

        # randomly subsample the datasets, because we may want to only test the trained DDPM on a fraction of the 
        # dataset to save time 
        if self.random_sample_nums is not None: 
            #partial_to_complete_index就是gt_data
            partial_to_complete_index = np.arange(self.gt_data.shape[0])  #(B,)
            partial_to_complete_index = np.repeat(partial_to_complete_index[:,np.newaxis], 26, axis=1)  #(B, 26)
            partial_to_complete_index = partial_to_complete_index.reshape((self.gt_data.shape[0]*26))  #(B*26,)
            index = list(range(self.input_data.shape[0]))
            idx = random.sample(index, self.random_sample_nums)
            idx = np.array(idx) 
            self.input_data = self.input_data[idx]
            self.labels = self.labels[idx]
            self.partial_to_complete_index = partial_to_complete_index[idx]
            if self.include_generated_samples:
                self.generated_sample = self.generated_sample[idx]
            if self.load_pre_computed_XT:
                self.generated_XT = self.generated_XT[idx]
        else:
            warnings.warn("it will sample all")

        self.scale = scale
        # shapes in mvp dataset range from -0.5 to 0.5
        # we rescale the, to make the, range from -scale to scale 
        if use_mirrored_partial_input:
            # note that in this case self.input_data is of shape B,N,4
            # the last dimension indicates whether the corresponding point is the original one or the mirrowed point
            self.input_data[:,:,0:3] = self.input_data[:,:,0:3] * 2 * scale
        else:
            self.input_data = self.input_data * 2 * scale

        self.gt_data = self.gt_data * 2 * scale
        if self.include_generated_samples:
            self.generated_sample = self.generated_sample * 2 * scale
        if self.load_pre_computed_XT:
            self.generated_XT = self.generated_XT * 2 * scale

        print('partial point clouds:', self.input_data.shape)
        print('gt complete point clouds:', self.gt_data.shape)
        print('labels:', self.labels.shape)
        if self.include_generated_samples:
            print('DDPM generated complete point clouds:', self.generated_sample.shape)
        if self.load_pre_computed_XT:
            print('DDPM generated intermediate complete point clouds:', self.generated_XT.shape)
        self.labels = self.labels.astype(int)
        self.len = self.input_data.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        # deepcopy is necessary here, because we may alter partial and complete for data augmentation
        # it will change the original data if we do not deep copy
        result = {}
        result['partial'] = copy.deepcopy(self.input_data[index])
        if self.random_sample_nums is not None:
            gt_idx = self.partial_to_complete_index[index]
        else:
            gt_idx = index // 26
        result['complete'] = copy.deepcopy(self.gt_data[gt_idx])
        
        if self.include_generated_samples:
            result['generated'] = copy.deepcopy(self.generated_sample[index])
        if self.load_pre_computed_XT:
            result['XT'] = copy.deepcopy(self.generated_XT[index])

        # augment the point clouds
        if isinstance(self.augmentation, dict):
            result_list = list(result.values())
            if self.return_augmentation_params:
                result_list, augmentation_params = augment_cloud(result_list, self.augmentation,
                                                                return_augmentation_params=True)
            else:
                result_list = augment_cloud(result_list, self.augmentation, return_augmentation_params=False)
            for idx, key in enumerate(result.keys()):
                result[key] = result_list[idx]
            if self.include_generated_samples:
                # add noise to every point in the point cloud generated by a trained DDPM
                # this is used to train the refinement network
                sigma = self.augmentation.get('noise_magnitude_for_generated_samples', 0)
                if sigma > 0:
                    noise = np.random.normal(scale=sigma, size=result['generated'].shape)
                    noise = noise.astype(result['generated'].dtype)
                    result['generated'] = result['generated'] + noise

        if self.return_augmentation_params:
            for key in augmentation_params.keys():
                result[key] = augmentation_params[key]
        for key in result.keys():
            result[key] = torch.from_numpy(result[key])
        result['label'] = self.labels[index]

        return result

if __name__ == '__main__':
    from tqdm import tqdm
    import time
    aug_args = {'pc_augm_scale':1.5, 'pc_augm_rot':True, 'pc_rot_scale':30.0, 'pc_augm_mirror_prob':0.5, 'pc_augm_jitter':False, 'translation_magnitude': 0.1}
    aug_args = False
    include_generated_samples=False
    path = os.getcwd()
    data_dir = os.path.join(path, 'pointnet2/mvp_dataloader/data/mvp_dataset') 
    # generated_sample_path='generated_samples/T1000_betaT0.02_shape_completion_mirror_rot_60_scale_1.2_translation_0.05/ckpt_623999'
    dataset = ShapeNetH5(data_dir=data_dir, train=True, npoints=2048, novel_input=True, novel_input_only=False,
                            augmentation=aug_args, scale=1,
                            random_sample_nums=None,
                            include_generated_samples=include_generated_samples, 
                            generated_sample_path=None,
                            use_mirrored_partial_input=False,
                            load_pre_computed_XT=False, T_step=10, 
                            XT_folder=None,
                            return_augmentation_params=False)
 
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    print(len(dataloader))
   
    for i, data in tqdm(enumerate(dataloader)):
        label, partial, complete = data['label'], data['partial'], data['complete']
        # data['M_inv'] is of shape (B,3,3)
        # data['translation'] is of shape (B,1,3)
        print('label %s partail shape %s [%.3f, %.3f] complete shape %s [%.3f, %.3f]' % (
            label.shape, partial.shape, partial.min(), partial.max(), complete.shape, complete.min(), complete.max(),))
        time.sleep(0.5)
     