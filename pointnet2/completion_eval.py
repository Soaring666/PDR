import os
import argparse
import json

import numpy as np
import torch
import torch.nn as nn

import pickle

from util import rescale, find_max_epoch, print_size, sampling, calc_diffusion_hyperparams, AverageMeter
from util_fastdpmv2 import fast_sampling_function_v2

from models.pointnet2_ssg_sem import PointNet2SemSegSSG
from models.pointnet2_with_pcld_condition import PointNet2CloudCondition
from models.point_upsample_module import point_upsample
from chamfer_loss_new import Chamfer_F1


from dataset import get_dataloader



from eval.plot_result import plot_result
from eval.compare_eval_result import plot_result_list


import h5py
import time

name_to_number ={
'plane': '02691156',
'bench': '02828884',
'cabinet': '02933112',
'car': '02958343',
'chair': '03001627',
'monitor': '03211117',
'lamp': '03636649',
'speaker': '03691459',
'firearm': '04090263',
'couch': '04256520',
'table': '04379243',
'cellphone': '04401088',
'watercraft': '04530566'}

number_to_name = {}
for k in name_to_number.keys():
    number_to_name[name_to_number[k]] = k


def evaluate(net, testloader, diffusion_hyperparams, print_every_n_steps=200, 
                dataset='mvp_dataset', scale=1, save_generated_samples=False, save_dir = None,
                task = 'completion', refine_output_scale_factor=None, max_print_nums=1e8,
                save_multiple_t_slices=False,
                t_slices=[5, 10, 20, 50, 100, 200, 400, 600, 800],
                use_a_precomputed_XT=False, T_step=100,
                point_upsample_factor=1, include_displacement_center_to_final_output=False,
                num_points=None, 
                noise_magnitude_added_to_gt=0.01, add_noise_to_generated_for_refine_exp=False,
                fast_sampling=False, fast_sampling_config=None, diffusion_config=None):
    CD_meter = AverageMeter()
    F1_meter = AverageMeter()
    EMD_meter = AverageMeter()
    total_len = len(testloader)

    # total meta is label info
    total_meta = torch.rand(0).cuda().long()  #初始化一个张量，内容为空
    metrics = {'cd_distance': torch.rand(0).cuda(), 'emd_distance': torch.rand(0).cuda(),
                'cd_p': torch.rand(0).cuda(), 'f1': torch.rand(0).cuda()}

    f1_threshold = 0.001 if dataset == 'mvp40' else 0.0001
    cd_module = Chamfer_F1(f1_threshold=f1_threshold)      


    if save_generated_samples:
        print('generated_samples will be saved to the directory', save_dir)
        total_generated_data = None
        if save_multiple_t_slices:
            generated_data_t_slices = None

    
    total_time = 0
    for idx, data in enumerate(testloader):

        print_interval = int(np.ceil(total_len / max_print_nums))
        if (idx) % print_interval == 0:
            print('begin generating')

        label = data['label'].cuda()
        condition = data['partial'].cuda()
        gt = data['complete'].cuda()
        if task == 'refine_completion':
            generated = data['generated'].cuda()
        if use_a_precomputed_XT:
            XT = data['XT'].cuda()
        else:
            XT = None
    
        batch = gt.shape[0]
        try:
            num_points = gt.shape[1]
        except:
            num_points = num_points
            print('num points is set to %d' % num_points)

        start = time.time()
        if task == 'refine_completion':
            if add_noise_to_generated_for_refine_exp:
                generated = generated + torch.normal(0, noise_magnitude_added_to_gt, size=generated.shape, device=generated.device)
            displacement = net(generated, condition, ts=None, label=label)
            if point_upsample_factor > 1:
                generated_data, _ = point_upsample(generated, displacement, point_upsample_factor, 
                                                        include_displacement_center_to_final_output,
                                                        refine_output_scale_factor)
            else:
                generated_data = generated + displacement * refine_output_scale_factor

        else:
            if save_multiple_t_slices:
                generated_data, result_slices = sampling(net, (batch,num_points,3), 
                                diffusion_hyperparams, 
                                print_every_n_steps=print_every_n_steps, label=label, 
                                condition=condition,
                                verbose=False, return_multiple_t_slices=True,
                                t_slices=t_slices,
                                use_a_precomputed_XT=use_a_precomputed_XT, step=T_step, XT=XT)
                # result_slices is a dict that contains torch tensors
            else:
                if fast_sampling:
                    generated_data = fast_sampling_function_v2(net, (batch,num_points,3), diffusion_hyperparams,  # DDPM parameters
                                diffusion_config,
                                print_every_n_steps=print_every_n_steps, label=label, 
                                verbose=False, condition=condition,
                                **fast_sampling_config)
                else:
                    generated_data = sampling(net, (batch,num_points,3), 
                                diffusion_hyperparams, 
                                print_every_n_steps=print_every_n_steps, label=label, 
                                condition=condition,
                                verbose=False,
                                use_a_precomputed_XT=use_a_precomputed_XT, step=T_step, XT=XT)
        generation_time = time.time() - start
        total_time = total_time + generation_time
        generated_data = generated_data/2/scale
        gt = gt/2/scale
        if save_multiple_t_slices:
            for key in result_slices.keys():
                result_slices[key] = result_slices[key]/2/scale
                result_slices[key] = result_slices[key].detach().cpu().numpy()
                
        torch.cuda.empty_cache()
        
        cd_p, dist, f1 = cd_module(generated_data, gt)
        cd_loss = dist.mean().detach().cpu().item()
        f1_loss = f1.mean().detach().cpu().item()
        emd_cost = torch.zeros_like(dist)
        emd_loss = emd_cost.mean().detach().cpu().item()
           
        total_meta = torch.cat([total_meta, label])
        

        metrics['cd_distance'] = torch.cat([metrics['cd_distance'], dist])
        metrics['emd_distance'] = torch.cat([metrics['emd_distance'], emd_cost])
        metrics['cd_p'] = torch.cat([metrics['cd_p'], cd_p])
        metrics['f1'] = torch.cat([metrics['f1'], f1])

        CD_meter.update(cd_loss, n=batch)
        F1_meter.update(f1_loss, n=batch)
        EMD_meter.update(emd_loss, n=batch)
        if (idx) % print_interval == 0:
            print('progress [%d/%d] %.4f (%d samples) CD distance %.8f EMD distance %.8f F1 score %.6f this batch time %.2f total generation time %.2f' % (idx, total_len, 
                idx/total_len, batch, CD_meter.avg, EMD_meter.avg, F1_meter.avg, generation_time, total_time), flush=True)

        if save_generated_samples:
            save_file = os.path.join(save_dir, 'mvp_generated_data_%dpts.h5' % num_points)
            
            if total_generated_data is None:
                total_generated_data = generated_data.detach().cpu().numpy()
            else:
                    total_generated_data = np.concatenate([total_generated_data, 
                                        generated_data.detach().cpu().numpy()], axis=0)
            hf = h5py.File(save_file, 'w')
            hf.create_dataset('data', data=total_generated_data)
            hf.close()

            # save t slices
            if save_multiple_t_slices:
                if generated_data_t_slices is None:
                    generated_data_t_slices = result_slices
                else:
                    for t in t_slices:
                        generated_data_t_slices[t] = np.concatenate([generated_data_t_slices[t],
                                        result_slices[t]], axis=0)
                
                for t in t_slices:
                    if dataset == 'mvp_dataset':
                        t_save_file = os.path.join(save_dir, 'mvp_generated_data_%dpts_T%d.h5' % (num_points, t))
                    hf = h5py.File(t_save_file, 'w')
                    hf.create_dataset('data', data=generated_data_t_slices[t])
                    hf.close()

            if (idx) % print_interval == 0:
                print('%d files have been saved to the directory %s' % (batch, save_dir))
            

    total_meta = total_meta.detach().cpu().numpy()
    
    return CD_meter.avg, EMD_meter.avg, total_meta, metrics

def get_each_category_distance(files):
    handle = open(files, 'rb')
    data = pickle.load(handle)
    handle.close()
    # pdb.set_trace()
    meta = data['meta']
    distance_keys = ['cd_distance', 'emd_distance']
    cate_split_result = []
    for distance in distance_keys:
        split_result = {}
        for k in name_to_number.keys():
            split_result[k] = []
        for i, m in enumerate(meta):
            number = m.split('/')[-2]
            cate = number_to_name[number]
            split_result[cate].append(data[distance][i])
        final_split_result = {}
        for k in split_result.keys():
            if len(split_result[k]) > 0:
                final_split_result[k] = np.array(split_result[k]).mean()
                # print(k, final_split_result[k])
        cate_split_result.append(final_split_result)
    for idx, dis in enumerate(distance_keys):
        new_key = dis + '_category_split_result'
        data[new_key] = cate_split_result[idx]
    handle = open(files, 'wb')
    pickle.dump(data, handle)
    handle.close()
    print('Have splitted distance of each category for file %s' % files, flush=True)
    return 0

def gather_eval_result_of_different_iters(directory, match1, match2, nomatch=None, split_category = False, save_suffix = '', plot=True,
    # gather all evaluation results from all ckpts and plot them in figures
    gathered_keys=['iter', 'avg_cd', 'avg_emd', 'cd_distance_category_split_result', 'emd_distance_category_split_result']):
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    files = [f for f in files if match1 in f and match2 in f]
    if not nomatch is None:
        files = [f for f in files if not nomatch in f]
    gathered_results = {}
    for f in files:
        if split_category:
            get_each_category_distance(os.path.join(directory, f))
        handle = open(os.path.join(directory, f), 'rb')
        data = pickle.load(handle)
        handle.close()
        for key in gathered_keys:
            if key in data.keys():
                if isinstance(data[key], dict): # data[key] is a dictionary
                    if key in gathered_results.keys():
                        for sub_key in data[key].keys():
                            gathered_results[key][sub_key].append(data[key][sub_key])
                            # data[key][sub_key] is a single number
                    else:
                        gathered_results[key] = {}
                        for sub_key in data[key].keys():
                            gathered_results[key][sub_key] = [ data[key][sub_key] ]
                else: # data[key] is a single number
                    if key in gathered_results.keys():
                        gathered_results[key].append(data[key])
                    else:
                        gathered_results[key] = [data[key]]
            else:
                print('key %s is not in the data loaded from file %s' % (key, f), flush=True)
    save_file = os.path.join(directory, 'gathered_eval_result'+save_suffix+'.pkl')
    handle = open(save_file, 'wb')
    pickle.dump(gathered_results, handle)
    handle.close()
    if plot:
        plot_result(gathered_results, gathered_keys[0], os.path.join(directory, 'figures'+save_suffix), 
                    plot_values=gathered_keys[1:], print_lowest_value=False)
    return gathered_results

def plot_train_and_val_eval_result(eval_dir):
    # plot testset and trainset figures in the same figure, and find the ckpt that has the lowest loss value
    label_list = ['test set', 'train set']
    files = ['gathered_eval_result.pkl', 'gathered_eval_result_trainset.pkl']
    
    file_list = [os.path.join(eval_dir, files[i]) for i in range(len(files))]

    plot_values = ['avg_cd', 'avg_emd', 'avg_cd_p', 'avg_f1']
    result_list = []
    for f in file_list:
        handle = open(f, 'rb')
        result = pickle.load(handle)
        result_list.append(result)
        handle.close()

    save_dir = os.path.join(eval_dir, 'compare_test_and_train_set')
    plot_result_list(result_list, 'iter', label_list, save_dir, line_style=None, plot_values=plot_values,
                        print_lowest_value=True)


