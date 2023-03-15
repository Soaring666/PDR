import os
import time
import argparse
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from dataset import get_dataloader
from util import find_max_epoch, print_size
from util import training_loss, calc_diffusion_hyperparams
from scheduler import QuantityScheduler

from models.pointnet2_with_pcld_condition import PointNet2CloudCondition  #net
from models.point_upsample_module import point_upsample  #refine
from chamfer_loss_new import calc_cd

from shutil import copyfile
import copy

from completion_eval import evaluate, get_each_category_distance, gather_eval_result_of_different_iters, plot_train_and_val_eval_result
from json_reader import replace_list_with_string_in_a_dict, restore_string_to_list_in_a_dict
import pickle

def evaluate_per_rank(net, rank, num_gpus, root_directory, local_path, n_iter, 
                    test_trainset=False, num_samples_tested_in_trainset=0, dataset='shapenet', scale=1,
                    task = 'completion', refine_output_scale_factor=None, compute_emd=True,
                    noise_magnitude_added_to_gt=0, add_noise_to_generated_for_refine_exp=False):
    net.eval()
    torch.cuda.empty_cache()
    phase = 'cascade_test' if dataset == 'shapenet_chunk' else 'val'
    if dataset in ['shapenet', 'shapenet_pytorch', 'shapenet_chunk', 'partnet']:
        testloader = get_dataloader(trainset_config, phase=phase, rank=rank, world_size=num_gpus,
                                    append_samples_to_last_rank=False)
    elif dataset in ['mvp_dataset', 'mvp40']:
        testloader = get_dataloader(trainset_config, phase=phase, rank=rank, world_size=num_gpus, 
                            random_subsample=True, num_samples=int(num_samples_tested_in_trainset/num_gpus),
                            append_samples_to_last_rank=False)
    else:
        raise Exception('%s do not supported evaluation yet' % dataset)

    point_upsample_factor = pointnet_config.get('point_upsample_factor', 1) 
    include_displacement_center_to_final_output = pointnet_config.get('include_displacement_center_to_final_output', False) 
    
    with torch.no_grad():
        CD_loss, EMD_loss, meta, metrics = evaluate(net, testloader, 
                        diffusion_hyperparams, print_every_n_steps=200, parallel=False,
                        dataset=dataset, scale=scale, task=task, refine_output_scale_factor=refine_output_scale_factor,
                        max_print_nums=10, point_upsample_factor=point_upsample_factor,
                        include_displacement_center_to_final_output=include_displacement_center_to_final_output,
                        compute_emd=compute_emd, noise_magnitude_added_to_gt=noise_magnitude_added_to_gt,
                        add_noise_to_generated_for_refine_exp=add_noise_to_generated_for_refine_exp,
                        return_all_metrics=True)
    if dataset=='shapenet':
        testloader.kill_data_processes()
    torch.cuda.empty_cache()
    save_dir = os.path.join(root_directory, local_path, 'eval_result')
    os.makedirs(save_dir, exist_ok=True)
    save_file = os.path.join(save_dir, 'eval_result_ckpt_%d_rank_%d.pkl' % (n_iter, rank))
    handle = open(save_file, 'wb')
    for key in metrics.keys():
        metrics[key] = metrics[key].detach().cpu().numpy()
    pickle.dump({'meta':meta, 
                    'cd_distance': metrics['cd_distance'], 
                    'emd_distance': metrics['emd_distance'],
                    'cd_p': metrics['cd_p'], 'f1': metrics['f1'],
                    'avg_cd':CD_loss, 'avg_emd':EMD_loss}, handle)
    handle.close()
    print('\nTestset evaluation result of the mini-batch samples from rank %d:' % rank)
    print('have saved eval result at iter %d for rank %d to %s' % (n_iter, rank, save_file))
    print("iteration: {} Testset \tCD loss: {} \tEMD loss: {}".format(n_iter, CD_loss, EMD_loss), flush=True)
    print("iteration: {} Testset \tcd_p: {} \tf1: {}".format(n_iter, 
                metrics['cd_p'].mean(), metrics['f1'].mean()), flush=True)

    if test_trainset:
        torch.cuda.empty_cache()
        testloader = get_dataloader(trainset_config, phase='test_trainset', rank=rank, world_size=num_gpus, 
                            random_subsample=True, num_samples=int(num_samples_tested_in_trainset/num_gpus),
                            append_samples_to_last_rank=False)
        with torch.no_grad():
            CD_loss, EMD_loss, meta, metrics = evaluate(net, testloader, 
                            diffusion_hyperparams, print_every_n_steps=200, parallel=False,
                            dataset=dataset, scale=scale, task=task, refine_output_scale_factor=refine_output_scale_factor,
                            max_print_nums=10, point_upsample_factor=point_upsample_factor,
                            include_displacement_center_to_final_output=include_displacement_center_to_final_output,
                            compute_emd=compute_emd, noise_magnitude_added_to_gt=noise_magnitude_added_to_gt,
                            add_noise_to_generated_for_refine_exp=add_noise_to_generated_for_refine_exp,
                            return_all_metrics=True)
        if dataset=='shapenet':
            testloader.kill_data_processes()
        torch.cuda.empty_cache()
        save_dir = os.path.join(root_directory, local_path, 'eval_result')
        os.makedirs(save_dir, exist_ok=True)
        save_file = os.path.join(save_dir, 'eval_result_ckpt_%d_rank_%d_trainset.pkl' % (n_iter, rank))
        handle = open(save_file, 'wb')
        for key in metrics.keys():
            metrics[key] = metrics[key].detach().cpu().numpy()
        pickle.dump({'meta':meta, 
                        'cd_distance': metrics['cd_distance'], 
                        'emd_distance': metrics['emd_distance'],
                        'cd_p': metrics['cd_p'], 'f1': metrics['f1'],
                        'avg_cd':CD_loss, 'avg_emd':EMD_loss}, handle)
        handle.close()
        print('\nTrainset evaluation result of the mini-batch samples from rank %d:' % rank)
        print('have saved eval result of trainset at iter %d for rank %d to %s' % (n_iter, rank, save_file))
        print("iteration: {} Trainset \tCD loss: {} \tEMD loss: {}".format(n_iter, CD_loss, EMD_loss), flush=True)
        print("iteration: {} Trainset \tcd_p: {} \tf1: {}".format(n_iter, 
                metrics['cd_p'].mean(), metrics['f1'].mean()), flush=True)
    net.train()

def gather_eval_result_from_all_ranks(save_dir, n_iter, num_gpus, tb=None, test_trainset=False, dataset ='shapenet'):
    if not dataset in ['shapenet', 'shapenet_pytorch', 'mvp_dataset', 'shapenet_chunk', 'mvp40', 'partnet']:
         raise Exception('%s dataset is not supported' % dataset)
    try:
        total_meta = []
        total_cd_distance = []
        total_emd_distance = []
        total_cd_p = []
        total_f1 = []
        files = [os.path.join(save_dir, 'eval_result_ckpt_%d_rank_%d.pkl' % (n_iter, r)) for r in range(num_gpus)]
        for f in files:
            handle = open(f, 'rb')
            saved_eval = pickle.load(handle)
            if dataset == 'shapenet' or dataset == 'shapenet_pytorch':
                total_meta = total_meta + saved_eval['meta']
            elif dataset in ['mvp_dataset', 'shapenet_chunk', 'mvp40', 'partnet']:
                total_meta.append(saved_eval['meta'])
            total_cd_distance.append(saved_eval['cd_distance'])
            total_emd_distance.append(saved_eval['emd_distance'])
            total_cd_p.append(saved_eval['cd_p'])
            total_f1.append(saved_eval['f1'])
            handle.close()
            os.remove(f)
        total_cd_distance = np.concatenate(total_cd_distance)
        total_emd_distance = np.concatenate(total_emd_distance)
        total_cd_p = np.concatenate(total_cd_p)
        total_f1 = np.concatenate(total_f1)
        if dataset in ['mvp_dataset', 'shapenet_chunk', 'mvp40', 'partnet']:
            total_meta = np.concatenate(total_meta)
        CD_loss_all = total_cd_distance.mean()
        EMD_loss_all = total_emd_distance.mean()
        CD_p_all = total_cd_p.mean()
        F1_all = total_f1.mean()

        testset_cd_loss = CD_loss_all
        testset_emd_loss = EMD_loss_all

        if not tb is None:
            tb.add_scalar("CD-Loss", CD_loss_all, n_iter)
            tb.add_scalar("EMD-Loss", EMD_loss_all, n_iter)
        print('\nGathered testset evaluation result from all ranks:')
        print("iteration: {} Testset \tGathered Avg CD loss: {} \tGathered Avg EMD loss: {}".format(
                                n_iter, CD_loss_all, EMD_loss_all), flush=True)
        print("iteration: {} Testset \tGathered Avg CD p: {} \tGathered Avg F1: {}".format(
                                n_iter, CD_p_all, F1_all), flush=True)
        save_file = os.path.join(save_dir, 'eval_result_ckpt_%d.pkl' % (n_iter))
        handle = open(save_file, 'wb')
        pickle.dump({'iter': n_iter, 'meta':total_meta, 'cd_distance':total_cd_distance, 
                    'emd_distance':total_emd_distance, 'avg_cd':CD_loss_all, 'avg_emd':EMD_loss_all,
                    'cd_p':total_cd_p, 'f1':total_f1, 'avg_cd_p':CD_p_all, 'avg_f1':F1_all}, handle)
        handle.close()
        print('have saved eval result at iter %d gathered from all process to %s' % (n_iter, save_file))
    except Exception as e:
        print('Some error happened during gathering results from all ranks')
        print('The error message is', str(e))
        save_file = os.path.join(save_dir, 'eval_result_ckpt_%d.pkl' % (n_iter))

    if dataset == 'shapenet' or dataset == 'shapenet_pytorch':
        get_each_category_distance(save_file)
        gathered_keys=['iter', 'avg_cd', 'avg_emd', 'cd_distance_category_split_result', 'emd_distance_category_split_result']
    elif dataset in ['mvp_dataset', 'shapenet_chunk', 'mvp40', 'partnet']:
        gathered_keys=['iter', 'avg_cd', 'avg_emd', 'avg_cd_p', 'avg_f1']
    gathered_results = gather_eval_result_of_different_iters(save_dir, match1='eval_result_ckpt', match2='.pkl', 
        nomatch = 'trainset', save_suffix = '', plot=True, split_category = False,
        gathered_keys=gathered_keys)

    if test_trainset:
        try:
            total_meta = []
            total_cd_distance = []
            total_emd_distance = []
            total_cd_p = []
            total_f1 = []
            files = [os.path.join(save_dir, 'eval_result_ckpt_%d_rank_%d_trainset.pkl' % (n_iter, r)) for r in range(num_gpus)]
            for f in files:
                handle = open(f, 'rb')
                saved_eval = pickle.load(handle)
                if dataset == 'shapenet' or dataset == 'shapenet_pytorch':
                    total_meta = total_meta + saved_eval['meta']
                elif dataset in ['mvp_dataset', 'shapenet_chunk', 'mvp40', 'partnet']:
                    total_meta.append(saved_eval['meta'])
                total_cd_distance.append(saved_eval['cd_distance'])
                total_emd_distance.append(saved_eval['emd_distance'])
                total_cd_p.append(saved_eval['cd_p'])
                total_f1.append(saved_eval['f1'])
                handle.close()
                os.remove(f)
            total_cd_distance = np.concatenate(total_cd_distance)
            total_emd_distance = np.concatenate(total_emd_distance)
            total_cd_p = np.concatenate(total_cd_p)
            total_f1 = np.concatenate(total_f1)
            if dataset in ['mvp_dataset', 'shapenet_chunk', 'mvp40', 'partnet']:
                total_meta = np.concatenate(total_meta)
            CD_loss_all = total_cd_distance.mean()
            EMD_loss_all = total_emd_distance.mean()
            CD_p_all = total_cd_p.mean()
            F1_all = total_f1.mean()
            if not tb is None:
                tb.add_scalar("Trainset CD-Loss", CD_loss_all, n_iter)
                tb.add_scalar("Trainset EMD-Loss", EMD_loss_all, n_iter)
            print('\nGathered trainset evaluation result from all ranks:')
            print("iteration: {} Trainset \tGathered Avg CD loss: {} \tGathered Avg EMD loss: {}".format(
                                    n_iter, CD_loss_all, EMD_loss_all), flush=True)
            print("iteration: {} Trainset \tGathered Avg CD p: {} \tGathered Avg F1: {}".format(
                                n_iter, CD_p_all, F1_all), flush=True)
            save_file = os.path.join(save_dir, 'eval_result_ckpt_%d_trainset.pkl' % (n_iter))
            handle = open(save_file, 'wb')
            pickle.dump({'iter': n_iter, 'meta':total_meta, 'cd_distance':total_cd_distance, 
                    'emd_distance':total_emd_distance, 'avg_cd':CD_loss_all, 'avg_emd':EMD_loss_all,
                    'cd_p':total_cd_p, 'f1':total_f1, 'avg_cd_p':CD_p_all, 'avg_f1':F1_all}, handle)
            handle.close()
            print('have saved eval result of trainset at iter %d gathered from all process to %s' % (n_iter, save_file))
        except Exception as e:
            print('Some error happened during gathering results from all ranks in training set')
            print('The error message is', str(e))
            save_file = os.path.join(save_dir, 'eval_result_ckpt_%d_trainset.pkl' % (n_iter))

        if dataset == 'shapenet' or dataset == 'shapenet_pytorch':
            get_each_category_distance(save_file)
            gathered_keys=['iter', 'avg_cd', 'avg_emd', 'cd_distance_category_split_result', 'emd_distance_category_split_result']
        elif dataset in ['mvp_dataset', 'shapenet_chunk', 'mvp40', 'partnet']:
            gathered_keys=['iter', 'avg_cd', 'avg_emd', 'avg_cd_p', 'avg_f1']
        gathered_results = gather_eval_result_of_different_iters(save_dir, match1='eval_result_ckpt', match2='trainset.pkl', 
        save_suffix = '_trainset', plot=True, split_category = False,
        gathered_keys = gathered_keys)
        print('\nFind the checkpoint that has the lowest CD t loss and EMD loss:')
        # plot testset and trainset figures in the same figure, and find the ckpt that has the lowest loss value
        plot_train_and_val_eval_result(save_dir)

    return testset_cd_loss, testset_emd_loss

def split_data(data, dataset, conditioned_on_cloud, include_class_condition, 
                random_shuffle_points, task = 'completion',
                random_subsample_partial_points=-1):
    generated = None
    condition = None
    if dataset in ['mvp_dataset', 'shapenet_chunk', 'mvp40', 'partnet']:
        label = data['label'].cuda()
        X = data['complete'].cuda()
        if task == 'refine_completion':
            generated = data['generated'].cuda() # coarse point clouds generated by the trained DDPM
        if conditioned_on_cloud:
            condition = data['partial'].cuda()
    else:
        raise Exception('%s dataset is not supported' % dataset)

    if random_shuffle_points:
        with torch.no_grad():
            num_points = X.shape[1]
            idx = torch.randperm(num_points)
            X = X[:,idx,:]
            if condition is not None:
                num_points = condition.shape[1]
                idx = torch.randperm(num_points)
                if random_subsample_partial_points > 1:
                    idx = idx[0:random_subsample_partial_points]
                condition = condition[:,idx,:]
            if generated is not None:
                num_points = generated.shape[1]
                idx = torch.randperm(num_points)
                generated = generated[:,idx,:]
                
    if task == 'refine_completion':
        return X, condition, label, generated
    else:
        return X, condition, label

def train(num_gpus, config_file, rank, group_name, dataset, root_directory, output_directory, 
          tensorboard_directory, ckpt_iter, n_epochs, epochs_per_ckpt, iters_per_logging,
          learning_rate, loss_type, conditioned_on_cloud, random_shuffle_points = True,
          eval_start_epoch = 0, eval_per_ckpt = 1, task='completion', only_save_the_best_model=False,
          compute_emd=True, split_dataset_to_multi_gpus=False,
          random_subsample_partial_points=-1):
    """
    Train the PointNet2SemSegSSG model on the 3D dataset

    Parameters:
    num_gpus, rank, group_name:     parameters for distributed training
    config_file:                    path to the config file
    output_directory (str):         save model checkpoints to this path
    tensorboard_directory (str):    save tensorboard events to this path
    ckpt_iter (int or 'max'):       the pretrained checkpoint to be loaded; 
                                    automitically selects the maximum iteration if 'max' is selected
    n_epochs (int):                 number of epochs to train
    epochs_per_ckpt (int):          number of epochs to save checkpoint
    iters_per_logging (int):        number of iterations to save training log and compute validation loss, default is 100
    learning_rate (float):          learning rate
    """
    assert task in ['completion', 'refine_completion']
    if task == 'completion' and only_save_the_best_model:
        raise Exception('To train the diffusion model, we should save every checkpoint')
    # generate experiment (local) path
    local_path = "T{}_betaT{}".format(diffusion_config["T"], diffusion_config["beta_T"])
    local_path = local_path + '_' + pointnet_config['model_name']
    if task == 'refine_completion':
        exp_name_split = list(os.path.split(refine_config['exp_name']))
        exp_name_split[-1] = 'refine_exp_' + exp_name_split[-1]
        exp_name = os.path.join(*exp_name_split)
        local_path = os.path.join(local_path, exp_name)

    # Get shared output_directory ready
    output_directory = os.path.join(root_directory, local_path, output_directory)
    if True:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        try:
            copyfile(config_file, os.path.join(output_directory, os.path.split(config_file)[1]))
        except:
            print('The two files are the same, no need to copy')
            
        print("output directory is", output_directory, flush=True)
        print("Config file has been copied from %s to %s" % (config_file, 
            os.path.join(output_directory, os.path.split(config_file)[1])), flush=True)
    
    # map diffusion hyperparameters to gpu
    for key in diffusion_hyperparams:
        if key != "T":
            diffusion_hyperparams[key] = diffusion_hyperparams[key].cuda()

    # load training data
    trainloader = get_dataloader(trainset_config)
    print('Data loaded')
    
    net = PointNet2CloudCondition(pointnet_config).cuda()
    net.train()
    print_size(net)

    # optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # load checkpoint model
    time0 = time.time()
    _, num_ckpts = find_max_epoch(output_directory, 'pointnet_ckpt', return_num_ckpts=True)
    # num_ckpts is number of ckpts found in the output_directory
    if ckpt_iter == 'max':
        ckpt_iter = find_max_epoch(output_directory, 'pointnet_ckpt')
    if ckpt_iter >= 0:
        try:
            # load checkpoint file
            model_path = os.path.join(output_directory, 'pointnet_ckpt_{}.pkl'.format(ckpt_iter))
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # feed model dict and optimizer state
            net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # record training time based on elapsed time
            time0 -= checkpoint['training_time_seconds']
            print('Model at iteration %s has been trained for %s seconds' % (ckpt_iter, checkpoint['training_time_seconds']))
            print('checkpoint model loaded successfully', flush=True)
        except:
            ckpt_iter = -1
            print('No valid checkpoint model found, start training from initialization.', flush=True)
    else:
        ckpt_iter = -1
        print('No valid checkpoint model found, start training from initialization.', flush=True)

    print(net)
    # training
    loader_len = len(trainloader)
    n_iters = int(loader_len * n_epochs) # number of total training steps 
    iters_per_ckpt = int(loader_len * epochs_per_ckpt) # save a ckpt every iters_per_ckpt steps
    n_iter = ckpt_iter + 1 # starting iter number
    eval_start_iter = eval_start_epoch *  loader_len - 1 
    # we start evaluating the trained model at least after eval_start_epoch steps
    
    if task == 'refine_completion':
        # refine ddpm generated coarse point clouds
        # loss_function = Chamfer_Loss()
        if refine_config.get('use_output_scale_factor_schedule', False):
            scale_factor_schedule = refine_config['output_scale_factor_schedule']
            output_scale_factor_scheduler = QuantityScheduler(scale_factor_schedule['init_epoch'], 
                    scale_factor_schedule['final_epoch'], scale_factor_schedule['init_value'], 
                    refine_config['output_scale_factor'], loader_len)
    else: # completion task, train the conditional generation DDPM
        loss_function = nn.MSELoss()

    last_saved_model = None
    last_saved_best_model = None
    best_cd = None

    # n_iter from 0 to n_iters if we train the model from sratch
    while n_iter < n_iters + 1:
        if trainset_config.get('randomly_select_generated_samples', False):
            # in this case, we are training the refinement network
            # we need to reload the dataset every epoch, 
            # we randomly select one trial from the multi-trial coarse point clouds generated by the trained DDPM
            if split_dataset_to_multi_gpus and num_gpus > 1:
                trainloader = get_dataloader(trainset_config, rank=rank, world_size=num_gpus, append_samples_to_last_rank=True)
            else:
                trainloader = get_dataloader(trainset_config)
        for data in trainloader: 
            # load data
            if task == 'refine_completion':
                X, condition, label, generated = split_data(data, dataset, conditioned_on_cloud, 
                                    pointnet_config.get('include_class_condition', False), 
                                    random_shuffle_points, task=task,
                                    random_subsample_partial_points=random_subsample_partial_points)
            else:
                #1.将数据放到cuda中 2.打乱数据  （看起来没什么用）
                X, condition, label = split_data(data, dataset, conditioned_on_cloud, 
                                    pointnet_config.get('include_class_condition', False), 
                                    random_shuffle_points, task=task,
                                    random_subsample_partial_points=random_subsample_partial_points)
            optimizer.zero_grad()
            
            scale = trainset_config['scale'] # scale of the shapes from the dataset

            if task == 'refine_completion':
                displacement = net(generated, condition, ts=None, label=label)
                if refine_config.get('use_output_scale_factor_schedule', False):
                    output_scale_factor_value = output_scale_factor_scheduler.get_quantity(n_iter)
                else:
                    output_scale_factor_value = refine_config['output_scale_factor']
                
                point_upsample_factor = pointnet_config.get('point_upsample_factor', 1)
                if point_upsample_factor > 1:
                    intermediate_refined_X_loss_weight = pointnet_config['intermediate_refined_X_loss_weight']
                    # refined_X is of shape (B, N*point_upsample_factor, 3)
                    refined_X, intermediate_refined_X = point_upsample(generated, displacement, point_upsample_factor, 
                                                        pointnet_config['include_displacement_center_to_final_output'],
                                                        output_scale_factor_value)
                else:
                    intermediate_refined_X_loss_weight = 0
                    refined_X = generated + displacement * output_scale_factor_value

                refined_X = refined_X / scale / 2
                X = X / scale / 2
                cd_loss_type = (refine_config.get('cd_loss_type', 'cd_t') if task == 'refine_completion' 
                                    else denoise_config['cd_loss_type'])
               
                if cd_loss_type == 'cd_t':
                    loss_idx = 1
                else:
                    loss_idx = 0
                loss = calc_cd(X, refined_X, calc_f1=False)[loss_idx]
                loss = loss.mean()
                if intermediate_refined_X_loss_weight > 0:
                    intermediate_refined_X_loss = calc_cd(X, intermediate_refined_X, calc_f1=False)[loss_idx]
                    loss = loss + intermediate_refined_X_loss.mean() * intermediate_refined_X_loss_weight

            else: # task = completion
                output_scale_factor_value = None
                loss = training_loss(net, loss_function, X, diffusion_hyperparams,
                                    label=label, condition=condition)
            
            reduced_loss = loss.item()
            loss.backward()
            optimizer.step()

            # output to log
            if n_iter % iters_per_logging == 0:
                if task in ['refine_completion', 'denoise']:
                    print("iteration: {} \treduced loss: {:.6f} \tloss: {:.6f} \toutput_scale_factor: {:.6f} \tdisplacement scale: [{:.6f}, {:.6f} | {:.6f}]".format(
                        n_iter, reduced_loss, loss.item(), output_scale_factor_value, 
                        displacement.min()*output_scale_factor_value, displacement.max()*output_scale_factor_value,
                        displacement.abs().mean()*output_scale_factor_value), flush=True)
                else: # completion task
                    print("iteration: {} \treduced loss: {:.6f} \tloss: {:.6f}".format(n_iter, reduced_loss, loss.item()), flush=True)
                
                tb = SummaryWriter('log_dir')
                tb.add_scalar("Log-Train-Loss", torch.log(loss).item(), n_iter)
                tb.add_scalar("Log-Train-Reduced-Loss", np.log(reduced_loss), n_iter)
                if task == 'refine_completion':
                    tb.add_scalar("output_scale_factor", output_scale_factor_value, n_iter)
            
            # save checkpoint
            if n_iter > 0 and (n_iter+1) % iters_per_ckpt == 0:
                num_ckpts = num_ckpts + 1
                # save checkpoint
                if last_saved_model is not None and only_save_the_best_model:
                    os.remove(last_saved_model)
                checkpoint_name = 'pointnet_ckpt_{}.pkl'.format(n_iter)
                torch.save({'iter': n_iter,
                            'model_state_dict': net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'training_time_seconds': int(time.time()-time0)}, 
                            os.path.join(output_directory, checkpoint_name))
                print('model at iteration %s at epoch %d is saved' % (n_iter, int((n_iter+1)/loader_len)), flush=True)
                last_saved_model = os.path.join(output_directory, checkpoint_name)

                # evaluate the model at the checkpoint
                if n_iter >= eval_start_iter and num_ckpts % eval_per_ckpt==0:
                    test_trainset_during_eval = trainset_config.get('test_trainset_during_eval', False)
                    if dataset in ['shapenet', 'shapenet_pytorch', 'shapenet_chunk', 'partnet']:
                        # these datasets have a small testset, therefore we could test the whole test set during training
                        # but they have a large training set, we only test num_samples_tested_in_trainset samples for the training set
                        num_samples_tested_in_trainset = trainset_config.get('num_samples_tested_in_trainset', 0)
                    elif dataset == 'mvp_dataset' or dataset == 'mvp40':
                        # these two datasets have too many samples even in the test set
                        # we only evalute num_samples_tested for both the training set and test set
                        num_samples_tested_in_trainset = trainset_config['num_samples_tested']
                    else:
                        raise Exception('%s dataset is not supported' % dataset)
                    
                    add_noise_to_generated_for_refine_exp = False
                    if task == 'refine_completion':
                        add_noise_to_generated_for_refine_exp = refine_config.get('add_noise_to_generated_for_refine_exp', False)
                        noise_magnitude_added_to_gt = trainset_config['augmentation']['noise_magnitude_for_generated_samples']
                    else:
                        noise_magnitude_added_to_gt = 0
                    print('\nBegin evaluting the saved checkpoint')
                    evaluate_per_rank(net, rank, num_gpus, root_directory, local_path, n_iter,
                            test_trainset=test_trainset_during_eval, num_samples_tested_in_trainset=num_samples_tested_in_trainset,
                            dataset=dataset, scale=scale, task=task, refine_output_scale_factor=output_scale_factor_value,
                            compute_emd=compute_emd, noise_magnitude_added_to_gt=noise_magnitude_added_to_gt,
                            add_noise_to_generated_for_refine_exp=add_noise_to_generated_for_refine_exp)
                    torch.distributed.barrier()

                    # save the best model so far
                    if rank == 0:
                        testset_cd_loss, testset_emd_loss = gather_eval_result_from_all_ranks(os.path.join(root_directory, local_path, 'eval_result'), 
                                            n_iter, num_gpus, tb, test_trainset=test_trainset_during_eval, dataset=dataset)
                        testset_cd_loss_tensor = torch.Tensor([testset_cd_loss]).cuda()
                        if only_save_the_best_model:
                            if best_cd is None or testset_cd_loss <= best_cd:
                                if last_saved_best_model is not None:
                                    os.remove(last_saved_best_model)
                                best_cd = testset_cd_loss
                                best_checkpoint_name = 'pointnet_ckpt_{}_best_cd.pkl'.format(n_iter)
                                torch.save({'iter': n_iter,
                                    'model_state_dict': net.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict(),
                                    'training_time_seconds': int(time.time()-time0),
                                    'testset_cd_loss':testset_cd_loss,
                                    'testset_emd_loss':testset_emd_loss}, 
                                    os.path.join(output_directory, best_checkpoint_name))
                                last_saved_best_model = os.path.join(output_directory, best_checkpoint_name)
                    else:
                        testset_cd_loss_tensor = torch.Tensor([0]).cuda()

                    print('Have finished evaluting the saved checkpoint\n')
                    torch.distributed.barrier()
                    
                    if task == 'refine_completion':
                        torch.distributed.broadcast(testset_cd_loss_tensor, 0)
                        # let other ranks know the current testset_cd_loss
                        print('The broadcasted tensor is', testset_cd_loss_tensor)
                        if refine_config.get('decrease_epochs_per_ckpt_for_fine_tuning', False):
                            # if the current testset_cd_loss is less than a specifed threo
                            # we reset the iters_per_ckpt to a smaller number
                            # we save and evaluate ckpts more frequently because we are close to convergence
                            if testset_cd_loss_tensor.cpu().item() <= refine_config['cd_loss_thred']:
                                iters_per_ckpt = int(loader_len * refine_config['epochs_per_ckpt_fine_tune'])



            n_iter += 1
    


if __name__ == "__main__":
    # import pdb
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='./exp_configs/mvp_configs/config_standard_attention_real_3072_partial_points_rot_90_scale_1.2_translation_0.1.json', 
                        help='JSON file for configuration')
    args = parser.parse_args()

    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    config = restore_string_to_list_in_a_dict(config)
    print('The configuration is:')
    print(json.dumps(replace_list_with_string_in_a_dict(copy.deepcopy(config)), indent=4))
    global train_config
    train_config = config["train_config"]        # training parameters
    global dist_config
    dist_config = config["dist_config"]         # to initialize distributed training
    global pointnet_config
    pointnet_config = config["pointnet_config"]     # to define pointnet
    global diffusion_config
    diffusion_config = config["diffusion_config"]    # basic hyperparameters
    
    global trainset_config
    if train_config['dataset'] == 'mvp_dataset':
        trainset_config = config["mvp_dataset_config"]
    elif train_config['dataset'] == 'shapenet_chunk':
        trainset_config = config['shapenet_chunk_config']
    elif train_config['dataset'] == 'mvp40':
        trainset_config = config['mvp40_config']
    elif train_config['dataset'] == 'partnet':
        trainset_config = config['partnet_config']
    else:
        raise Exception('%s dataset is not supported' % train_config['dataset'])

    global diffusion_hyperparams 
    diffusion_hyperparams = calc_diffusion_hyperparams(**diffusion_config)  # dictionary of all diffusion hyperparameters
    
    if train_config['task'] == 'refine_completion':
        global refine_config
        refine_config = config['refine_config']
        for key in refine_config.keys():
            if key in train_config.keys():
                train_config[key] = refine_config[key]
            if key in trainset_config.keys():
                trainset_config[key] = refine_config[key]
        if 'randomly_select_generated_samples' in refine_config.keys():
            trainset_config['randomly_select_generated_samples'] = refine_config['randomly_select_generated_samples']
        pointnet_config['include_t'] = False

    
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    train(args.config, **train_config)
