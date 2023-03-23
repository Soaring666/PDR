import os
import time
import argparse
import json
import wandb
import numpy as np
import torch
import torch.nn as nn
import copy
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP
from mvp_dataloader.mvp_dataset import ShapeNetH5
from util import find_max_epoch, print_size
from util import training_loss, calc_diffusion_hyperparams
from models.pointnet2_with_pcld_condition import PointNet2CloudCondition  #net
from evaluate import evaluate  #evaluate function
from shutil import copyfile
from tqdm import tqdm



def train(config_file, dataset, root_directory, checkpoint_directory, continue_ckpts, 
          ckpt_epoch, n_epochs, epochs_per_ckpt, batch_size, 
          learning_rate, value):
    """
    Train the PointNet2SemSegSSG model on the 3D dataset

    Parameters:
    config_file(str):               path to the config file
    dataset(str):                   mvp_dataset, the name of the dataset
    root_directory(str):            path of the root directory
    checkpoint_directory (str):         save model checkpoints to this path
    continue_ckpts(bool):           if continue train
    ckpt_epoch (int or 'max'):      the pretrained checkpoint to be loaded; 
                                    automitically selects the maximum epoch if 'max' is selected
    n_epochs (int):                 number of epochs to train
    epochs_per_ckpt (int):          number of epochs to save checkpoint
    batch_size(int):                batch_size
    learning_rate (float):          learning rate
    value(bool):                    if value during training
    """
    dist.init_process_group(backend='nccl')   
    torch.cuda.set_device(args.local_rank)

    local_path = "T{}_betaT{}".format(diffusion_config["T"], diffusion_config["beta_T"])
    local_path = local_path + '_' + pointnet_config['model_name']
    

    checkpoint_directory = os.path.join(root_directory, local_path, checkpoint_directory)
    
    if not os.path.isdir(checkpoint_directory):
        os.makedirs(checkpoint_directory)
        os.chmod(checkpoint_directory, 0o775)
   
        
    print("checkpoint directory is", checkpoint_directory, flush=True)

    
    # map diffusion hyperparameters to gpu
    for key in diffusion_hyperparams:
        if key != "T":
            diffusion_hyperparams[key] = diffusion_hyperparams[key].cuda()

    # load training data
    train_dataset = ShapeNetH5(**trainset_config)
    test_dataset = ShapeNetH5(**testset_config)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
    train_value_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=False)
    print('Data loaded')

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=train_sampler, num_workers=4, drop_last=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, sampler=test_sampler, num_workers=4, drop_last=True)
    train_value_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=train_value_sampler, num_workers=4, drop_last=True)
    
    device = torch.device("cuda", args.local_rank)
    net = PointNet2CloudCondition(pointnet_config).to(device)
    net = DDP(net, device_ids=[args.local_rank])
    print_size(net)

    # optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # load checkpoint model
    time0 = time.time()
    if continue_ckpts:
        if ckpt_epoch == 'max':
            ckpt_epoch = find_max_epoch(checkpoint_directory, 'pointnet_ckpt')
            try:
                # load checkpoint file
                model_path = os.path.join(checkpoint_directory, 'pointnet_ckpt_{}.pkl'.format(ckpt_epoch))
                checkpoint = torch.load(model_path, map_location='cpu')
                
                # feed model dict and optimizer state
                net.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

                # record training time based on elapsed time
                time0 -= checkpoint['training_time_seconds']
                print('Model at iteration %s has been trained for %s seconds' % (ckpt_epoch, checkpoint['training_time_seconds']))
                print('checkpoint model loaded successfully', flush=True)
            except:
                ckpt_epoch = -1
                print('No valid checkpoint model found, start training from initialization.', flush=True)
    else:
        ckpt_epoch = -1
        print('No valid checkpoint model found, start training from initialization.', flush=True)

    # training
    n_epoch = ckpt_epoch + 1 # starting cpoch number
    
    loss_function = nn.MSELoss()

    while n_epoch < n_epochs + 1:
        net.train()
        sum_loss = 0
        for data in tqdm(train_dataloader):
            label, condition, X = data['label'], data['partial'], data['complete']
            label = label.to(device)
            condition = condition.to(device)
            X = X.to(device)
            optimizer.zero_grad()
            loss = training_loss(net, loss_function, X, diffusion_hyperparams,
                                label=label, condition=condition)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()

            # save checkpoint
            if n_epoch > 0 and (n_epoch+1) % epochs_per_ckpt == 0:
                # save checkpoint
                checkpoint_name = 'pointnet_ckpt_{}.pkl'.format(n_epoch)
                if dist.get_rank() == 0:
                    torch.save({'iter': n_epoch,
                                'model_state_dict': net.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'training_time_seconds': int(time.time()-time0)}, 
                                os.path.join(checkpoint_directory, checkpoint_name))
                    print(('model at epoch %s is saved' %n_epoch), flush=True)
        average_loss = sum_loss / (len(train_dataloader) * batch_size)
        wandb.log({"loss": average_loss, "epoch": n_epoch})
        n_epoch += 1
        
        #value
        if value:
            net.eval()
            if n_epoch % 10 ==0:
                print('\nBegin evaluting the saved checkpoint')

                with torch.no_grad():
                    CD_sum = 0
                    F1_sum = 0
                    for i, data in tqdm(enumerate(test_dataloader)):
                        label = data['label'].to(device)
                        condition = data['partial'].to(device)
                        gt = data['complete'].to(device)
                        size = gt.shape()
                        if i == 10:
                            CD_test_loss, F1_test_loss, _, _ = evaluate(net, batch_size, size, diffusion_hyperparams,
                                                                            label, condition, gt, n_epoch, local_path, save_slices=True)
                        else:
                            CD_test_loss, F1_test_loss, _, _ = evaluate(net, batch_size, size, diffusion_hyperparams,
                                                                            label, condition, gt, n_epoch, local_path, save_slices=False)
                        CD_sum += CD_test_loss
                        F1_sum += F1_test_loss
                        wandb.log({"CD_test_loss": CD_test_loss/len(test_dataloader),
                                   "F1_test_loss": F1_test_loss/len(test_dataloader)})

                    CD_sum = 0
                    F1_sum = 0
                    for i, data in tqdm(enumerate(train_value_dataloader)):
                        label = data['label'].to(device)
                        condition = data['partial'].to(device)
                        gt = data['complete'].to(device)
                        size = gt.shape()
                        if i == 10:
                            CD_train_loss, F1_train_loss, _, _ = evaluate(net, batch_size, size, diffusion_hyperparams,
                                                                            label, condition, gt, n_epoch, local_path, save_slices=True)
                        else:
                            CD_train_loss, F1_train_loss, _, _ = evaluate(net, batch_size, size, diffusion_hyperparams,
                                                                            label, condition, gt, n_epoch, local_path, save_slices=False)
                        CD_sum += CD_train_loss
                        F1_sum += F1_train_loss
                        wandb.log({"CD_train_loss": CD_train_loss/len(train_value_dataloader),
                                   "F1_train_loss": F1_train_loss/len(train_value_dataloader)})



                

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, 
                        default='exp_configs/mvp_configs/config_standard_attention_real_3072_partial_points_rot_90_scale_1.2_translation_0.1.json',
                        help='JSON file for configuration')
    parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')
    args = parser.parse_args()
    path = os.getcwd()
    config_path = os.path.join(path,'pointnet2', args.config)
    with open(config_path,'r') as f:
        json_config = json.load(f)

    wandb.login()
    run = wandb.init(project="mirror_train", config=json_config)

    global train_config
    train_config = wandb.config["train_config"]        
    global pointnet_config
    pointnet_config = wandb.config["pointnet_config"]     
    global diffusion_config
    diffusion_config = wandb.config["diffusion_config"]   
    global trainset_config
    trainset_config = wandb.config["mvp_dataset_config"]
    global testset_config
    testset_config = copy.deepcopy(trainset_config)
    testset_config["train"] = False
    global diffusion_hyperparams 
    diffusion_hyperparams = calc_diffusion_hyperparams(**diffusion_config)  # dictionary of all diffusion hyperparameters
    
    
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    train(args.config, args.multigpu, **train_config)
