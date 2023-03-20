import os
import time
import argparse
import json
import wandb
import numpy as np
import torch
import torch.nn as nn
import copy

from mvp_dataloader.mvp_dataset import ShapeNetH5
from util import find_max_epoch, print_size
from util import training_loss, calc_diffusion_hyperparams
from models.pointnet2_with_pcld_condition import PointNet2CloudCondition  #net
from shutil import copyfile



def train(config_file, dataset, root_directory, output_directory, continue_ckpts, 
          ckpt_epoch, n_epochs, epochs_per_ckpt,
          learning_rate, only_save_the_best_model=False):
    """
    Train the PointNet2SemSegSSG model on the 3D dataset

    Parameters:
    config_file(str):               path to the config file
    dataset(str):                   mvp_dataset, the name of the dataset
    root_directory(str):            path of the root directory
    output_directory (str):         save model checkpoints to this path
    continue_ckpts(bool):           if continue train
    ckpt_epoch (int or 'max'):      the pretrained checkpoint to be loaded; 
                                    automitically selects the maximum epoch if 'max' is selected
    n_epochs (int):                 number of epochs to train
    epochs_per_ckpt (int):          number of epochs to save checkpoint
    learning_rate (float):          learning rate
    only_save_the_best_model(bool): save only the best model
    """
    local_path = "T{}_betaT{}".format(diffusion_config["T"], diffusion_config["beta_T"])
    local_path = local_path + '_' + pointnet_config['model_name']
    

    output_directory = os.path.join(root_directory, local_path, output_directory)
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
    dataset = ShapeNetH5(**trainset_config)
    print('Data loaded')
    
    net = PointNet2CloudCondition(pointnet_config).cuda()
    net.train()
    print_size(net)

    # optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # load checkpoint model
    time0 = time.time()
    if continue_ckpts:
        if ckpt_epoch == 'max':
            ckpt_epoch = find_max_epoch(output_directory, 'pointnet_ckpt')
            try:
                # load checkpoint file
                model_path = os.path.join(output_directory, 'pointnet_ckpt_{}.pkl'.format(ckpt_epoch))
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

    print(net)
    # training
    n_epoch = ckpt_epoch + 1 # starting cpoch number
    
    loss_function = nn.MSELoss()

    last_saved_model = None

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)
    while n_epoch < n_epochs + 1:
        for i, data in enumerate(dataloader):
            label, condition, X = data['label'], data['partial'], data['complete']
            optimizer.zero_grad()
            loss = training_loss(net, loss_function, X, diffusion_hyperparams,
                                label=label, condition=condition)
            loss.backward()
            optimizer.step()
            wandb.log({"loss": loss.item(), "epoch": n_epoch})

            # save checkpoint
            if n_epoch > 0 and (n_epoch+1) % epochs_per_ckpt == 0:
                # save checkpoint
                if last_saved_model is not None and only_save_the_best_model:
                    os.remove(last_saved_model)
                checkpoint_name = 'pointnet_ckpt_{}.pkl'.format(n_epoch)
                torch.save({'iter': n_epoch,
                            'model_state_dict': net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'training_time_seconds': int(time.time()-time0)}, 
                            os.path.join(output_directory, checkpoint_name))
                print(('model at epoch %s is saved' %n_epoch), flush=True)
                last_saved_model = os.path.join(output_directory, checkpoint_name)

                

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, 
                        default='./exp_configs/mvp_configs/config_standard_attention_real_3072_partial_points_rot_90_scale_1.2_translation_0.1.json', 
                        help='JSON file for configuration')
    args = parser.parse_args()

    wandb.login()
    run = wandb.init(project="mirror_500")
    path = os.getcwd()
    config_path = os.path.join(path,'pointnet2', args.config)
    with open(config_path,'r') as f:
        json_config = json.load(f)
    wandb.init(config=json_config)

    global train_config
    train_config = wandb.config["train_config"]        
    global pointnet_config
    pointnet_config = wandb.config["pointnet_config"]     
    global diffusion_config
    diffusion_config = wandb.config["diffusion_config"]   
    global trainset_config
    trainset_config = wandb.config["mvp_dataset_config"]
    global diffusion_hyperparams 
    diffusion_hyperparams = calc_diffusion_hyperparams(**diffusion_config)  # dictionary of all diffusion hyperparameters
    
    
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    train(args.config, **train_config)
