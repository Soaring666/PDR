import os
import h5py
import torch
from chamfer_loss_new import Chamfer_F1
from util import sampling, AverageMeter


def evaluate(net, batch_size, size, diffusion_hyperparams,
                  label, condition, gt, n_epoch, local_path, save_slices):
    """
    input:
        batch_size: the value batch size
        size: the size of generater samples
        n_epoch: the number of epoch
        local_path: the path to save generater samples
        save_slices(bool): whether to save generater samples
    output:
        CD_meter: CD loss
        F1_meter: F1 loss
        label_data: label data
        metric: other metrics

    """
    CD_meter = AverageMeter()
    F1_meter = AverageMeter()


    label_data = torch.rand(0).cuda().long()  #初始化一个张量，内容为空
    metrics = {'cd_distance': torch.rand(0).cuda(), 
               'cd_p': torch.rand(0).cuda(), 
               'f1': torch.rand(0).cuda()}

    f1_threshold = 0.0001
    cd_module = Chamfer_F1(f1_threshold=f1_threshold)      

    if save_slices:
        generated_data, result_slices = sampling(net, size, diffusion_hyperparams, 
                                label=label, condition=condition,
                                save_slices=True)

        #save the generated data
        save_file = os.path.join(local_path, 'mvp_generated_data_2048pts_epoch%d.h5' % n_epoch)
        hf = h5py.File(save_file, 'w')
        hf.create_dataset('data', data=result_slices)
        hf.close()

    else:
        generated_data = sampling(net, size, diffusion_hyperparams, 
                                label=label, condition=condition,
                                save_slices=False)


    cd_p, dist, f1 = cd_module(generated_data, gt)
    cd_loss = dist.mean().detach().cpu().item()
    f1_loss = f1.mean().detach().cpu().item()
    metrics['cd_distance'] = torch.cat([metrics['cd_distance'], dist])
    metrics['cd_p'] = torch.cat([metrics['cd_p'], cd_p])
    metrics['f1'] = torch.cat([metrics['f1'], f1])

    CD_meter.update(cd_loss, n=batch_size)
    F1_meter.update(f1_loss, n=batch_size)
    label_data = torch.cat([label_data, label])
    label_data = label_data.detach().cpu().numpy()

    return CD_meter.avg, F1_meter.avg, label_data, metrics
