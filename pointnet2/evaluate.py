

from models.point_upsample_module import point_upsample  #refine
from scheduler import QuantityScheduler
from chamfer_loss_new import calc_cd
from completion_eval import evaluate, get_each_category_distance, gather_eval_result_of_different_iters, plot_train_and_val_eval_result
import pickle







# evaluate the model at the checkpoint
def evaluate_per_rank(net, root_directory, local_path, n_epoch, test_trainset_during_eval=False, 
                    dataset='mvp_dataset', scale=1,
                    task = 'completion', refine_output_scale_factor=None, compute_emd=True,
                    noise_magnitude_added_to_gt=0, add_noise_to_generated_for_refine_exp=False):
    net.eval()
    test_dataset = ShapeNetH5(**trainset_config)
    test_dataloader = torch.utils.data.DataLoader(test_dataset)
    

    point_upsample_factor = pointnet_config.get('point_upsample_factor', 1)  #1
    include_displacement_center_to_final_output = pointnet_config.get('include_displacement_center_to_final_output', False)  #False
    
    with torch.no_grad():
        CD_loss, EMD_loss, meta, metrics = evaluate(net, test_dataloader, 
                        diffusion_hyperparams, 
                        dataset=dataset, scale=scale, task=task, refine_output_scale_factor=refine_output_scale_factor,
                        max_print_nums=10, point_upsample_factor=point_upsample_factor,
                        include_displacement_center_to_final_output=include_displacement_center_to_final_output,
                        noise_magnitude_added_to_gt=noise_magnitude_added_to_gt,
                        add_noise_to_generated_for_refine_exp=add_noise_to_generated_for_refine_exp,
                        )
    torch.cuda.empty_cache()
    save_dir = os.path.join(root_directory, local_path, 'eval_result')
    os.makedirs(save_dir, exist_ok=True)
    save_file = os.path.join(save_dir, 'eval_result_ckpt_%d.pkl' % n_epoch)
    handle = open(save_file, 'wb')
    for key in metrics.keys():
        metrics[key] = metrics[key].detach().cpu().numpy()
    pickle.dump({'meta':meta, 
                    'cd_distance': metrics['cd_distance'], 
                    'emd_distance': metrics['emd_distance'],
                    'cd_p': metrics['cd_p'], 'f1': metrics['f1'],
                    'avg_cd':CD_loss, 'avg_emd':EMD_loss}, handle)  #序列化字典并写入handle中
    handle.close()
    print('have saved eval result at epoch %d' % n_epoch)
    print("epoch: {} Testset \tCD loss: {} \tEMD loss: {}".format(n_epoch, CD_loss, EMD_loss), flush=True)
    print("epoch: {} Testset \tcd_p: {} \tf1: {}".format(n_epoch, 
                metrics['cd_p'].mean(), metrics['f1'].mean()), flush=True)

    if test_trainset_during_eval:

        test_dataset = ShapeNetH5(**trainset_config)
        test_dataloader = torch.utils.data.DataLoader(test_dataset)
        
        with torch.no_grad():
            CD_loss, EMD_loss, meta, metrics = evaluate(net, test_dataloader, 
                            diffusion_hyperparams, print_every_n_steps=200, parallel=False,
                            dataset=dataset, scale=scale, task=task, refine_output_scale_factor=refine_output_scale_factor,
                            max_print_nums=10, point_upsample_factor=point_upsample_factor,
                            include_displacement_center_to_final_output=include_displacement_center_to_final_output,
                            compute_emd=compute_emd, noise_magnitude_added_to_gt=noise_magnitude_added_to_gt,
                            add_noise_to_generated_for_refine_exp=add_noise_to_generated_for_refine_exp,
                            return_all_metrics=True)
        torch.cuda.empty_cache()
        save_dir = os.path.join(root_directory, local_path, 'eval_result')
        os.makedirs(save_dir, exist_ok=True)
        save_file = os.path.join(save_dir, 'eval_result_ckpt_%d_trainset.pkl' % n_epoch)
        handle = open(save_file, 'wb')
        for key in metrics.keys():
            metrics[key] = metrics[key].detach().cpu().numpy()
        pickle.dump({'meta':meta, 
                        'cd_distance': metrics['cd_distance'], 
                        'emd_distance': metrics['emd_distance'],
                        'cd_p': metrics['cd_p'], 'f1': metrics['f1'],
                        'avg_cd':CD_loss, 'avg_emd':EMD_loss}, handle)
        handle.close()
        print('have saved eval result of trainset at epoch %d' % n_epoch)
        print("epoch: {} Trainset \tCD loss: {} \tEMD loss: {}".format(n_epoch, CD_loss, EMD_loss), flush=True)
        print("epoch: {} Trainset \tcd_p: {} \tf1: {}".format(n_epoch, 
                metrics['cd_p'].mean(), metrics['f1'].mean()), flush=True)
    net.train()

def gather_eval_result_from_all_ranks(save_dir, n_iter, num_gpus, tb=None, test_trainset=False, dataset ='mvp_dataset'):
    try:
        total_meta = []
        total_cd_distance = []
        total_emd_distance = []
        total_cd_p = []
        total_f1 = []
        files = [os.path.join(save_dir, 'eval_result_ckpt_%d.pkl' % n_epoch)]
        for f in files:
            handle = open(f, 'rb')
            saved_eval = pickle.load(handle)
            if dataset in ['mvp_dataset', 'shapenet_chunk', 'mvp40', 'partnet']:
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

    if dataset in ['mvp_dataset', 'shapenet_chunk', 'mvp40', 'partnet']:
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









                if n_epoch % eval_per_epoch==0:
                    test_trainset_during_eval = trainset_config.get('test_trainset_during_eval', False)  #True
                    if dataset == 'mvp_dataset' or dataset == 'mvp40':
                        # we only evalute num_samples_tested for both the training set and test set
                        num_samples_tested_in_trainset = trainset_config['num_samples_tested']  #1600
                    else:
                        raise Exception('%s dataset is not supported' % dataset)
                    
                    add_noise_to_generated_for_refine_exp = False
                    if task == 'refine_completion':
                        add_noise_to_generated_for_refine_exp = refine_config.get('add_noise_to_generated_for_refine_exp', False)
                        noise_magnitude_added_to_gt = trainset_config['augmentation']['noise_magnitude_for_generated_samples']
                    else:
                        noise_magnitude_added_to_gt = 0
                    print('\nBegin evaluting the saved checkpoint')
                    #将测出来的指标保存在文件夹中
                    evaluate_per_rank(net, rank, num_gpus, root_directory, local_path, n_iter,
                            test_trainset_during_eval=test_trainset_during_eval, num_samples_tested_in_trainset=num_samples_tested_in_trainset,
                            dataset=dataset, scale=scale, task=task, refine_output_scale_factor=output_scale_factor_value,
                            compute_emd=compute_emd, noise_magnitude_added_to_gt=noise_magnitude_added_to_gt,
                            add_noise_to_generated_for_refine_exp=add_noise_to_generated_for_refine_exp)

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
    
