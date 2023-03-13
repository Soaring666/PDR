trainloader = get_dataloader(trainset_config)
for data in trainloader: 
    X, condition, label = split_data(data, dataset, conditioned_on_cloud, 
                        pointnet_config.get('include_class_condition', False), 
                        random_shuffle_points, task=task,
                        random_subsample_partial_points=random_subsample_partial_points)
    optimizer.zero_grad()
    scale = trainset_config['scale'] # scale of the shapes from the dataset

# task = completion, train the conditional generation DDPM
output_scale_factor_value = None
loss = training_loss(net, loss_function, X, diffusion_hyperparams,
                    label=label, condition=condition)
loss.backward()
optimizer.step()