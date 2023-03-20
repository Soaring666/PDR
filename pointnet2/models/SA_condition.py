





class build_SA_model(nn.Module):
    def __init__(self, config):
        super(self).__init__()
        npoint_condition = config['npoint']#[1024, 256, 64, 16]
        radius_condition = config['radius']#np.array([0.1, 0.2, 0.4, 0.8])
        nsample_condition = config['nsample']#[32, 32, 32, 32]
        feature_dim_condition = config['feature_dim']#[32, 32, 64, 64, 128]
        mlp_depth_condition = config['mlp_depth']#3
        neighbor_def=config['neighbor_definition']#"radius"
        #activation=relu, bn=True, attention_setting=dictionary
    
    def forward(self):
        pass



# build SA module for condition point cloud
condition_arch = self.hparams['condition_net_architecture']

SA_modules_condition = build_SA_model(condition_arch)

