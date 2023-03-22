#nothing changed
import torch
import torch.nn as nn
from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule, FeatureMapModule
from pointnet2_ops.pointnet2_utils import QueryAndGroup
from torch.utils.data import DataLoader

from models.pointnet2_ssg_sem import PointNet2SemSegSSG, calc_t_emb, swish
from models.pnet import Pnet2Stage
from models.model_utils import get_embedder

# from pointnet2_ssg_sem import PointNet2SemSegSSG, calc_t_emb, swish
# from pnet import Pnet2Stage
# from model_utils import get_embedder

import copy
import numpy as np

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return swish(x)

class PointNet2CloudCondition(PointNet2SemSegSSG):

    def _build_model(self):
        self.l_uvw = None 
        # a list that stores xyz positions of the condition point cloud in different levels of the pointnet++ 
        self.encoder_cond_features = None
        # a list that stores features of the condition point cloud in different levels of encoder part of the pointnet++ 
        self.decoder_cond_features = None
        # a list that stores features of the condition point cloud in different levels of decoder part of the pointnet++ 
        self.global_feature = None
        # global feature extracted from the condition point cloud

        self.attention_setting = self.hparams.get("attention_setting", None)
        # FeatureMapper refers to the feature transfer module
        self.FeatureMapper_attention_setting = None    #没用的参数

        self.global_attention_setting = self.hparams.get('global_attention_setting', None)

        self.bn = self.hparams.get("bn", True) # bn here refers to group norm
        self.scale_factor = 1 # has no actual use, we can ignore this parameter
        self.record_neighbor_stats = self.hparams["record_neighbor_stats"]
        if self.hparams["include_class_condition"]:
            # utilize the class information of the partial point cloud
            self.class_emb = nn.Embedding(self.hparams["num_class"], self.hparams["class_condition_dim"])
        
        in_fea_dim = self.hparams['in_fea_dim']  #0
        partial_in_fea_dim = self.hparams.get('partial_in_fea_dim', in_fea_dim)  #1
        self.attach_position_to_input_feature = self.hparams['attach_position_to_input_feature']  #True
        if self.attach_position_to_input_feature:
            in_fea_dim = in_fea_dim + 3  #3
            partial_in_fea_dim = partial_in_fea_dim + 3  #4
        
        self.partial_in_fea_dim = partial_in_fea_dim

        self.include_abs_coordinate = self.hparams['include_abs_coordinate']
        self.pooling = self.hparams.get('pooling', 'max')
        # pooling should be max, avg or avg_max
        # pooling will have no effect and will not be used if self.attention_setting.use_attention_module
        # we will use attention mechanism to aggregate features instead of pooling 

        self.network_activation = self.hparams.get('activation', 'relu')
        assert self.network_activation in ['relu', 'swish']
        if self.network_activation == 'relu':
            self.network_activation_function = nn.ReLU(True)
        elif self.network_activation == 'swish':
            self.network_activation_function = Swish()

        self.include_local_feature = self.hparams.get('include_local_feature', True)   #True
        # whether to use multi-level local features from the condition point cloud to guide the diffusion model
        self.include_global_feature = self.hparams.get('include_global_feature', False)   #True
        # whether to use the global feature from the condition point cloud to guide the diffusion model

        self.global_feature_dim = None
        remove_last_activation = self.hparams.get('global_feature_remove_last_activation', True)  #False
        if self.include_global_feature:
            self.global_feature_dim = self.hparams['pnet_global_feature_architecture'][1][-1]  #1024
            self.global_pnet = Pnet2Stage(self.hparams['pnet_global_feature_architecture'][0],
                                            self.hparams['pnet_global_feature_architecture'][1],
                                            bn=self.bn, remove_last_activation=remove_last_activation)

        # time step embedding setting
        # we should include t in the conditional generation network
        # not include t in th refinement network
        include_t = self.hparams['include_t']
        t_dim = self.hparams['t_dim']
        # timestep embedding fc layers
        self.fc_t1 = nn.Linear(t_dim, 4*t_dim)
        self.fc_t2 = nn.Linear(4*t_dim, 4*t_dim)
        self.activation = swish # activation function for t embedding

        
        if self.include_local_feature:
            # build SA module for condition point cloud
            condition_arch = self.hparams['condition_net_architecture']
            npoint_condition = condition_arch['npoint']#[1024, 256, 64, 16]
            radius_condition = condition_arch['radius']#np.array([0.1, 0.2, 0.4, 0.8])
            nsample_condition = condition_arch['nsample']#[32, 32, 32, 32]
            feature_dim_condition = condition_arch['feature_dim']#[32, 32, 64, 64, 128]
            mlp_depth_condition = condition_arch['mlp_depth']#3
            neighbor_def=condition_arch['neighbor_definition']#"radius"
            #activation=relu, bn=True, attention_setting=dictionary
            self.SA_modules_condition = self.build_SA_model(npoint_condition, radius_condition, 
                                    nsample_condition, feature_dim_condition, mlp_depth_condition, partial_in_fea_dim, False, False,
                                    neighbor_def=neighbor_def,
                                    activation=self.network_activation, bn=self.bn,
                                    attention_setting=self.attention_setting)
        

            # build feature transfer modules from condition point cloud to the noisy point cloud x_t at encoder
            mapper_arch = self.hparams['feature_mapper_architecture']
            encoder_feature_map_dim = mapper_arch['encoder_feature_map_dim']#[32, 32, 64, 64]
            encoder_feature_map_mlp_depth = mapper_arch['encoder_mlp_depth']
            encoder_radius_feature_map = mapper_arch['encoder_radius']
            encoder_nsample_feature_map = mapper_arch['encoder_nsample']
            self.encoder_feature_map = nn.ModuleList()
            for i in range(len(encoder_feature_map_dim)):
                if i==0:
                    first_conv = self.hparams["bn_first"]
                    if first_conv:
                        in_dim = feature_dim_condition[i]
                    else:
                        in_dim = partial_in_fea_dim
                    query_feature_dim = in_fea_dim
                else:
                    first_conv = False
                    in_dim = feature_dim_condition[i]
                    query_feature_dim = self.hparams['architecture']['feature_dim'][i]
                out_dim = encoder_feature_map_dim[i]
                mlp_spec = [in_dim] + [out_dim]*encoder_feature_map_mlp_depth
                self.encoder_feature_map.append(FeatureMapModule(mlp_spec, encoder_radius_feature_map[i], encoder_nsample_feature_map[i], 
                        use_xyz=self.hparams["model.use_xyz"], include_abs_coordinate=self.include_abs_coordinate,
                        include_center_coordinate = self.hparams.get("include_center_coordinate", False),
                        bn=self.bn, bn_first=self.hparams["bn_first"], bias=self.hparams["bias"], res_connect=self.hparams["res_connect"],
                        first_conv=first_conv, first_conv_in_channel=partial_in_fea_dim, neighbor_def=mapper_arch['neighbor_definition'],
                        activation=self.network_activation,
                        attention_setting=self.FeatureMapper_attention_setting, query_feature_dim=query_feature_dim))

        # build SA module for the noisy point cloud x_t
        arch = self.hparams['architecture']
        npoint = arch['npoint']#[1024, 256, 64, 16]
        radius = arch['radius']#[0.1, 0.2, 0.4, 0.8]
        nsample = arch['nsample']#[32, 32, 32, 32]
        feature_dim = arch['feature_dim']#[32, 64, 128, 256, 512]
        mlp_depth = arch['mlp_depth']#3
        # if first conv, first conv in_fea_dim + encoder_feature_map_dim[0] -> feature_dim[0]
        # if not first conv, mlp[0] = in_fea_dim + encoder_feature_map_dim[0]
        additional_fea_dim = encoder_feature_map_dim if self.include_local_feature else None  #[32, 32, 64, 64]
        self.SA_modules = self.build_SA_model(npoint, radius, 
                                nsample, feature_dim, mlp_depth, 
                                in_fea_dim+encoder_feature_map_dim[0] if self.include_local_feature else in_fea_dim,
                                self.hparams['include_t'], self.hparams["include_class_condition"], 
                                include_global_feature=self.include_global_feature, global_feature_dim=self.global_feature_dim,
                                additional_fea_dim = additional_fea_dim,
                                neighbor_def=arch['neighbor_definition'], activation=self.network_activation,
                                bn=self.bn, attention_setting=self.attention_setting,
                                global_attention_setting=self.global_attention_setting)

        if self.include_local_feature:
            # build FP module for condition cloud
            include_grouper_condition = condition_arch.get('include_grouper', False)
            use_knn_FP_condition =  condition_arch.get('use_knn_FP', False)
            K_condition = condition_arch.get('K', 3)
            decoder_feature_dim_condition = condition_arch['decoder_feature_dim']#[32, 32, 64, 64, 128]
            decoder_mlp_depth_condition = condition_arch['decoder_mlp_depth']#3
            assert decoder_feature_dim_condition[-1] == feature_dim_condition[-1]
            self.FP_modules_condition = self.build_FP_model(decoder_feature_dim_condition, decoder_mlp_depth_condition, 
                                        feature_dim_condition, partial_in_fea_dim, False, False,
                                        use_knn_FP=use_knn_FP_condition, K=K_condition,
                                        include_grouper = include_grouper_condition, radius=radius_condition, 
                                        nsample=nsample_condition, neighbor_def=condition_arch['neighbor_definition'],
                                        activation=self.network_activation, bn=self.bn,
                                        attention_setting=self.attention_setting)

            # build mapper from condition cloud to input cloud at decoder
            decoder_feature_map_dim = mapper_arch['decoder_feature_map_dim']#[32, 32, 64, 64, 128]
            decoder_feature_map_mlp_depth = mapper_arch['decoder_mlp_depth']#2
            decoder_radius_feature_map = mapper_arch['decoder_radius']#np.array([0.1, 0.2, 0.4, 0.8, 1.6])
            decoder_nsample_feature_map = mapper_arch['decoder_nsample']#[32, 32, 32, 32, 32]
            self.decoder_feature_map = nn.ModuleList()
            for i in range(len(decoder_feature_map_dim)):
                in_dim = decoder_feature_dim_condition[i]
                out_dim = decoder_feature_map_dim[i]
                mlp_spec = [in_dim] + [out_dim]*decoder_feature_map_mlp_depth
                query_feature_dim = self.hparams['architecture']['decoder_feature_dim'][i]
                self.decoder_feature_map.append(FeatureMapModule(mlp_spec, decoder_radius_feature_map[i], decoder_nsample_feature_map[i], 
                        use_xyz=self.hparams["model.use_xyz"], include_abs_coordinate=self.include_abs_coordinate,
                        include_center_coordinate = self.hparams.get("include_center_coordinate", False),
                        bn=self.bn, bn_first=self.hparams["bn_first"], bias=self.hparams["bias"], res_connect=self.hparams["res_connect"],
                        first_conv=False, first_conv_in_channel=0, neighbor_def=mapper_arch['neighbor_definition'],
                        activation=self.network_activation,
                        attention_setting=self.FeatureMapper_attention_setting, query_feature_dim=query_feature_dim))


        # build FP module for noisy point cloud x_t
        include_grouper = arch.get('include_grouper', False)
        use_knn_FP =  arch.get('use_knn_FP', False)
        K = arch.get('K', 3)
        decoder_feature_dim = arch['decoder_feature_dim']#[128, 128, 256, 256, 512]
        decoder_mlp_depth = arch['decoder_mlp_depth']#3
        assert decoder_feature_dim[-1] == feature_dim[-1]
        additional_fea_dim = decoder_feature_map_dim[1:] if self.include_local_feature else None
        self.FP_modules = self.build_FP_model(decoder_feature_dim, decoder_mlp_depth, feature_dim, in_fea_dim,
                                                self.hparams['include_t'], self.hparams["include_class_condition"], 
                        include_global_feature=self.include_global_feature, global_feature_dim=self.global_feature_dim,
                                                additional_fea_dim=additional_fea_dim,
                                                use_knn_FP=use_knn_FP, K=K,
                                                include_grouper = include_grouper, radius=radius, 
                                                nsample=nsample, neighbor_def=arch['neighbor_definition'],
                                                activation=self.network_activation, bn=self.bn,
                                                attention_setting=self.attention_setting,
                                                global_attention_setting=self.global_attention_setting)
        
        # set point upsampling factor
        # this is used in the refinement network, we refine and upsample the input coarse point cloud at the same time
        point_upsample_factor = self.hparams.get('point_upsample_factor', 1)
        if point_upsample_factor > 1:
            if self.hparams.get('include_displacement_center_to_final_output', False):
                point_upsample_factor = point_upsample_factor-1
            self.hparams['out_dim'] = int(self.hparams['out_dim'] * (point_upsample_factor+1))
        
        input_dim = decoder_feature_dim[0]+3
        if self.include_local_feature:
            input_dim = input_dim + decoder_feature_map_dim[0]
        if self.hparams["bn_first"]:
            self.fc_lyaer = nn.Sequential(
                copy.deepcopy(self.network_activation_function),
                nn.Conv1d(input_dim, self.hparams['out_dim'], kernel_size=1),
            )
        else:
            if self.bn:
                self.fc_lyaer = nn.Sequential(
                    nn.Conv1d(input_dim, 128, kernel_size=1, bias=self.hparams["bias"]),
                    nn.GroupNorm(32, 128),
                    copy.deepcopy(self.network_activation_function),
                    nn.Conv1d(128, self.hparams['out_dim'], kernel_size=1),
                )
            else:
                self.fc_lyaer = nn.Sequential(
                    nn.Conv1d(input_dim, 128, kernel_size=1, bias=self.hparams["bias"]),
                    copy.deepcopy(self.network_activation_function),
                    nn.Conv1d(128, self.hparams['out_dim'], kernel_size=1),
                )
        

    def reset_cond_features(self):
        self.l_uvw = None
        self.encoder_cond_features = None
        self.decoder_cond_features = None
        self.global_feature = None

    def forward(self, pointcloud, condition, ts=None, label=None):
        """
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)

            condition: (B,M,3 + input_channels) tensor, a condition point cloud.
        """
        if self.include_global_feature or self.include_local_feature:
            assert condition is not None

        with torch.no_grad():
            #最后三维作为坐标
            if self.attach_position_to_input_feature:
                xyz_ori = pointcloud[:,:,0:3] / self.scale_factor
                pointcloud = torch.cat([pointcloud, xyz_ori], dim=2)

                if condition is not None:
                    uvw_ori = condition[:,:,0:3] / self.scale_factor
                    condition = torch.cat([condition, uvw_ori], dim=2)
                # in this case, the input pointcloud is of shape (B,N,C)
                # the output pointcloud is of shape (B,N,C+3)
                # we want the X not only as position, but also as input feature
                partial_in_fea_dim = self.partial_in_fea_dim - 3  #1
            else:
                partial_in_fea_dim = self.partial_in_fea_dim   #1
            
            xyz, features = self._break_up_pc(pointcloud)  #用于分离坐标和特征
            xyz = xyz / self.scale_factor
            #uvw是条件点云的坐标
            if condition is not None:
                uvw, cond_features = self._break_up_pc(condition)
                uvw = uvw / self.scale_factor

        if (ts is not None) and self.hparams['include_t']:
            t_emb = calc_t_emb(ts, self.hparams['t_dim'])
            t_emb = self.fc_t1(t_emb)
            t_emb = self.activation(t_emb)
            t_emb = self.fc_t2(t_emb)  #(B, t_dim*4)
            t_emb = self.activation(t_emb)
        else:
            t_emb = None

        if (label is not None) and self.hparams['include_class_condition']:
            # label should be 1D tensor of integers of shape (B)
            class_emb = self.class_emb(label) # shape (B, condition_emb_dim)
        else:
            class_emb = None
        
        if self.include_global_feature:
            if partial_in_fea_dim> 0:
                condition_input_fea = condition[:,:,3:(3+partial_in_fea_dim)] #提取第四维？
                global_input = torch.cat([uvw, condition_input_fea], dim=2)  #(B, N, 4)
            else:
                global_input = uvw
            global_feature = self.global_pnet(global_input.transpose(1,2)) #(B, 1024)
            condition_emb = global_feature
            #second_condition_emb即为标签的embedding(B, 128)
            second_condition_emb = class_emb if self.hparams['include_class_condition'] else None
        else:
            condition_emb = class_emb if self.hparams['include_class_condition'] else None
            second_condition_emb = None

        if condition is not None:
            l_uvw, l_cond_features = [uvw], [cond_features]
        l_xyz, l_features = [xyz], [features]

        for i in range(len(self.SA_modules)):
            """
            li_uvw: the coordinates of the condition points
            li_xyz: the coordinates of the noisy points
            """
            if self.include_local_feature:
                if (self.encoder_cond_features is not None): #None
                    mapped_feature = self.encoder_feature_map[i](self.l_uvw[i], self.encoder_cond_features[i], l_xyz[i], subset=False, 
                                    record_neighbor_stats=self.record_neighbor_stats, pooling=self.pooling,
                                    features_at_new_xyz = l_features[i])
                #聚合condition points的特征
                else:
                    li_uvw, li_cond_features = self.SA_modules_condition[i](l_uvw[i], l_cond_features[i], t_emb=None, condition_emb=None,
                                                            subset=True, record_neighbor_stats=self.record_neighbor_stats,
                                                            pooling=self.pooling) #neighbor=False
                    l_uvw.append(li_uvw)
                    l_cond_features.append(li_cond_features)

                    mapped_feature = self.encoder_feature_map[i](l_uvw[i], l_cond_features[i], l_xyz[i], subset=False, 
                                    record_neighbor_stats=self.record_neighbor_stats, pooling=self.pooling,
                                    features_at_new_xyz = l_features[i])    #(B, 32, 2048)

                input_feature = torch.cat([ mapped_feature, l_features[i] ], dim=1)   #(B, 35, 2048)
            else:
                input_feature = l_features[i]

            li_xyz, li_features = self.SA_modules[i](l_xyz[i], input_feature, t_emb=t_emb, 
                                    condition_emb=condition_emb, second_condition_emb=second_condition_emb,
                                    subset=True, record_neighbor_stats=self.record_neighbor_stats,
                                    pooling=self.pooling)
            l_xyz.append(li_xyz)
            l_features.append(li_features)
        
        

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            # i from -1 to -len(self.FP_modules)
            # equivalent to i from len(self.SA_modules)-1 to 0
            if self.include_local_feature:
                if (self.decoder_cond_features is not None):
                    mapped_feature = self.decoder_feature_map[i](self.l_uvw[i], self.decoder_cond_features[i], l_xyz[i],
                                                subset=False, record_neighbor_stats=self.record_neighbor_stats, 
                                                pooling=self.pooling, features_at_new_xyz = l_features[i])
                else:
                    l_cond_features[i - 1] = self.FP_modules_condition[i](
                                                l_uvw[i - 1], l_uvw[i], l_cond_features[i - 1], l_cond_features[i],
                                                t_emb = None, condition_emb=None, record_neighbor_stats=self.record_neighbor_stats, 
                                                pooling=self.pooling)
                    # l_uvw[i - 1] is position of previous layer, which we want to propagate feature to
                    # l_cond_features[i - 1] is the feature from the encoder in the previous layer
                    # l_uvw[i], l_cond_features[i] are position and feature of current layer
                    # print(l_uvw[i].shape, l_cond_features[i].shape, l_xyz[i].shape)
                    
                    mapped_feature = self.decoder_feature_map[i](l_uvw[i], l_cond_features[i], l_xyz[i],
                                                subset=False, record_neighbor_stats=self.record_neighbor_stats, 
                                                pooling=self.pooling, features_at_new_xyz = l_features[i]) 
                # when i=-1, this the condition feature from the last layer of the encoder
                input_feature = torch.cat([ mapped_feature, l_features[i] ], dim=1)
            else:
                input_feature = l_features[i]

            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], input_feature,
                t_emb = t_emb, condition_emb=condition_emb, second_condition_emb=second_condition_emb, 
                record_neighbor_stats=self.record_neighbor_stats, pooling=self.pooling)
            # l_xyz[i - 1] is position of previous layer, which we want to propagate feature to
            # l_features[i - 1] is the feature from the encoder in the previous layer
            # l_xyz[i], l_features[i] are position and feature of current layer
        
        # l_cond_features[0] has not been used
        if self.include_local_feature:
            mapped_feature = self.decoder_feature_map[0](l_uvw[0], l_cond_features[0], l_xyz[0],
                                    subset=False, record_neighbor_stats=self.record_neighbor_stats, pooling=self.pooling,
                                    features_at_new_xyz = l_features[0])
            out_feature = torch.cat([ mapped_feature,l_features[0] ], dim=1)
        else:
            out_feature = l_features[0]

        out_feature = torch.cat([out_feature, xyz.transpose(1,2)], dim=1)
        
        out = self.fc_lyaer(out_feature)
        out = torch.transpose(out, 1,2)

        return out

    def report_feature_map_neighbor_stats(self, FM_module, module_name='FM_module'):
        with torch.no_grad():
            # num_groupers_per_layer = len(SA_module[0].groupers)
            neigh_stats = []
            neigh_quantile = []
            for i in range(len(FM_module)):
                neigh_stats.append(FM_module[i].mapper.neighbor_stats)
                neigh_quantile.append(FM_module[i].mapper.neighbor_num_quantile)
            
            neigh_stats = torch.stack(neigh_stats, dim=0)
            neigh_quantile = torch.stack(neigh_quantile, dim=0)

        print('%s: neighbor number (min, mean, max)' % (module_name))
        print(neigh_stats)
        print('%s: neighbor quantile (0-0.1-1)' % (module_name))
        print(neigh_quantile)

    def report_neighbor_stats(self):
        if not self.record_neighbor_stats:
            print('neighbor stats is not recorded')
            return
        self.report_SA_module_neighbor_stats(self.SA_modules, module_name='Input cloud SA_module')
        if self.include_local_feature:
            self.report_SA_module_neighbor_stats(self.SA_modules_condition, module_name='Condition cloud SA_module')

        self.report_FP_module_neighbor_stats(self.FP_modules, module_name='Input cloud FP_module')
        if self.include_local_feature:
            self.report_FP_module_neighbor_stats(self.FP_modules_condition, module_name='Condition cloud FP_module')

        if self.include_local_feature:
            self.report_feature_map_neighbor_stats(self.encoder_feature_map, module_name='Encoder feature mapper')
            self.report_feature_map_neighbor_stats(self.decoder_feature_map, module_name='Decoder feature mapper')
        # self.report_feature_map_neighbor_stats([self.last_map], module_name='Last mapper')




if __name__ == '__main__':
    import argparse
    import wandb
    import json
    import os

    wandb.login()
    run = wandb.init(project="load_from_json")


    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, 
                        default='exp_configs/mvp_configs/config_standard_attention_real_3072_partial_points_rot_90_scale_1.2_translation_0.1.json',
                        help='JSON file for configuration')
    args = parser.parse_args()

    path = os.getcwd()
    config_path = os.path.join(path,'pointnet2', args.config)
    with open(config_path,'r') as f:
        json_config = json.load(f)
    wandb.init(config=json_config)
    train_config = wandb.config["train_config"]

    device = 'cuda'
    pointnet_config = wandb.config["pointnet_config"]     # to define pointnet
    net = PointNet2CloudCondition(pointnet_config).to(device)

    log = False
    train = True
    if train:
        net.train()
        """
            transform_X: (B, N, 3) X_t in the forward
            condition:  (B, N, 4) incomplete points 
            ts: (B, ) diffusion steps
            label: (B, ) points label
        """
        transformed_X = torch.randn(8, 2048, 3).to(device)
        print(transformed_X.device)
        condition = torch.randn(8, 3072, 4).to(device)
        ts = torch.tensor([4, 8 ,9, 23, 32, 42, 24, 56]).to(device)
        label = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]).to(device)
        epsilon_theta = net(transformed_X, condition, ts, label=label)
        print(epsilon_theta)
        print(epsilon_theta.shape)

    if log:
        import logging
        logger = logging.getLogger(__name__)
        logger.setLevel(level = logging.INFO)
        handler = logging.FileHandler("log.txt")
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.info(net)

    