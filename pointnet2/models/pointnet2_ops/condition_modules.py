from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from pointnet2_ops import pointnet2_utils
from pointnet2_ops.attention import AttentionModule, GlobalAttentionModule

class MyGroupNorm(nn.Module):
    def __init__(self, num_groups, num_channels):
        super(MyGroupNorm, self).__init__()
        self.num_channels = num_channels - num_channels % num_groups
        self.num_groups = num_groups
        self.group_norm = nn.GroupNorm(self.num_groups, self.num_channels)
    def forward(self, x):
        # x is of shape BCHW
        if x.shape[1] == self.num_channels:
            out = self.group_norm(x)
        else:
            # some times we may attach position info to the end of feature in the channel dimension
            # we do not need to normalize them
            x0 = x[:,0:self.num_channels,:,:]
            res = x[:,self.num_channels:,:,:]
            x0_out = self.group_norm(x0)
            out = torch.cat([x0_out, res], dim=1)
        return out


def build_shared_mlp(mlp_spec: List[int]):
    layers = []
    for i in range(1, len(mlp_spec)):
        layers.append(nn.Conv2d(mlp_spec[i - 1], mlp_spec[i], kernel_size=1, bias=True))
        layers.append(MyGroupNorm(32, mlp_spec[i]))
        layers.append(nn.ReLU(True))

    return nn.Sequential(*layers)

class Mlp_condition(nn.Module):
    def __init__(self, mlp_spec, res_connect=True):
        super(self).__init__()

        self.res_connect_bool = res_connect
        if res_connect:
            if mlp_spec[0] == mlp_spec[-1]:
                self.res_connect = None
            else:
                self.res_connect = nn.Conv2d(mlp_spec[0], mlp_spec[-1], kernel_size=1, bias=True)

        self.first_mlp = build_shared_mlp(mlp_spec[0:2])
        self.second_mlp = build_shared_mlp(mlp_spec[1:3])
        self.rest_mlp = build_shared_mlp(mlp_spec[2:])

    def forward(self, feature):
        x = feature
        x = self.first_mlp(x)
        x = self.second_mlp(x)
        x = self.rest_mlp(x)
        x = x + self.res_connect(feature)
        return x


class _PointnetSAModuleBase(nn.Module):
    def __init__(self):
        super(_PointnetSAModuleBase, self).__init__()
        self.npoint = None
        self.groupers = None
        self.mlps = None

    def forward(
        self, xyz: torch.Tensor, features: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (B, C, N) tensor of the descriptors of the the features
        Returns
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B,  \sum_k(mlps[k][-1]), npoint) tensor of the new_features descriptors
        """

        new_features_list = []

        xyz_flipped = xyz.transpose(1, 2).contiguous()
        furthest_point_idx = pointnet2_utils.furthest_point_sample(xyz, self.npoint)
        new_xyz = (pointnet2_utils.gather_operation(xyz_flipped, furthest_point_idx).transpose(1, 2).contiguous())  #coordinates of center points
        new_xyz_feat = pointnet2_utils.gather_operation(features, furthest_point_idx)

        #multi scale groups of features
        for i in range(len(self.groupers)):
            grouped_features, count = self.groupers[i](xyz, new_xyz, features, return_counts=True)  # (B, C+3, npoint, nsample)
            out_features = self.mlps[i](grouped_features)  # (B, mlp[-1], npoint, K)
            new_features = self.attention_modules[i](new_xyz_feat, grouped_features, out_features, count)
            new_features_list.append(new_features)

        return new_xyz, torch.cat(new_features_list, dim=1)

#在init中修改attention_modules

class PointnetSAModuleMSG(_PointnetSAModuleBase):
    '''
    Pointnet set abstrction layer with multiscale grouping
    Parameters
    ----------
    npoint : int
        Number of FPS points
    radii : list of float32
        list of radii to group with
    nsamples : list of int32
        Number of samples in each ball query
    mlps : list of list of int32
        Spec of the pointnet before the global max_pool for each scale
    include_abs_coordinate: 计算邻近点位置信息的时候不仅计算和中心点的相对位置差，同时加入邻近点的绝对位置
    include_center_coordinates: 计算邻近点位置信息的时候再额外加入中心点的位置信息
    neighbor_def: 计算邻近点的方式,默认为计算半径内的点
    attention_setting: 注意力的设置
    '''

    def __init__(self, npoint, radii, nsamples, mlps,
                 include_abs_coordinate, include_center_coordinate, 
                 neighbor_def, attention_setting, use_xyz=True):
        super(PointnetSAModuleMSG, self).__init__()

        assert len(radii) == len(nsamples) == len(mlps)

        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(
                pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz,
                                            include_abs_coordinate=include_abs_coordinate,
                                            include_center_coordinate=include_center_coordinate,
                                            neighbor_def=neighbor_def)
                if npoint is not None
                else pointnet2_utils.GroupAll(use_xyz)
            )
            mlp_spec = mlps[i]
            ori_mlp_spec0 = copy.deepcopy(mlp_spec[0])
            if use_xyz:
                mlp_spec[0] += 3
                if include_abs_coordinate:
                    mlp_spec[0] += 3
                if include_center_coordinate:
                    mlp_spec[0] += 3

            self.mlps.append(Mlp_condition(mlp_spec))

            C_in1 = ori_mlp_spec0
            C_in2 = mlp_spec[0]
            C1 = C_in1
            C2 = C_in2
            C_out = mlp_spec[-1]
            self.attention_modules.append(AttentionModule(C_in1, C_in2, C1, C2, C_out, 
                attention_bn=attention_setting['attention_bn'], 
                transform_grouped_feat_out=attention_setting['transform_grouped_feat_out'], 
                last_activation=attention_setting['last_activation']))  #皆为True
            
            

class PointnetSAModule(PointnetSAModuleMSG):
    r"""Pointnet set abstrction layer
    Parameters
    ----------
    npoint : int
        Number of features
    radius : float
        Radius of ball
    nsample : int
        Number of samples in the ball query
    mlp : list
        Spec of the pointnet before the global max_pool
    """

    def __init__(
        self, mlp, npoint, radius, nsample,
        include_abs_coordinate, include_center_coordinate,
        neighbor_def, attention_setting, use_xyz=True):
        super(PointnetSAModule, self).__init__(
            mlps=[mlp],
            npoint=npoint,
            radii=[radius],
            nsamples=[nsample],
            use_xyz=use_xyz,
            include_abs_coordinate=include_abs_coordinate,
            include_center_coordinate=include_center_coordinate,
            neighbor_def=neighbor_def,
            attention_setting=attention_setting
        )


class PointnetFPModule(nn.Module):
    r"""Propigates the features of one set to another
    Parameters
    ----------
    mlp : list
        Pointnet module parameters
    bn : bool
        Use batchnorm
    """

    def __init__(self, mlp, bn=True):
        # type: (PointnetFPModule, List[int], bool) -> None
        super(PointnetFPModule, self).__init__()
        self.mlp = build_shared_mlp(mlp, bn=bn)

    def forward(self, unknown, known, unknow_feats, known_feats):
        # type: (PointnetFPModule, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""
        Parameters
        ----------
        即将聚合过的中心点known特征,向上传播,取target point的最近三个点,根据距离计算加权和
        unknown : torch.Tensor
            (B, n, 3) tensor of the xyz positions of the target features
        known : torch.Tensor
            (B, m, 3) tensor of the xyz positions of the known features
        unknow_feats : torch.Tensor
            (B, C1, n) tensor of the features to be propigated to
        known_feats : torch.Tensor
            (B, C2, m) tensor of features to be propigated
        Returns
        -------
        new_features : torch.Tensor
            (B, mlp[-1], n) tensor of the features of the unknown features
        """

        if known is not None:
            dist, idx = pointnet2_utils.three_nn(unknown, known)
            dist_recip = 1.0 / (dist + 1e-8)  #取距离的倒数
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            interpolated_feats = pointnet2_utils.three_interpolate(
                known_feats, idx, weight
            )
        else:
            interpolated_feats = known_feats.expand(
                *(known_feats.size()[0:2] + [unknown.size(1)])
            )

        if unknow_feats is not None:
            new_features = torch.cat(
                [interpolated_feats, unknow_feats], dim=1
            )  # (B, C2 + C1, n)
        else:
            new_features = interpolated_feats

        new_features = new_features.unsqueeze(-1)
        new_features = self.mlp(new_features)

        return new_features.squeeze(-1)