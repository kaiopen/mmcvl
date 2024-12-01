import torch
from VGG import VGG16
from new_output_head import Decoder
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from GNN.MAGNAConv import MAGNALayer
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from transformer import GPT
from PointSetProcess import PointSetPooling
from resnet import resnet18

class SA(nn.Module):
    def __init__(self, in_dim, num=8):
        super().__init__()
        hid_dim = in_dim // 2
        self.w1, self.b1 = self.init_weights_(in_dim, hid_dim, num)
        self.w2, self.b2 = self.init_weights_(hid_dim, in_dim, num)

    def init_weights_(self, din, dout, dnum):
        weight = torch.empty(din, dout, dnum)
        nn.init.normal_(weight, mean=0.0, std=0.005)
        bias = torch.empty(1, dout, dnum)
        nn.init.constant_(bias, val=0.1)
        weight = torch.nn.Parameter(weight)
        bias = torch.nn.Parameter(bias)
        return weight, bias

    def forward(self, x):
        mask, _ = x.max(1)
        batch, height, width = mask.shape
        mask = mask.view(batch, height*width)
        mask = torch.einsum('bi, ijd -> bjd', mask, self.w1) + self.b1
        mask = torch.einsum('bjd, jid -> bid', mask, self.w2) + self.b2
        return mask


class CVML(nn.Module):
    def __init__(self, sa_num=8, grdH=256, grdW=1024, satH=512, satW=512):
        super().__init__()
        # grd
        self.resnet_grd = resnet18()
        # self.vgg_grd = VGG16()
        self.grd_max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        in_dim_grd = (grdH // 32) * (grdW // 32)
        self.grd_SA = SA(in_dim_grd, num=sa_num)
        # sat
        self.sat_split = 8  # split satellite feature into 8*8 sub-volumes
        # self.vgg_sat = VGG16()
        self.resnet_sat = resnet18()
        in_dim_sat = (satH // 16 // self.sat_split // 2) * (satW // 16 // self.sat_split // 2)
        self.sat_SA = SA(in_dim_sat, num=sa_num)
        self.dimension = sa_num
        
        self.hidden_dim = 512
        self.gdt_layers = nn.ModuleList()
        self.gdt_layers.append(MAGNALayer(in_feats=64, hop_num=4, top_k=10, num_heads=8, hidden_dim=self.hidden_dim,
                                                 topk_type='local', layer_norm=True, feed_forward=True, head_tail_shared=1,
                                                 alpha=0.15, negative_slope=0.2, feat_drop=0, attn_drop=0))

        self.gdt_layers.append(MAGNALayer(in_feats=self.hidden_dim, hop_num=4, top_k=10, num_heads=8, hidden_dim=self.hidden_dim,
                                                 topk_type='local', layer_norm=True, feed_forward=True, head_tail_shared=1,
                                                 alpha=0.15, negative_slope=0.2, feat_drop=0, attn_drop=0))
        
        self.pc_SA = SA(1024, num=sa_num)
        
        # self.transformer = GPT(512, 4, 4, 8, grdH // 32, grdW // 32, 1, 1024, 1, 0.1, 0.1, 0.1)
        self.setpooling = PointSetPooling()
        self.costmap_decoder = Decoder()

    

    def forward(self, img_sat, img_grd, all_lidar_data, g):
        # grd
        
        _, _, _, _, grd_local = self.resnet_grd(img_grd)
        
        # grd_local, _, _, _, _ = self.vgg_grd(img_grd)
        # grd_local = self.grd_max_pool(grd_local)
        
        batch, channel, g_height, g_width = grd_local.shape
        # GNN
        all_lidar_features = []
        
        for lidar_data in all_lidar_data:
            all_lidar_features.append(self.setpooling(lidar_data[0], lidar_data[1], lidar_data[2], lidar_data[3]))

        points_feat = torch.cat(all_lidar_features)

        points_feat = self.gdt_layers[0](g, points_feat, None) # (points_num * bz, 512)
        points_feat = self.gdt_layers[1](g, points_feat, None) # (points_num * bz, 512)
        points_feat = points_feat.view(batch, -1, self.hidden_dim).unsqueeze(2).permute(0, 3, 1, 2)

        # transformer
        
        # grd_local, points_feat = self.transformer(grd_local, points_feat)
        
        
        grd_w = self.grd_SA(grd_local)

        grd_local = grd_local.view(batch, channel, g_height*g_width)
        grd_global = torch.matmul(grd_local, grd_w).view(batch, -1)  # (Batch, channel*sa_num = 512*8)
        grd_global = F.normalize(grd_global, p=2, dim=1)
        
        #points cloud

        points_w = self.pc_SA(points_feat)
    
        points_feat = points_feat.view(batch, self.hidden_dim, 1024)
        points_feat = torch.matmul(points_feat, points_w).view(batch, -1)
        points_feat = F.normalize(points_feat, p=2, dim=1)
        
        # sat
        # sat_local, sat512, sat256, sat128, sat64 = self.vgg_sat(img_sat)  # sat_local [Batch, 512, 32, 32]
        sat128, _sat128, sat64, sat32, sat_local = self.resnet_sat(img_sat)
        _, channel, s_height, s_width = sat_local.shape

        sat_global = []
        for i in range(0, self.sat_split):
            strip_horizontal = sat_local[:, :, i*s_height//self.sat_split:(i+1)*s_height//self.sat_split, :]
            sat_global_horizontal = []
            for j in range(0, self.sat_split):
                patch = strip_horizontal[:, :, :, j*s_height//self.sat_split:(j+1)*s_height//self.sat_split]
                # print(patch.shape)
                sat_w = self.sat_SA(patch)  # Batch 16 8
                _, channel, p_height, p_width = patch.shape
                patch = patch.reshape(batch, channel, p_height*p_width)  # Batch 512 16
                # Batch 512 8 --> Batch 1, 1, 4096
                patch_global = torch.matmul(patch, sat_w).reshape(batch, 1, 1, self.dimension*channel)
                patch_global = F.normalize(patch_global, p=2, dim=-1)

                if j == 0:
                    sat_global_horizontal = patch_global
                else:
                    sat_global_horizontal = torch.cat([sat_global_horizontal, patch_global], dim=2)
            if i == 0:
                sat_global = sat_global_horizontal
            else:
                sat_global = torch.cat([sat_global, sat_global_horizontal], dim=1)
        # get matching score and logits
        # B 4096 --> B 1 1 4096 --> B 8 8 4096
        grd_global_broadcasted = torch.broadcast_to(grd_global.reshape(batch, 1, 1, grd_global.shape[-1]),
                                                    [grd_global.shape[0], self.sat_split, self.sat_split, grd_global.shape[-1]])
        
        points_feat_broadcasted = torch.broadcast_to(points_feat.reshape(batch, 1, 1, points_feat.shape[-1]),
                                                    [points_feat.shape[0], self.sat_split, self.sat_split, points_feat.shape[-1]])
        
        matching_score = torch.sum(torch.mul(grd_global_broadcasted, sat_global), dim=-1, keepdim=True)
        matching_score_points = torch.sum(torch.mul(points_feat_broadcasted, sat_global), dim=-1, keepdim=True)
        
        matching_score = matching_score + matching_score_points
        
        cost_map = torch.cat([matching_score, sat_global], dim=3)  # Batch 8 8 4097
        # print(cost_map.shape)
        
        # logits = self.costmap_decoder(cost_map.permute(0, 3, 1, 2), sat512, sat256, sat128, sat64, sat_local)
        logits = self.costmap_decoder(cost_map.permute(0, 3, 1, 2), sat128, _sat128, sat64, sat32, sat_local)
        return logits, matching_score.permute(0, 3, 1, 2)