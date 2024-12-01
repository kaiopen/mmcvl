import torch
from VGG import VGG16
import torch.nn as nn
import torch.nn.functional as F

def fc_layer(x, input_dim, output_dim, init_dev, init_bias, activation_fn=F.relu, reuse=False):
    
    weight = torch.empty((input_dim, output_dim), requires_grad=True).normal_(mean=0.0, std=init_dev)
    bias = torch.empty((output_dim,), requires_grad=True).fill_(init_bias)

    if activation_fn is not None:
        out = F.linear(x, weight, bias)
        out = activation_fn(out)
    else:
        out = F.linear(x, weight, bias)

    return out




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

    
def compute_loss(sat_global, grd_global,delta_regression, gt_delta, loss_weight = 10.0):
    batch_size = sat_global.shape[0]
    dist_array = 2 - 2 * torch.matmul(sat_global, grd_global.t())
    pos_dist = torch.diag(dist_array)
    pair_n = batch_size * (batch_size - 1.0)

    # ground to satellite
    triplet_dist_g2s = pos_dist - dist_array
    loss_g2s = torch.sum(torch.log(1 + torch.exp(triplet_dist_g2s * loss_weight))) / pair_n

    # satellite to ground
    triplet_dist_s2g = pos_dist.unsqueeze(1) - dist_array
    loss_s2g = torch.sum(torch.log(1 + torch.exp(triplet_dist_s2g * loss_weight))) / pair_n

    loss = (loss_g2s + loss_s2g) / 2.0
    
    loss_delta = torch.mean(torch.sum((delta_regression - gt_delta)**2, dim=1))
    
    
    return loss + loss_delta

class CVR(nn.Module):
    def __init__(self, sa_num=8, grdH=320, grdW=640, satH=512, satW=512):
        super().__init__()
        # grd
        self.vgg_grd = VGG16()
        self.grd_max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        in_dim_grd = (grdH // 32) * (grdW // 32)
        self.grd_SA = SA(in_dim_grd, num=sa_num)
        # sat
        self.sat_max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.vgg_sat = VGG16()
        in_dim_sat = (satH // 32) * (satW // 32)
        self.sat_SA = SA(in_dim_sat, num=sa_num)
        self.dimension = sa_num
        
        self.fc1 = nn.Linear(4096 * 2, 512)
        self.fc2 = nn.Linear(512, 2)


    def forward(self, img_sat, img_grd):
        # grd
        grd_local, _, _, _, _ = self.vgg_grd(img_grd)
        grd_local = self.grd_max_pool(grd_local)
        batch, channel, g_height, g_width = grd_local.shape
        grd_w = self.grd_SA(grd_local)
        grd_local = grd_local.view(batch, channel, g_height*g_width)
        grd_local = torch.matmul(grd_local, grd_w).view(batch, -1)
        grd_local = F.normalize(grd_local, p=2, dim=1)

        # sat
        sat_local, _, _, _, _ = self.vgg_sat(img_sat)  
        sat_local = self.sat_max_pool(sat_local)
        batch, channel, s_height, s_width = sat_local.shape
        sat_w = self.sat_SA(sat_local)
        sat_local = sat_local.view(batch, channel, s_height*s_width)
        sat_local = torch.matmul(sat_local, sat_w).view(batch, -1)
        sat_local = F.normalize(sat_local, p=2, dim=1)
        
        both_feature = torch.cat([sat_local, grd_local], dim=-1)
        both_feature = F.relu(self.fc1(both_feature))
        delta_regression = self.fc2(both_feature)
        
        return sat_local, grd_local, delta_regression