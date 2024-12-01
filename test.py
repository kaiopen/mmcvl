from model_gat import CVML
from VIGOR_GAT import VIGOR
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from points_process import point_cloud_process
import dgl
import datetime

# CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=4 --max_restarts=0 --rdzv_id=1234576890 --rdzv_backend=c10d train.py

area = 'same'
learning_rate = 1e-5
start_epoch = 0
end_epoch = 100
batch_size = 2
keep_prob_val = 0.8
dimension = 8
beta = 1e4
temperature = 0.1


parallel = True


def clean_state_dict(state_dict):
    new_state = {}
    for k,v in state_dict.items():
        if "module" in k:
            new_name = k[7:]
        else:
            new_name = k
        new_state[new_name] = v
    return new_state

def seed_worker(worker_id):
    # Torch initial seed is properly set across the different workers, we need to pass it to numpy and random.
    worker_seed = (torch.initial_seed()) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class ContrastiveLoss(nn.Module):
    def __init__(self, tem=temperature):
        super().__init__()
        self.temperature = tem

    def forward(self, scores, labels):
        exp_scores = torch.exp(scores / self.temperature)
        bool_mask = labels.ge(1e-2)
        denominator = torch.sum(exp_scores, [1, 2, 3], keepdim=True)

        inner_element = torch.log(torch.masked_select(exp_scores/denominator, bool_mask))

        return -torch.sum(inner_element*torch.masked_select(labels, bool_mask)) / torch.sum(torch.masked_select(labels, bool_mask))

def save_checkpoint(state, filename="checkpoint.pth"):
    torch.save(state, filename)

class SoftmaxCrossEntropyWithLogits(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, labels, logits, dim=-1):
        return (-labels * F.log_softmax(logits, dim=dim)).sum(dim=dim)

def main():
    # setup train/val dataset
    if(parallel == True):
        rank       = int(os.environ["RANK"]) #Rank accross all processes
        local_rank = int(os.environ["LOCAL_RANK"]) # Rank on Node
        world_size = int(os.environ['WORLD_SIZE']) # Number of processes
        print(f"RANK, LOCAL_RANK and WORLD_SIZE in environ: {rank}/{local_rank}/{world_size}")
        device = torch.device('cuda:{}'.format(local_rank))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank) # Hide devices that are not used by this process
        torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank,
                                             timeout=datetime.timedelta(minutes=15))
        torch.distributed.barrier(device_ids=[local_rank])

    else:
        device = "cuda:1"
        print(device)

    torch.cuda.set_device(device)
    
    if(parallel == True):
        train_set = VIGOR(train_test='train')
        test_set = VIGOR(train_test='test')
        g_cuda = torch.Generator(device='cpu')
        g_cuda.manual_seed(torch.initial_seed())
        sampler_train = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=True, num_replicas=world_size, rank=rank)
        sampler_val   = torch.utils.data.distributed.DistributedSampler(test_set,   shuffle=True, num_replicas=world_size, rank=rank)
        train_dataloader = DataLoader(train_set, sampler=sampler_train, batch_size=batch_size, worker_init_fn=seed_worker, generator=g_cuda, num_workers=2, pin_memory=False, drop_last=True)
        test_dataloader   = DataLoader(test_set,   sampler=sampler_val,   batch_size=2, worker_init_fn=seed_worker, generator=g_cuda, num_workers=2, pin_memory=False)
    else:
        train_dataset = VIGOR(train_test='train')
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        test_dataset = VIGOR(train_test='test')
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # build model
    for idx in range(10,80):
        # print("Epoch %d : " % idx)
        model = CVML(sa_num=8, grdH=320, grdW=640, satH=512, satW=512)
        check_point = torch.load('save/town2/GAT_model_%d.pth' % idx, map_location=device)
        check_point = clean_state_dict(check_point)
        model.load_state_dict(check_point)
    
        if(parallel == True):
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False, find_unused_parameters=False)
        

        model = model.to(device)

        bottleneck_loss = ContrastiveLoss()
        heatmap_loss = SoftmaxCrossEntropyWithLogits()
        
        softmax = torch.nn.Softmax(dim=1)

        
        model.eval()
        distance = []
        epoch_loss = []
        

        for iteration, (sat, grd, gt, lidar, points_num) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            bz = lidar.shape[0]
            all_lidar_data = []
            graphs = []
            for i in range(bz):
                points = lidar[i][:points_num[i]]
                edges_list, lidar_data = point_cloud_process(points.numpy())
                lidar_data[0], lidar_data[1], lidar_data[2], lidar_data[3] = lidar_data[0].to(device), lidar_data[1].to(device), lidar_data[2].to(device), lidar_data[3].to(device)
                g = dgl.graph((edges_list[:,0], edges_list[:,1]))
                graphs.append(g)
                all_lidar_data.append(lidar_data)
            
            graphs = dgl.batch(graphs)
            n_edges = graphs.number_of_edges()
            edge_id = torch.arange(0, n_edges, dtype=torch.long)
            graphs.edata.update({'e_id': edge_id})
            graphs = graphs.to(device)

            sat_img = sat.to(device)
            grd_img = grd.to(device)
            gt = gt.to(device)  # B 1 512 512
            gt_bottleneck = torch.max_pool2d(gt, kernel_size=64, stride=64)  # B 1 8 8

            
            logits, matching_score = model(sat_img, grd_img, all_lidar_data, graphs)  # matching_score: B 1 8 8

            logits_reshaped = logits.reshape(logits.shape[0], 512*512)
            gt_reshape = gt.reshape(logits.shape[0], 512*512)
            gt_reshape = gt_reshape / torch.sum(gt_reshape, dim=1, keepdim=True)
            loss_heatmap = torch.mean(heatmap_loss(gt_reshape, logits_reshaped))
            loss_bottleneck = bottleneck_loss(matching_score, gt_bottleneck)

            loss = loss_heatmap + loss_bottleneck * beta

            epoch_loss.append(loss.item())
            
            val_heatmap = softmax(logits_reshaped).reshape(logits.shape)
            for batch_idx in range(bz):
                current_gt = gt[batch_idx, :, :, :].cpu().detach().numpy()   # B 1 512 512
                loc_gt = np.unravel_index(current_gt.argmax(), current_gt.shape)
                current_pred = val_heatmap[batch_idx, :, :, :].cpu().detach().numpy()
                loc_pred = np.unravel_index(current_pred.argmax(), current_pred.shape)
                distance.append(np.sqrt((loc_gt[1]-loc_pred[1])**2+(loc_gt[2]-loc_pred[2])**2))
        
        curr_Testing_loss = sum(epoch_loss) / (iteration + 1)
        distance_error = np.mean(distance)
        gathered_loss = [None for _ in range(world_size)]
        gathered_error = [None for _ in range(world_size)]

        if(parallel == True):
            torch.distributed.gather_object(obj=curr_Testing_loss,
                                            object_gather_list=gathered_loss if rank==0 else None,
                                            dst=0)
            torch.distributed.gather_object(obj=distance_error,
                                            object_gather_list=gathered_error if rank==0 else None,
                                            dst=0)

        if(rank==0):
            print(f"Epoch Training Loss: {sum(gathered_loss) / len(gathered_loss)}")
            distance_error = sum(gathered_error) / len(gathered_error)
            print(f"Epoch Training error: {distance_error}")
            

        
                


if __name__ == '__main__':
    main()

