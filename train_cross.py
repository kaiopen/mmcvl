from model import CVML
# from model_only_image import CVML
from cross_VIGOR import VIGOR
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from kitti import SatGrdDatasetTest, SatGrdDataset
from torchvision import transforms
import datetime

# CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=4 --max_restarts=0 --rdzv_id=1234576890 --rdzv_backend=c10d train.py

area = 'same'
learning_rate = 1e-5
start_epoch = 0
end_epoch = 200
batch_size = 4
dimension = 8
temperature = 0.1
beta = 1e4


parallel = True
dataset_name = 'vigor'

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

    # build model
    # model = CVML(sa_num=8, grdH=256, grdW=1024, satH=512, satW=512)
    model = CVML()
    check_point = torch.load('save/carla/model.pth',map_location=device)
    check_point = clean_state_dict(check_point)
    model.load_state_dict(check_point)
    
    if(parallel == True):
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False, find_unused_parameters=False)

    if(dataset_name == "vigor"):
        if(parallel == True):
            train_set = VIGOR(train_test='train')
            test_set = VIGOR(train_test='test')
            g_cuda = torch.Generator(device='cpu')
            g_cuda.manual_seed(torch.initial_seed())
            sampler_train = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=True, num_replicas=world_size, rank=rank)
            sampler_val   = torch.utils.data.distributed.DistributedSampler(test_set,   shuffle=True, num_replicas=world_size, rank=rank)
            train_dataloader = DataLoader(train_set, sampler=sampler_train, batch_size=batch_size, worker_init_fn=seed_worker, generator=g_cuda, num_workers=2, pin_memory=False, drop_last=True)
            test_dataloader   = DataLoader(test_set,   sampler=sampler_val,   batch_size=batch_size, worker_init_fn=seed_worker, generator=g_cuda, num_workers=2, pin_memory=False)
        else:
            train_dataset = VIGOR(train_test='train')
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

            test_dataset = VIGOR(train_test='test')
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    if(dataset_name == "kitti"):
        if(parallel == True):
            GrdImg_H = 256  # 256 # original: 375 #224, 256
            GrdImg_W = 1024  # 1024 # original:1242 #1248, 1024
            GrdOriImg_H = 375
            GrdOriImg_W = 1242
            num_thread_workers = 1

            train_file = 'train_files.txt'
            test1_file = 'test1_files.txt'
            test2_file = 'test2_files.txt'
            dataset_root = '' 


            satmap_transform = transforms.Compose([
                    transforms.Resize(size=[512, 512]),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

            grdimage_transform = transforms.Compose([
                    transforms.Resize(size=[GrdImg_H, GrdImg_W]),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            
            train_set = SatGrdDataset(root=dataset_root, file=train_file,
                                transform=(satmap_transform, grdimage_transform),
                                shift_range_lat=20,
                                shift_range_lon=20,
                                rotation_range=180)
            
            test_set= SatGrdDatasetTest(root=dataset_root, file=test1_file,
                                transform=(satmap_transform, grdimage_transform),
                                shift_range_lat=20,
                                shift_range_lon=20,
                                rotation_range=180)
            
            # train_set = VIGOR(train_test='train')
            # test_set = VIGOR(train_test='test')
            g_cuda = torch.Generator(device='cpu')
            g_cuda.manual_seed(torch.initial_seed())
            sampler_train = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=True, num_replicas=world_size, rank=rank)
            sampler_val   = torch.utils.data.distributed.DistributedSampler(test_set,   shuffle=True, num_replicas=world_size, rank=rank)
            train_dataloader = DataLoader(train_set, sampler=sampler_train, batch_size=batch_size, worker_init_fn=seed_worker, generator=g_cuda, num_workers=2, pin_memory=False, drop_last=True)
            test_dataloader   = DataLoader(test_set,   sampler=sampler_val,   batch_size=batch_size, worker_init_fn=seed_worker, generator=g_cuda, num_workers=2, pin_memory=False)

    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    bottleneck_loss = ContrastiveLoss()
    heatmap_loss = SoftmaxCrossEntropyWithLogits()
    
    softmax = torch.nn.Softmax(dim=1)
    best_distance_error = 10000
    for epoch_idx in range(start_epoch, end_epoch):
        model.train()
        distance = []
        epoch_loss = []
        iteration = 0
        
        for iteration, (sat, grd, gt) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            bz = sat.shape[0]
            
            sat_img = sat.to(device)
            grd_img = grd.to(device)
            gt = gt.to(device)  # B 1 512 512
            gt_bottleneck = torch.max_pool2d(gt, kernel_size=64, stride=64)  # B 1 8 8

            optimizer.zero_grad()
            logits, matching_score = model(sat_img, grd_img)  # matching_score: B 1 8 8

            logits_reshaped = logits.reshape(logits.shape[0], 512*512)
            gt_reshape = gt.reshape(logits.shape[0], 512*512)
            gt_reshape = gt_reshape / torch.sum(gt_reshape, dim=1, keepdim=True)
            loss_heatmap = torch.mean(heatmap_loss(gt_reshape, logits_reshaped))
            loss_bottleneck = bottleneck_loss(matching_score, gt_bottleneck)

            loss = loss_heatmap + loss_bottleneck * beta
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())
            
            val_heatmap = softmax(logits_reshaped).reshape(logits.shape)
            for batch_idx in range(bz):
                current_gt = gt[batch_idx, :, :, :].cpu().detach().numpy()   # B 1 512 512
                loc_gt = np.unravel_index(current_gt.argmax(), current_gt.shape)
                current_pred = val_heatmap[batch_idx, :, :, :].cpu().detach().numpy()
                loc_pred = np.unravel_index(current_pred.argmax(), current_pred.shape)
                distance.append(np.sqrt((loc_gt[1]-loc_pred[1])**2+(loc_gt[2]-loc_pred[2])**2))
        
        curr_training_loss = sum(epoch_loss) / (iteration + 1)
        distance_error = np.mean(distance)

        gathered_loss = [None for _ in range(world_size)]
        gathered_error = [None for _ in range(world_size)]

        if(parallel == True):
            torch.distributed.gather_object(obj=curr_training_loss,
                                            object_gather_list=gathered_loss if rank==0 else None,
                                            dst=0)
            torch.distributed.gather_object(obj=distance_error,
                                            object_gather_list=gathered_error if rank==0 else None,
                                            dst=0)

        if(rank==0):
            print(f"Epoch {epoch_idx} Training Loss: {sum(gathered_loss) / len(gathered_loss)}")
            distance_error = sum(gathered_error) / len(gathered_error)
            print(f"Epoch {epoch_idx} Training error: {distance_error}")
            
            # torch.save(model.state_dict(), 'save/town2/model_%d.pth' % epoch_idx)
        

        # print('**********************validate*********************')
        # if(distance_error > 5):
        #     continue
        model.eval()
        distance = []
        epoch_loss = []
        iteration = 0
        for iteration, (sat, grd, gt) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            bz = sat.shape[0]
            
            sat_img = sat.to(device)
            grd_img = grd.to(device)
            gt = gt.to(device)  # B 1 512 512
            gt_bottleneck = torch.max_pool2d(gt, kernel_size=64, stride=64)  # B 1 8 8

            logits, matching_score = model(sat_img, grd_img)  # matching_score: B 1 8 8

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
        
        curr_testing_loss = sum(epoch_loss) / (iteration + 1)
        distance_error = np.mean(distance)
        median_error = np.median(distance)

        test_gathered_loss = [None for _ in range(world_size)]
        test_gathered_error = [None for _ in range(world_size)]
        test_gathered_median_error = [None for _ in range(world_size)]
        if(parallel == True):
            torch.distributed.gather_object(obj=curr_testing_loss,
                                            object_gather_list=test_gathered_loss if rank==0 else None,
                                            dst=0)
            torch.distributed.gather_object(obj=distance_error,
                                            object_gather_list=test_gathered_error if rank==0 else None,
                                            dst=0)
            torch.distributed.gather_object(obj=median_error,
                                            object_gather_list=test_gathered_median_error if rank==0 else None,
                                            dst=0)

        if(rank==0):
            test_distance_error = sum(test_gathered_error) / len(test_gathered_error)
            print(f"Epoch {epoch_idx} Testing Loss: {sum(test_gathered_loss) / len(test_gathered_loss)}")
            print(f"Epoch {epoch_idx} test error: {test_distance_error}")
            print(f"Epoch {epoch_idx} test median error: {sum(test_gathered_median_error) / len(test_gathered_median_error)}")
            if best_distance_error > test_distance_error:
                best_distance_error = test_distance_error
                torch.save(model.state_dict(), 'save/carla/model.pth')
            print(f"Best Testing error: {best_distance_error}")

                


if __name__ == '__main__':
    main()
