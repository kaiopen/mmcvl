import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import random
from sklearn.model_selection import train_test_split

def input_transform(size):
    return transforms.Compose([
        transforms.Resize(size=tuple(size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

class VIGOR(Dataset):
    def __init__(self, train_test):
        super(VIGOR, self).__init__()
        
        self.isTrain = train_test
        
        self.root = '/home/rv/cst/Town02/rain/' 
        
        #town1 : 0.83
        #town2 : 0.45
        #town7 : 0.47
        
        
        self.sat_size = [512, 512]  # [320, 320] or [512, 512]
        self.grd_size = [320, 640]  # [320, 640]  # [224, 1232]
        
        
        self.grd_img_root = self.root + train_test +  "/rgb/"
        self.grd_points_root = self.root + train_test +  "/lidar/"
        self.delta = np.loadtxt(self.root + train_test +  "/label.txt")[:,2:4]
        
        self.delta = np.array(self.delta)
        if(train_test == 'train'):
            self.delta = self.delta[:7000]
        elif(train_test == 'test'):
            self.delta = self.delta[:2000]

        self.data_size = len(self.delta)
        
        self.grd_transform = input_transform(self.grd_size)
        self.sat_transform = input_transform(self.sat_size)

        # load train satellite image
        
        sat_img = Image.open(self.root + 'world.png').convert('RGB')
        self.sat_img = self.sat_transform(sat_img)
        # print(self.data_size)
        

    def __len__(self):
        return self.data_size

    def __getitem__(self, index):
        # print(index)
        # load train ground image
        # if(self.data_size < 4000):
        #     grd_img = Image.open(self.grd_img_root + str(index + 6000) + '.png')
        #     grd_points = np.loadtxt(self.grd_points_root + str(index + 6000) + '.txt')[:,0:3]
        # else:
        #     grd_points = np.loadtxt(self.grd_points_root + str(index) + '.txt')[:,0:3]
        #     grd_img = Image.open(self.grd_img_root + str(index) + '.png')
        
        
        grd_img = Image.open(self.grd_img_root + str(index) + '.png')
        grd_img = self.grd_transform(grd_img)
        grd_points = np.loadtxt(self.grd_points_root + str(index) + '.txt')[:,0:3]
        grd_points[:, 1] *= -1
        points_num = grd_points.shape[0]
        
        padded_grd_points = np.pad(grd_points, ((0, 40000 - points_num), (0, 0)), mode='constant')
        padded_grd_points = torch.from_numpy(padded_grd_points)
        points_num = torch.Tensor([points_num]).long()

        [row_offset , col_offset] = self.delta[index]  # delta = [delta_lat, delta_lon]
        row_offset_resized = 256 - (row_offset).astype(np.int32)
        col_offset_resized = (col_offset).astype(np.int32) - 256

        x, y = np.meshgrid(
            np.linspace(-self.sat_size[0] / 2 + row_offset_resized, self.sat_size[0] / 2 + row_offset_resized,
                        self.sat_size[0]),
            np.linspace(-self.sat_size[0] / 2 - col_offset_resized, self.sat_size[0] / 2 - col_offset_resized,
                        self.sat_size[0]))
        d = np.sqrt(x * x + y * y)
        sigma, mu = 4, 0.0
        gt = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))
        gt = torch.from_numpy(gt)
        gt = gt.unsqueeze(0)
        

        return self.sat_img, grd_img, gt, padded_grd_points, points_num