import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import random
from sklearn.model_selection import train_test_split
import shutil
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
        self.root = '/home/rv/cst/Town02/rain/'
        
        self.sat_size = [512, 512]  # [320, 320] or [512, 512]
        self.grd_size = [320, 640]  # [320, 640]  # [224, 1232]
        
        self.pc_root = self.root + "train" +  "/lidar/"
        self.grd_img_root = self.root + "train" +  "/rgb/"
        self.delta = np.loadtxt(self.root + "train" +  "/label.txt")[:,2:4]

        if(train_test == "train"):
            self.delta = self.delta[:1000]
        elif(train_test == 'test'):
            self.delta = self.delta[:500]
        
        self.delta = np.array(self.delta)
        self.data_size = len(self.delta)

        self.grd_transform = input_transform(self.grd_size)
        self.sat_transform = input_transform(self.sat_size)

        # load train satellite image
        sat_img = Image.open(self.root + 'world.png').convert('RGB')
        self.sat_img = self.sat_transform(sat_img)

    def __len__(self):
        return self.data_size

    def __getitem__(self, index):

        # load train ground image
        # grd_img = Image.open(self.grd_img_root + str(index) + '.png')
        # if(self.data_size < 4000):
        #     grd_img = Image.open(self.grd_img_root + str(index + 7000) + '.png')
        
        shutil.copy(self.grd_img_root + str(index) + '.png', "results/carla/2/grd.png")
        shutil.copy(self.root + 'world.png', "results/carla/2/sat.png")
        shutil.copy(self.pc_root + str(index) + '.txt', "results/carla/2/pc.txt")
        
        grd_img = Image.open(self.grd_img_root + str(index) + '.png')
        grd_img = self.grd_transform(grd_img)

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

        return self.sat_img, grd_img, gt