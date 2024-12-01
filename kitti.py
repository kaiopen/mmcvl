import os
from torch.utils.data import Dataset
import PIL.Image
import torch
import numpy as np
import random
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torchvision.transforms.functional as TF
import math
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import torch
import torch.nn as nn
import numpy as np
import open3d as o3d
import shutil

torch.manual_seed(17)
np.random.seed(0)

Default_lat = 49.015
Satmap_zoom = 18
SatMap_original_sidelength = 512 
SatMap_process_sidelength = 512 
satmap_dir = '/home/rv/cst/satmap'
grdimage_dir = '/home/rv/cst/kitti/'
oxts_dir = 'oxts/data'  
points_cloud_dir = 'velodyne_points/data'
left_color_camera_dir = 'image_02/data'
CameraGPS_shift_left = [1.08, 0.26]

def get_meter_per_pixel(lat=Default_lat, zoom=Satmap_zoom, scale=SatMap_process_sidelength/SatMap_original_sidelength):
    meter_per_pixel = 156543.03392 * np.cos(lat * np.pi/180.) / (2**zoom)	
    meter_per_pixel /= 2 # because use scale 2 to get satmap 
    meter_per_pixel /= scale
    return meter_per_pixel

class SatGrdDataset(Dataset):
    def __init__(self, root, file,
                 transform=None, shift_range_lat=20, shift_range_lon=20, rotation_range=10):
        self.root = root

        self.meter_per_pixel = get_meter_per_pixel(scale=1)
        self.shift_range_meters_lat = shift_range_lat  # in terms of meters
        self.shift_range_meters_lon = shift_range_lon  # in terms of meters
        self.shift_range_pixels_lat = shift_range_lat / self.meter_per_pixel  # shift range is in terms of pixels
        self.shift_range_pixels_lon = shift_range_lon / self.meter_per_pixel  # shift range is in terms of pixels

        self.rotation_range = rotation_range  # in terms of degree

        self.skip_in_seq = 2  # skip 2 in sequence: 6,3,1~
        if transform != None:
            self.satmap_transform = transform[0]
            self.grdimage_transform = transform[1]

        self.pro_grdimage_dir = '/home/rv/cst/kitti/'

        self.satmap_dir = satmap_dir

        with open(file, 'r') as f:
            file_name = f.readlines()

        self.file_name = [file[:-1] for file in file_name]

    def __len__(self):
        return len(self.file_name)

    def get_file_list(self):
        return self.file_name

    def __getitem__(self, idx):
        # read cemera k matrix from camera calibration files, day_dir is first 10 chat of file name

        file_name = self.file_name[idx]
        day_dir = file_name[:10]
        drive_dir = file_name[:38]
        image_no = file_name[38:]

        # =================== read satellite map ===================================
        SatMap_name = os.path.join(self.root, self.satmap_dir, file_name)
        with PIL.Image.open(SatMap_name, 'r') as SatMap:
            sat_map = SatMap.convert('RGB')

        # =================== initialize some required variables ============================
        grd_left_imgs = torch.tensor([])
        image_no = file_name[38:]

        # oxt: such as 0000000000.txt
        oxts_file_name = os.path.join(self.root, grdimage_dir, drive_dir, oxts_dir,
                                      image_no.lower().replace('.png', '.txt'))

        pcloud_file_name = os.path.join(self.root, grdimage_dir, drive_dir, points_cloud_dir,
                                      image_no.lower().replace('.png', '.bin'))
        
        grd_points = np.fromfile(pcloud_file_name, dtype=np.float32, count=-1).reshape(-1, 4)
        grd_points = grd_points[:,0:3]
 
        scan = o3d.geometry.PointCloud()
        scan.points = o3d.utility.Vector3dVector(grd_points)
        scan = scan.voxel_down_sample(voxel_size=0.3)
        grd_points = np.asarray(scan.points)

        points_num = grd_points.shape[0]
        padded_grd_points = np.pad(grd_points, ((0, 50000 - points_num), (0, 0)), mode='constant')
        padded_grd_points = torch.from_numpy(padded_grd_points)
        points_num = torch.Tensor([points_num]).long()

        with open(oxts_file_name, 'r') as f:
            content = f.readline().split(' ')
            # get heading
            lat = float(content[0])
            lon = float(content[1])
            heading = float(content[5])

            left_img_name = os.path.join(self.root, self.pro_grdimage_dir, drive_dir, left_color_camera_dir,
                                         image_no.lower())
            with PIL.Image.open(left_img_name, 'r') as GrdImg:
                grd_img_left = GrdImg.convert('RGB')
                if self.grdimage_transform is not None:
                    grd_img_left = self.grdimage_transform(grd_img_left)

            grd_left_imgs = torch.cat([grd_left_imgs, grd_img_left.unsqueeze(0)], dim=0)
        
        sat_rot = sat_map.rotate((-heading) / np.pi * 180) # make the east direction the vehicle heading
        sat_align_cam = sat_rot.transform(sat_rot.size, PIL.Image.AFFINE,
                                          (1, 0, CameraGPS_shift_left[0] / self.meter_per_pixel,
                                           0, 1, CameraGPS_shift_left[1] / self.meter_per_pixel),
                                          resample=PIL.Image.BILINEAR) 
        
        # randomly generate shift
        gt_shift_x = np.random.uniform(-1, 1)  # --> right as positive, parallel to the heading direction
        gt_shift_y = np.random.uniform(-1, 1)  # --> up as positive, vertical to the heading direction
        
        sat_rand_shift = \
            sat_align_cam.transform(
                sat_align_cam.size, PIL.Image.AFFINE,
                (1, 0, gt_shift_x * self.shift_range_pixels_lon,
                 0, 1, -gt_shift_y * self.shift_range_pixels_lat),
                resample=PIL.Image.BILINEAR)
        
        # randomly generate roation
        random_ori = np.random.uniform(-1, 1) * self.rotation_range # 0 means the arrow in aerial image heading Easting, counter-clockwise increasing
        sat_rand_shift_rand_rot = sat_rand_shift.rotate(random_ori)
        
        sat_map =TF.center_crop(sat_rand_shift_rand_rot, SatMap_process_sidelength)
        
        # transform
        if self.satmap_transform is not None:
            sat_map = self.satmap_transform(sat_map)
        

        # gt heat map
        x_offset = int(gt_shift_x*self.shift_range_pixels_lon*np.cos(random_ori/180*np.pi) - gt_shift_y*self.shift_range_pixels_lat*np.sin(random_ori/180*np.pi)) # horizontal direction
        y_offset = int(-gt_shift_y*self.shift_range_pixels_lat*np.cos(random_ori/180*np.pi) - gt_shift_x*self.shift_range_pixels_lon*np.sin(random_ori/180*np.pi)) # vertical direction
        
        x, y = np.meshgrid(np.linspace(-256+x_offset,256+x_offset,512), np.linspace(-256+y_offset,256+y_offset,512))
        d = np.sqrt(x*x+y*y)
        sigma, mu = 4, 0.0
        gt = np.zeros([1, 512, 512], dtype=np.float32)
        gt[0, :, :] = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
        # gt = [x_offset, y_offset]
        gt = torch.tensor(gt)
        
        # orientation gt
        orientation_angle = 90 - random_ori 
        if orientation_angle < 0:
            orientation_angle = orientation_angle + 360
        elif orientation_angle > 360:
            orientation_angle = orientation_angle - 360
        
        # gt_with_ori = np.zeros([16, 512, 512], dtype=np.float32)
        # index = int(orientation_angle // 22.5)
        # ratio = (orientation_angle % 22.5) / 22.5
        # if index == 0:
        #     gt_with_ori[0, :, :] = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) ) * (1-ratio)
        #     gt_with_ori[15, :, :] = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) ) * ratio
        # else:
        #     gt_with_ori[16-index, :, :] = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) ) * (1-ratio)
        #     gt_with_ori[16-index-1, :, :] = np.ex/media/hudi/disk0
        # orientation_map = torch.full([2, 512, 512], np.cos(orientation_angle * np.pi/180))
        # orientation_map[1,:,:] = np.sin(orientation_angle * np.pi/180)
        

        # return sat_map, grd_left_imgs[0], gt, padded_grd_points, points_num
        return sat_map, grd_left_imgs[0], gt
               
class SatGrdDatasetTest(Dataset):
    def __init__(self, root, file,
                 transform=None, shift_range_lat=20, shift_range_lon=20, rotation_range=10):
        self.root = root

        self.meter_per_pixel = get_meter_per_pixel(scale=1)
        self.shift_range_meters_lat = shift_range_lat  # in terms of meters
        self.shift_range_meters_lon = shift_range_lon  # in terms of meters
        self.shift_range_pixels_lat = shift_range_lat / self.meter_per_pixel  # shift range is in terms of meters
        self.shift_range_pixels_lon = shift_range_lon / self.meter_per_pixel  # shift range is in terms of meters

        self.rotation_range = rotation_range  # in terms of degree

        self.skip_in_seq = 2  # skip 2 in sequence: 6,3,1~
        if transform != None:
            self.satmap_transform = transform[0]
            self.grdimage_transform = transform[1]

        self.pro_grdimage_dir = '/home/rv/cst/kitti/'

        self.satmap_dir = satmap_dir

        with open(file, 'r') as f:
            file_name = f.readlines()

        self.file_name = [file[:-1] for file in file_name]
       

    def __len__(self):
        return len(self.file_name)

    def get_file_list(self):
        return self.file_name

    def __getitem__(self, idx):

        line = self.file_name[idx]
        file_name, gt_shift_x, gt_shift_y, theta = line.split(' ')
        day_dir = file_name[:10]
        drive_dir = file_name[:38]
        image_no = file_name[38:]
        
        # =================== read satellite map ===================================
        SatMap_name = os.path.join(self.root, self.satmap_dir, file_name)
        shutil.copy(SatMap_name, "results/kitti/2/sat.png")
        with PIL.Image.open(SatMap_name, 'r') as SatMap:
            sat_map = SatMap.convert('RGB')

        # =================== initialize some required variables ============================
        grd_left_imgs = torch.tensor([])
        grd_left_depths = torch.tensor([])
        # image_no = file_name[38:]

        # oxt: such as 0000000000.txt
        oxts_file_name = os.path.join(self.root, grdimage_dir, drive_dir, oxts_dir,
                                      image_no.lower().replace('.png', '.txt'))

        pcloud_file_name = os.path.join(self.root, grdimage_dir, drive_dir, points_cloud_dir,
                                      image_no.lower().replace('.png', '.bin'))
        
        grd_points = np.fromfile(pcloud_file_name, dtype=np.float32, count=-1).reshape(-1, 4)
        grd_points = grd_points[:,0:3]
        np.savetxt("results/kitti/2/pc.txt", grd_points)
 
        scan = o3d.geometry.PointCloud()
        scan.points = o3d.utility.Vector3dVector(grd_points)
        scan = scan.voxel_down_sample(voxel_size=0.3)
        grd_points = np.asarray(scan.points)

        points_num = grd_points.shape[0]
        padded_grd_points = np.pad(grd_points, ((0, 50000 - points_num), (0, 0)), mode='constant')
        padded_grd_points = torch.from_numpy(padded_grd_points)
        points_num = torch.Tensor([points_num]).long()

        with open(oxts_file_name, 'r') as f:
            content = f.readline().split(' ')
            # get heading
            lat = float(content[0])
            lon = float(content[1])
            heading = float(content[5])

            left_img_name = os.path.join(self.root, self.pro_grdimage_dir, drive_dir, left_color_camera_dir,
                                         image_no.lower())
            shutil.copy(left_img_name, "results/kitti/2/grd.png")
            with PIL.Image.open(left_img_name, 'r') as GrdImg:
                grd_img_left = GrdImg.convert('RGB')
                if self.grdimage_transform is not None:
                    grd_img_left = self.grdimage_transform(grd_img_left)

            grd_left_imgs = torch.cat([grd_left_imgs, grd_img_left.unsqueeze(0)], dim=0)
        
        sat_rot = sat_map.rotate(-heading / np.pi * 180)
        
        sat_align_cam = sat_rot.transform(sat_rot.size, PIL.Image.AFFINE,
                                          (1, 0, CameraGPS_shift_left[0] / self.meter_per_pixel,
                                           0, 1, CameraGPS_shift_left[1] / self.meter_per_pixel),
                                          resample=PIL.Image.BILINEAR)
        
        # load the shifts 
        gt_shift_x = -float(gt_shift_x)  # --> right as positive, parallel to the heading direction
        gt_shift_y = -float(gt_shift_y)  # --> up as positive, vertical to the heading direction

        sat_rand_shift = \
            sat_align_cam.transform(
                sat_align_cam.size, PIL.Image.AFFINE,
                (1, 0, gt_shift_x * self.shift_range_pixels_lon,
                 0, 1, -gt_shift_y * self.shift_range_pixels_lat),
                resample=PIL.Image.BILINEAR)

        random_ori = float(theta) * self.rotation_range # degree
        sat_rand_shift_rand_rot = sat_rand_shift.rotate(random_ori)
        
        sat_map =TF.center_crop(sat_rand_shift_rand_rot, SatMap_process_sidelength)

        # transform
        if self.satmap_transform is not None:
            sat_map = self.satmap_transform(sat_map)

        # gt heat map
        x_offset = int(gt_shift_x*self.shift_range_pixels_lon*np.cos(random_ori/180*np.pi) - gt_shift_y*self.shift_range_pixels_lat*np.sin(random_ori/180*np.pi)) # horizontal direction
        y_offset = int(-gt_shift_y*self.shift_range_pixels_lat*np.cos(random_ori/180*np.pi) - gt_shift_x*self.shift_range_pixels_lon*np.sin(random_ori/180*np.pi)) # vertical direction
        
        x, y = np.meshgrid(np.linspace(-256+x_offset,256+x_offset,512), np.linspace(-256+y_offset,256+y_offset,512))
        d = np.sqrt(x*x+y*y)
        sigma, mu = 4, 0.0
        gt = np.zeros([1, 512, 512], dtype=np.float32)
        gt[0, :, :] = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
        # gt = [x_offset, y_offset]
        gt = torch.tensor(gt)
        
        # orientation gt
        orientation_angle = 90 - random_ori 
        if orientation_angle < 0:
            orientation_angle = orientation_angle + 360
        elif orientation_angle > 360:
            orientation_angle = orientation_angle - 360
            
        # gt_with_ori = np.zeros([16, 512, 512], dtype=np.float32)
        # index = int(orientation_angle // 22.5)
        # ratio = (orientation_angle % 22.5) / 22.5
        # if index == 0:
        #     gt_with_ori[0, :, :] = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) ) * (1-ratio)
        #     gt_with_ori[15, :, :] = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) ) * ratio
        # else:
        #     gt_with_ori[16-index, :, :] = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) ) * (1-ratio)
        #     gt_with_ori[16-index-1, :, :] = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) ) * ratio
        # gt_with_ori = torch.tensor(gt_with_ori)
        
        # orientation_map = torch.full([2, 512, 512], np.cos(orientation_angle * np.pi/180))
        # orientation_map[1,:,:] = np.sin(orientation_angle * np.pi/180)
        
        
        
        return sat_map, grd_left_imgs[0], gt
        # return sat_map, grd_left_imgs[0], gt, padded_grd_points, points_num