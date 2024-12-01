import numpy as np
import math
import sys
import os

import numpy as np
import time
import torch
import sklearn.metrics as metrics
from sklearn.neighbors import NearestNeighbors
import random
from torch import nn
from torch_scatter import scatter_max
import copy


def farthest_point_sample(point, npoint=1024):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point, centroids.astype(np.int32)



def multi_layer_downsampling_random(points_xyz, base_voxel_size, levels=[1],
    add_rnd3d=False):
    """Downsample the points at different scales by randomly select a point
    within a voxel cell.

    Args:
        points_xyz: a [N, D] matrix. N is the total number of the points. D is
        the dimension of the coordinates.
        base_voxel_size: scalar, the cell size of voxel.
        level_configs: a dict of 'level', 'graph_gen_method',
        'graph_gen_kwargs', 'graph_scale'.
        add_rnd3d: boolean, whether to add random offset when downsampling.

    returns: vertex_coord_list, keypoint_indices_list
    """
    xmax, ymax, zmax = np.amax(points_xyz, axis=0)
    xmin, ymin, zmin = np.amin(points_xyz, axis=0)
    xyz_offset = np.asarray([[xmin, ymin, zmin]])
    xyz_zeros = np.asarray([0, 0, 0], dtype=np.float32)
    vertex_coord_list = [points_xyz]
    keypoint_indices_list = []
    last_level = 0
    for level in levels:
        last_points_xyz = vertex_coord_list[-1]
        if np.isclose(last_level, level):
            # same downsample scale (gnn layer), just copy it
            vertex_coord_list.append(np.copy(last_points_xyz))
            keypoint_indices_list.append(
                np.expand_dims(np.arange(len(last_points_xyz)), axis=1))
        else:
            if not add_rnd3d:
                xyz_idx = (last_points_xyz - xyz_offset) \
                    // (base_voxel_size*level)
            else:
                xyz_idx = (
                    last_points_xyz - xyz_offset
                    + base_voxel_size * level * np.random.random((1, 3))
                ) // (base_voxel_size * level)
            xyz_idx = xyz_idx.astype(np.int32)

            # Assign an ID to a vexol.
            dim_x, dim_y, dim_z = np.amax(xyz_idx, axis=0) + 1
            keys = xyz_idx[:, 0] + xyz_idx[:, 1] * dim_x \
                + xyz_idx[:, 2] * dim_y * dim_x  # flatten
            # num_points = xyz_idx.shape[0]

            # Assign points to voxels.
            voxels_idx = {}
            for pidx in range(len(last_points_xyz)):
                key = keys[pidx]
                if key in voxels_idx:
                    voxels_idx[key].append(pidx)
                else:
                    voxels_idx[key] = [pidx]

            downsampled_xyz = []
            downsampled_xyz_idx = []
            for key in voxels_idx:  # for each voxel
                # Choose randomly a point in the voxel.
                center_idx = random.choice(voxels_idx[key])
                downsampled_xyz.append(last_points_xyz[center_idx])
                downsampled_xyz_idx.append(center_idx)
            vertex_coord_list.append(np.array(downsampled_xyz))
            keypoint_indices_list.append(
                np.expand_dims(np.array(downsampled_xyz_idx), axis=1)
            )
        last_level = level

    keypoint_indices_list = np.array(keypoint_indices_list).reshape(-1)
    # print(keypoint_indices_list.shape)
    _, indces = farthest_point_sample(points_xyz[keypoint_indices_list])
    keypoint_indices_list = keypoint_indices_list[indces]
    vertex_coord_list[1] = vertex_coord_list[1][indces]

    return vertex_coord_list, keypoint_indices_list


def gen_multi_level_local_graph_v3(
    points_xyz, base_voxel_size, add_rnd3d=False,
    downsample_method='center'):
    """Generating graphs at multiple scale. This function enforce output
    vertices of a graph matches the input vertices of next graph so that
    gnn layers can be applied sequentially.

    Args:
        points_xyz: a [N, D] matrix. N is the total number of the points. D is
        the dimension of the coordinates.
        base_voxel_size: scalar, the cell size of voxel.
        level_configs: a dict of 'level', 'graph_gen_method',
        'graph_gen_kwargs', 'graph_scale'.
        add_rnd3d: boolean, whether to add random offset when downsampling.
        downsample_method: string, the name of downsampling method.
    returns: vertex_coord_list, keypoint_indices_list, edges_list
    """
    if isinstance(base_voxel_size, list):
        base_voxel_size = np.array(base_voxel_size)
    # Gather the downsample scale for each graph
#     scales = [config['graph_scale'] for config in level_configs]
    # Generate vertex coordinates
    if downsample_method=='center':
        vertex_coord_list, keypoint_indices_list = \
            multi_layer_downsampling_select(
                points_xyz, base_voxel_size)
    if downsample_method=='random':
        vertex_coord_list, keypoint_indices_list = \
            multi_layer_downsampling_random(
                points_xyz, base_voxel_size)

    # Create edges
    keypoint_indices_list = np.array(keypoint_indices_list).reshape(-1)
    vertex_coord_list.append(points_xyz[keypoint_indices_list])
    edges_list = []
    radius = [1, 3]
    num_neighbors = [-1, 64]
    for i in range(2):
        points_xyz = vertex_coord_list[i]
        center_xyz = vertex_coord_list[i+1]
        vertices = gen_disjointed_rnn_local_graph_v3(points_xyz, center_xyz, radius[i], num_neighbors[i])
        edges_list.append(vertices)
    return vertex_coord_list, keypoint_indices_list, edges_list


def gen_disjointed_rnn_local_graph_v3(
    points_xyz, center_xyz, radius, num_neighbors,
    neighbors_downsample_method='random',
    scale=None):
    """Generate a local graph by radius neighbors.
    """
    if scale is not None:
        scale = np.array(scale)
        points_xyz = points_xyz/scale
        center_xyz = center_xyz/scale
    nbrs = NearestNeighbors(
        radius=radius,algorithm='ball_tree', n_jobs=1, ).fit(points_xyz)
    indices = nbrs.radius_neighbors(center_xyz, return_distance=False)
    if num_neighbors > 0:
        if neighbors_downsample_method == 'random':
            indices = [neighbors if neighbors.size <= num_neighbors else
                np.random.choice(neighbors, num_neighbors, replace=False)
                for neighbors in indices]
    vertices_v = np.concatenate(indices)
    vertices_i = np.concatenate(
        [i*np.ones(neighbors.size, dtype=np.int32)
            for i, neighbors in enumerate(indices)])
    vertices = np.array([vertices_v, vertices_i]).transpose()
    return vertices




def point_cloud_process(points):
    vertex_coord_list, keypoint_indices_list, edges_list = gen_multi_level_local_graph_v3(points, 1, add_rnd3d=False,downsample_method='random')

    input_v = torch.zeros(len(vertex_coord_list[0]))
    point_features, point_coordinates, keypoint_indices, set_indices = input_v.reshape(-1,1), vertex_coord_list[0], keypoint_indices_list.reshape(-1,1), edges_list
    set_indices[0] = torch.tensor(set_indices[0],dtype=torch.long)
    set_indices[1] = torch.tensor(set_indices[1],dtype=torch.long)
    keypoint_indices = torch.tensor(keypoint_indices,dtype=torch.long)
    point_coordinates = torch.Tensor(point_coordinates)
    #point_set_pooling = PointSetPooling()
    #set_features = point_set_pooling(point_features,point_coordinates,keypoint_indices,set_indices)

    return set_indices[1], [point_features, point_coordinates, keypoint_indices, set_indices[0]]