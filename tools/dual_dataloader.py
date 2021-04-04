#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: data.py
@Time: 2018/10/13 6:21 PM
"""

import os
import sys
import glob
import h5py
import json
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms, datasets
import torch
# from tools.visualize import showpoints


def load_data(partition):
    # download()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = './dataset/'
    # print(DATA_DIR)
    all_data = []
    all_label = []
    img_lst = []
    # print(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition))
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
        split = h5_name[-4]
        jason_name = './dataset/modelnet40_ply_hdf5_2048/ply_data_' + partition +'_' + split + '_id2file.json'

        # print('index name :',idx_name)
        with open(jason_name) as json_file:
            images = json.load(json_file)

        img_lst = img_lst + images

        f = h5py.File(h5_name)

        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        # print(data.shape, label.shape)
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0) # DEBUG:
    print(len(all_data), len(all_label), len(img_lst))
    return all_data, all_label, img_lst


def load_modelnet10_data(partition):
    # download()
    # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = './dataset/'
    # print(DATA_DIR)
    all_data = []
    all_label = []
    img_lst = []
    # print(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition))
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet10_hdf5_2048', '%s*.h5'%partition)):
    # for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet10_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
        split = h5_name[-4]
        # jason_name = './dataset/modelnet10_ply_hdf5_2048/ply_data_' + partition +'_' + split + '_id2file.json'
        jason_name = './dataset/modelnet10_hdf5_2048/'+partition + split + '_id2file.json'

        # print('index name :',idx_name)
        with open(jason_name) as json_file:
            images = json.load(json_file)
        # print(images)

        img_lst = img_lst + images
        # for img in img_lst:
        #     print(img)

        f = h5py.File(h5_name)

        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        # print(data.shape, label.shape)
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0) # DEBUG:
    print(len(all_data), len(all_label), len(img_lst))
    return all_data, all_label, img_lst


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud

def rotate_pointcloud(pointcloud):
    theta = np.pi*2 * np.random.rand()
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    pointcloud[:,[0,2]] = pointcloud[:,[0,2]].dot(rotation_matrix) # random rotation (x,z)
    return pointcloud

def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud

def random_scale(pointcloud, scale_low=0.8, scale_high=1.25):
    N, C = pointcloud.shape
    scale = np.random.uniform(scale_low, scale_high)
    pointcloud = pointcloud*scale
    return pointcloud

class SingleViewDataloader(Dataset):
    def __init__(self, dataset, num_points, partition='train'):
        self.dataset = dataset
        if self.dataset == 'ModelNet40':
            self.data, self.label, self.img_lst = load_data(partition)
        else:
            self.data, self.label, self.img_lst = load_modelnet10_data(partition)
        self.num_points = num_points
        self.partition = partition

        self.img_train_transform = transforms.Compose([
            transforms.RandomCrop(224),
            transforms.Resize(112),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.img_test_transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.Resize(112),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


    def get_data(self, item):

        # Get Image Data first
        names = self.img_lst[item]
        names = names.split('/')
        
        #random select one image from the 12 images for each object
        img_idx = random.randint(0, 179)
        img_names = './dataset/ModelNet40-Images-180/%s/%s/%s.%d.png' % (names[0], names[1][:-4], names[1][:-4], img_idx)
        img = Image.open(img_names).convert('RGB')

        img_idx2 = random.randint(0, 179)
        while img_idx == img_idx2:
            img_idx2 = random.randint(0, 179)

        img_name2 = './dataset/ModelNet40-Images-180/%s/%s/%s.%d.png' % (names[0], names[1][:-4], names[1][:-4], img_idx2)
        img2 = Image.open(img_name2).convert('RGB')

        label = self.label[item]

        pointcloud = self.data[item]
        choice = np.random.choice(len(pointcloud), self.num_points, replace=True)
        pointcloud = pointcloud[choice, :]

        if self.partition == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            pointcloud = rotate_pointcloud(pointcloud)
            pointcloud = jitter_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)

            img = self.img_train_transform(img)
            img2 = self.img_train_transform(img2)
        else:
            img = self.img_test_transform(img)
            img2 = self.img_test_transform(img2)

        # print(pointcloud.shape, label, img.size())
        return pointcloud, label, img, img2

    def __getitem__(self, item):
        #name of this point cloud
        idx = random.randint(0, len(self.data)-1)
        # while(idx == item):
        #     idx = random.randint(0, len(self.data))

        pt, target, img, img_v = self.get_data(item)
        # pt2, label2, img2, img2_v = self.get_data(idx)
        # pos = 1
        # neg = 0
        pt = torch.from_numpy(pt)
        # pt2 = torch.from_numpy(pt2)
        # return pt1, pt2, img1, img2, img1_v, img2_v, label1, label2, pos, neg
        return pt, img, img_v, target

    def __len__(self):
        return self.data.shape[0]

class MultiViewDataloader(Dataset):
    def __init__(self, num_points, num_views, partition='train'):
        self.data, self.label, self.img_lst = load_data(partition)
        self.num_points = num_points
        self.num_views = num_views
        self.partition = partition

        if partition == 'train':
            self.img_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])
        else:
            self.img_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])


    def get_data(self, item):
        # ?
        pointcloud = self.data[item][:self.num_points]
        img_names = self.img_lst[item]
        # print(img_names)
        label = self.label[item]

        #random select K images from the 12 images for each object
        random.shuffle(img_names)

        imgs = []
        for img_idx in range(self.num_views):
            im = Image.open('./modelnet40_images_new_12x/' + img_names[img_idx]).convert('RGB')
            im = self.img_transform(im)
            imgs.append(im)


        # if self.partition == 'train':
        #     pointcloud = translate_pointcloud(pointcloud)
        #     np.random.shuffle(pointcloud)

        # print(pointcloud.shape, label, img.size())
        return pointcloud, label, imgs

    def __getitem__(self, item):
        #name of this point cloud
        idx = random.randint(0, len(self.data)-1)
        while(idx == item):
            idx = random.randint(0, len(self.data))

        pt1, label1, imgs1 = self.get_data(item)
        pt2, label2, imgs2 = self.get_data(idx)
        pos = 1
        neg = 0
        imgs1 =  torch.stack(imgs1)
        imgs2 = torch.stack(imgs2)
        # print('--------->:  ', imgs1.size())
        return pt1, pt2, imgs1, imgs2, label1, label2, pos, neg

    def __len__(self):
        return self.data.shape[0]

if __name__ == '__main__':
    
    train_set = SingleViewDataloader(num_points = 1024, partition='train')
    
    data_loader_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=False,num_workers=1)
    
    cnt = 0
    for data in data_loader_loader:
        pt1, pt2, img1, img2, img1_v, img2_v, label1, label2, pos, neg  = data
        pt1 = pt1.numpy()
        pt1 = pt1[0,:,:]
        # print(pt1.shape)
        # print(pt1)
        # print(np.amin(pt1), np.amax(pt1))
        # showpoints(pt1)
