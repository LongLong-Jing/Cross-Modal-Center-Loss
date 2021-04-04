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


def load_data(partition,dataset_dir):
    all_data = []
    all_label = []
    img_lst = []
    for h5_name in glob.glob(os.path.join(dataset_dir, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
        split = h5_name[-4]
        jason_name = dataset_dir+'modelnet40_ply_hdf5_2048/ply_data_' + partition +'_' + split + '_id2file.json'
        with open(jason_name) as json_file:
            images = json.load(json_file)
        img_lst = img_lst + images
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0) 
    print(len(all_data), len(all_label), len(img_lst))
    return all_data, all_label, img_lst


def load_modelnet10_data(partition,dataset_dir):
    all_data = []
    all_label = []
    img_lst = []
    for h5_name in glob.glob(os.path.join(dataset_dir, 'modelnet10_hdf5_2048', '%s*.h5'%partition)):
        split = h5_name[-4]
        jason_name = dataset_dir+'modelnet10_hdf5_2048/'+partition + split + '_id2file.json'
        with open(jason_name) as json_file:
            images = json.load(json_file)
        img_lst = img_lst + images
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0) # DEBUG:
    print(len(all_data), len(all_label), len(img_lst))
    return all_data, all_label, img_lst

class TestDataloader(Dataset):
    def __init__(self, dataset, num_points, dataset_dir, partition='test'):
        self.dataset_dir = dataset_dir
        self.dataset = dataset
        if self.dataset == 'ModelNet40':
            self.data, self.label, self.img_lst = load_data(partition,self.dataset_dir)
        else:
            self.data, self.label, self.img_lst = load_modelnet10_data(partition,self.dataset_dir)

        self.num_points = num_points
        self.partition = partition

        self.img_transform = transforms.Compose([
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
        ##############################
        #random select one image from the images for each object
        img_list=[]
        img_index_list=[]
        for view in range(4):
            img_idx = random.randint(0, 179)
            while img_idx in img_index_list:
                img_idx = random.randint(0, 179)
            img_index_list.append(img_idx)
            img_names = self.dataset_dir+'ModelNet40-Images-180/%s/%s/%s.%d.png' % ( names[0], names[1][:-4], names[1][:-4], img_idx)
            img = Image.open(img_names).convert('RGB')
            img = self.img_transform(img)
            img_list.append(img)
        ##############################
        label = self.label[item]
        ##############################
        pointcloud = self.data[item]
        choice = np.random.choice(len(pointcloud), self.num_points, replace=True)
        pointcloud = pointcloud[choice, :]
        return pointcloud, label, img_list


    def check_exist(self, item):
        print('inside check_exist')
        names = self.img_lst[item]
        names = names.split('/')
        mesh_path = os.path.join(self.dataset_dir+'ModelNet40_Mesh/', names[0], self.partition, '%s.npz'%(names[1][:-4]))
        return os.path.isfile(mesh_path)


    def get_mesh(self, item):
        names = self.img_lst[item]
        names = names.split('/')
        mesh_path = os.path.join(self.dataset_dir+'ModelNet40_Mesh/', names[0], self.partition, '%s.npz'%(names[1][:-4]))
        data = np.load(mesh_path)
        face = data['face']
        neighbor_index = data['neighbor_index']
        max_faces = 1024 
        num_point = len(face)
        if num_point < max_faces:
            fill_face = []
            fill_neighbor_index = []
            for i in range(max_faces - num_point):
                index = np.random.randint(0, num_point)
                fill_face.append(face[index])
                fill_neighbor_index.append(neighbor_index[index])
            face = np.concatenate((face, np.array(fill_face)))
            neighbor_index = np.concatenate((neighbor_index, np.array(fill_neighbor_index)))
        # to tensor
        face = torch.from_numpy(face).float()
        neighbor_index = torch.from_numpy(neighbor_index).long()
        # reorganize
        face = face.permute(1, 0).contiguous()
        centers, corners, normals = face[:3], face[3:12], face[12:]
        corners = corners - torch.cat([centers, centers, centers], 0)
        return centers, corners, normals, neighbor_index

    def __getitem__(self, item):

        if self.dataset == 'ModelNet40' and item in [312, 1091, 1178, 1832]:
            while item in [312, 1091, 1178, 1832]:
                idx = random.randint(0, len(self.data)-1)
                item = idx

        pt, target, img_list = self.get_data(item)
        pt = torch.from_numpy(pt)
        centers, corners, normals, neighbor_index = self.get_mesh(item)
        return pt, img_list, centers, corners, normals, neighbor_index, target


    def __len__(self):
        return self.data.shape[0]

