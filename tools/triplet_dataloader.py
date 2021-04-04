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
import numpy as np

# from tools.visualize import showpoints

def ind2vec(ind, N=None):
    ind = np.asarray(ind)
    if N is None:
        N = ind.max() + 1
    return np.arange(N) == np.repeat(ind, N, axis=1)


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
    all_label = np.concatenate(all_label, axis=0) # DEBUG:
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

class TripletDataloader(Dataset):
    def __init__(self, dataset, num_points, num_classes, dataset_dir, partition='train'):
        self.dataset = dataset
        self.dataset_dir = dataset_dir

        if self.dataset == 'ModelNet40':
            self.data, self.label, self.img_lst = load_data(partition,self.dataset_dir)
        else:
            self.data, self.label, self.img_lst = load_modelnet10_data(partition,self.dataset_dir)
        self.num_points = num_points
        self.partition = partition
        self.num_classes=num_classes

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
        
        #randomly select one image from the 12 images for each object
        img_idx = random.randint(0, 179)
        img_names =self.dataset_dir+'ModelNet40-Images-180/%s/%s/%s.%d.png' % (names[0], names[1][:-4], names[1][:-4], img_idx)
        img = Image.open(img_names).convert('RGB')

        img_idx2 = random.randint(0, 179)
        while img_idx == img_idx2:
            img_idx2 = random.randint(0, 179)

        img_name2 = self.dataset_dir+'ModelNet40-Images-180/%s/%s/%s.%d.png' % (names[0], names[1][:-4], names[1][:-4], img_idx2)
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

        return pointcloud, label, img, img2


    def get_mesh(self, item):
        
        names = self.img_lst[item]
        names = names.split('/')
        mesh_path = self.dataset_dir+'ModelNet40_Mesh/' + names[0] + '/train/' + names[1][:-4] + '.npz'

        data = np.load(mesh_path)
        face = data['face']
        neighbor_index = data['neighbor_index']

        # data augmentation
        if self.partition == 'train':
            sigma, clip = 0.01, 0.05
            jittered_data = np.clip(sigma * np.random.randn(*face[:, :12].shape), -1 * clip, clip)
            face = np.concatenate((face[:, :12] + jittered_data, face[:, 12:]), 1)

        # fill for n < max_faces with randomly picked faces
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


    def check_exist(self, item):
        names = self.img_lst[item]
        names = names.split('/')
        mesh_path = self.dataset_dir+'ModelNet40_Mesh/' + names[0] + '/train/' + names[1][:-4] + '.npz'

        return os.path.isfile(mesh_path)

    def __getitem__(self, item):
        
        while not self.check_exist(item):
            idx = random.randint(0, len(self.data)-1)
            item = idx

        pt, target, img, img_v = self.get_data(item)
        centers, corners, normals, neighbor_index = self.get_mesh(item)
        pt = torch.from_numpy(pt)
        target_vec = np.zeros((1, self.num_classes))
        target_vec[0, target] = 1

        return pt, img, img_v, centers, corners, normals, neighbor_index, target, target_vec

    def __len__(self):
        return self.data.shape[0]

