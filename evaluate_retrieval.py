from tools.test_dataloader import TestDataloader
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
import argparse
import torch.optim as optim
import time
import torchvision.models as models
from models.meshnet import MeshNet
from models.dgcnn import get_graph_feature
import numpy as np
import copy
from collections import defaultdict
import sys
import warnings
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as metrics
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from sklearn.preprocessing import normalize
import scipy

def extract(args):
    img_net = torch.load('./checkpoints/%s/%d-img_net.pkl'%(args.model_folder, args.iterations), map_location=lambda storage, loc: storage)
    img_net = img_net.eval()
    dgcnn = torch.load('./checkpoints/%s/%d-pt_net.pkl'%(args.model_folder, args.iterations), map_location=lambda storage, loc: storage)
    dgcnn = dgcnn.eval()
    mesh_net= torch.load('./checkpoints/%s/%d-mesh_net.pkl'%(args.model_folder, args.iterations), map_location=lambda storage, loc: storage)
    mesh_net = mesh_net.eval()
    torch.cuda.empty_cache()
    #################################    
    test_set = TestDataloader(dataset=args.dataset, num_points = 1024 , dataset_dir = args.dataset_dir, partition= 'test')
    data_loader_loader = torch.utils.data.DataLoader(test_set, batch_size=1,shuffle=False, num_workers=8)
    print('length of the dataset: ', len(test_set))
    #################################
    img_feat_1 = np.zeros((len(test_set),512))
    img_feat_2 = np.zeros((len(test_set),512))
    img_feat_4 = np.zeros((len(test_set),512))
    pt_feat = np.zeros((len(test_set), 512))
    mesh_feat = np.zeros((len(test_set), 512))
    label = np.zeros((len(test_set)))
    #################################
    iteration = 0
    for data in data_loader_loader: 
        print(iteration)
        pt, img_list, centers, corners, normals, neighbor_index, target = data
        #################################
        img_v1, img_v2, img_v3 , img_v4= img_list
        img_v1 = Variable(img_v1).to('cuda')
        img_v2 = Variable(img_v2).to('cuda')
        img_v3 = Variable(img_v3).to('cuda')
        img_v4 = Variable(img_v4).to('cuda')        
        #################################
        target = target[:,0]
        target = Variable(target).to('cuda')
        pt = Variable(pt).to('cuda') 
        pt = pt.permute(0,2,1)
        centers = Variable(torch.cuda.FloatTensor(centers.cuda()))
        corners = Variable(torch.cuda.FloatTensor(corners.cuda()))
        normals = Variable(torch.cuda.FloatTensor(normals.cuda()))
        neighbor_index = Variable(torch.cuda.LongTensor(neighbor_index.cuda()))
        ##########################################
        img_net = img_net.to('cuda')
        feat_1_view = img_net(img_v1, img_v1)
        feat_2_views = img_net(img_v1, img_v2)
        feat_4_views = 0.5*(img_net(img_v1, img_v2)+img_net(img_v3, img_v4))
        dgcnn = dgcnn.to('cuda')
        cloud_feat = dgcnn(pt)    
        mesh_net = mesh_net.to('cuda')
        M_feat = mesh_net(centers, corners, normals, neighbor_index)
        ########################################
        img_feat_1[iteration,:] = img_feat_1[iteration,:] + feat_1_view.data.cpu().numpy()
        img_feat_2[iteration,:] = img_feat_2[iteration,:] + feat_2_views.data.cpu().numpy()
        img_feat_4[iteration,:] = img_feat_4[iteration,:] + feat_4_views.data.cpu().numpy()
        pt_feat[iteration,:] = cloud_feat.data.cpu().numpy()
        mesh_feat[iteration,:] = M_feat.data.cpu().numpy()
        label[iteration] = target.data.cpu().numpy()
        iteration = iteration + 1
    np.save(args.save+'/img_feat_{}.np'.format(1),img_feat_1)    
    np.save(args.save+'/img_feat_{}.np'.format(2),img_feat_2)    
    np.save(args.save+'/img_feat_{}.np'.format(4),img_feat_4)    
    np.save(args.save+'/pt_feat.np',pt_feat)    
    np.save(args.save+'/mesh_feat.np',mesh_feat)    
    np.save(args.save+'/label.np',label) 

def fx_calc_map_label(view_1, view_2, label_test):
    dist = scipy.spatial.distance.cdist(view_1, view_2, 'cosine') #rows view_1 , columns view 2 
    ord = dist.argsort()
    numcases = dist.shape[0]
    res = []
    for i in range(numcases):
        order = ord[i] 
        p = 0.0
        r = 0.0
        for j in range(numcases):
            if label_test[i] == label_test[order[j]]: 
                r += 1
                p += (r / (j + 1))
        if r > 0:
            res += [p / r]
        else:
            res += [0]
    return np.mean(res)


def eval_func(img_pairs):
    print('number of img views: ',img_pairs)
    img_feat = np.load(args.save+'/img_feat_{}.np.npy'.format(img_pairs))    
    pt_feat = np.load(args.save+'/pt_feat.np.npy')    
    mesh_feat = np.load(args.save+'/mesh_feat.np.npy')    
    label = np.load(args.save+'/label.np.npy') 
    ########################################
    img_test = normalize(img_feat, norm='l1', axis=1)
    cloud_test = normalize(pt_feat, norm='l1', axis=1)
    mesh_test = normalize(mesh_feat, norm='l1', axis=1)
    ########################################
    par_list = [
    (img_test,img_test,'Image2Image'), 
    (img_test,mesh_test,'Image2Mesh'),        
    (img_test,cloud_test,'Image2Point'), 
    (mesh_test,mesh_test,'Mesh2Mesh'),
    (mesh_test,img_test,'Mesh2Image'),
    (mesh_test,cloud_test,'Mesh2Point'),
    (cloud_test,cloud_test,'Point2Point'),
    (cloud_test,img_test,'Point2Image'),
    (cloud_test,mesh_test,'Point2Mesh')]
    ########################################
    for index in range(9):
        view_1,view_2,name = par_list[index]
        print(name+ '---------------------------')
        acc = fx_calc_map_label(view_1,view_2 , label)
        acc_round = round(acc*100,2)
        print(str(acc_round))





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cross Modal Retrieval for Point Cloud, Mesh, and Image Models')

    parser.add_argument('--dataset', type=str, default='ModelNet40', metavar='dataset',help='ModelNet10 or ModelNet40')
                        
    parser.add_argument('--dataset_dir', type=str, default='/media/super-server/32d81dc0-e480-40a0-9cc6-4a371fcd2824/CrossModalRetrieval/CrossModalRetrieval/dataset/',
    metavar='dataset_dir',help='dataset_dir')

    parser.add_argument('--model_folder', type=str,  default='M40',help='path to load model')
                        
    parser.add_argument('--iterations', type=int,  default=55000,help='iteration to load the model')

    parser.add_argument('--gpu_id', type=str,  default='0,1,2,3',help='GPU used to train the network')
                                                
    parser.add_argument('--save', type=str,  default='extracted_features/M40',help='save features')
                        

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    torch.backends.cudnn.enabled = False 

    if not os.path.exists(args.save):
        os.mkdir(args.save)
    
    extract(args)
    eval_func(1)
    eval_func(2)
    eval_func(4)

