from __future__ import division, absolute_import
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
from models.dgcnn import DGCNN
from models.meshnet import MeshNet
from models.SVCNN import SingleViewNet, CorrNet
from tools.triplet_dataloader import TripletDataloader
from tools.utils import calculate_accuracy
from center_loss import CrossModalCenterLoss
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
from torch.utils.tensorboard import SummaryWriter

def training(args):
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    img_net = SingleViewNet(pre_trained = None)
    pt_net = DGCNN(args)
    mesh_net = MeshNet()
    model = CorrNet(img_net, pt_net, mesh_net,num_classes=args.num_classes)

    model.train(True)
    model = model.to('cuda')
    model = torch.nn.DataParallel(model)
    model.train(True)


    writer = SummaryWriter(os.path.join(args.save, 'summary'))
    
    #cross entropy loss for classification
    ce_criterion = nn.CrossEntropyLoss()
    #cross modal center loss
    cmc_criterion = CrossModalCenterLoss(num_classes=args.num_classes, feat_dim=512, use_gpu=True)
    #mse loss
    mse_criterion = nn.MSELoss()
 
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer_centloss = optim.SGD(cmc_criterion.parameters(), lr=args.lr_center)

    train_set = TripletDataloader(dataset = args.dataset, num_points = args.num_points, num_classes=args.num_classes, dataset_dir=args.dataset_dir,  partition='train')
    data_loader_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,num_workers=8)

    iteration = 0
    start_time = time.time()
    for epoch in range(args.epochs):
        for data in data_loader_loader:
            pt, img, img_v, centers, corners, normals, neighbor_index, target, target_vec = data

            img = Variable(img).to('cuda')
            img_v = Variable(img_v).to('cuda')
            pt = Variable(pt).to('cuda')
            pt = pt.permute(0,2,1)
            target = target[:,0]
            target = Variable(target).to('cuda')
            target_vec = Variable(target_vec).to('cuda')
            centers = Variable(torch.cuda.FloatTensor(centers.cuda()))
            corners = Variable(torch.cuda.FloatTensor(corners.cuda()))
            normals = Variable(torch.cuda.FloatTensor(normals.cuda()))
            neighbor_index = Variable(torch.cuda.LongTensor(neighbor_index.cuda()))

            optimizer.zero_grad()
            optimizer_centloss.zero_grad()

            img_pred, pt_pred, mesh_pred, img_feat, pt_feat, mesh_feat = model(pt, img, img_v, centers, corners, normals, neighbor_index)

            #cross-entropy loss for all the three modalities
            pt_ce_loss = ce_criterion(pt_pred, target)
            img_ce_loss = ce_criterion(img_pred, target)
            mesh_ce_loss = ce_criterion(mesh_pred, target)            
            ce_loss = pt_ce_loss + img_ce_loss + mesh_ce_loss

            #cross-modal center loss 
            cmc_loss = cmc_criterion(torch.cat((img_feat, pt_feat, mesh_feat), dim = 0), torch.cat((target, target, target), dim = 0))
      
            # MSE Loss   
            img_pt_mse_loss = mse_criterion(img_feat, pt_feat)
            img_mesh_mse_loss = mse_criterion(img_feat, mesh_feat)
            mesh_pt_mse_loss = mse_criterion(mesh_feat, pt_feat)
            mse_loss = img_pt_mse_loss + img_mesh_mse_loss + mesh_pt_mse_loss
	
	    #weighted the three losses as final loss 
            loss = ce_loss + args.weight_center * cmc_loss +  0.1 * mse_loss
            loss.backward()

            optimizer.step()

            for param in cmc_criterion.parameters():
                param.grad.data *= (1. / args.weight_center)

            #update the parameters for the cmc_loss
            optimizer_centloss.step()

            img_acc = calculate_accuracy(img_pred, target)
            pt_acc = calculate_accuracy(pt_pred, target)
            mesh_acc = calculate_accuracy(mesh_pred, target)

            #tensorboard visualization
            writer.add_scalar('Loss/pt_ce_loss', pt_ce_loss.item(), iteration)
            writer.add_scalar('Loss/img_ce_loss', img_ce_loss.item(), iteration)
            writer.add_scalar('Loss/mesh_ce_loss', mesh_ce_loss.item(), iteration)
            writer.add_scalar('Loss/ce_loss', ce_loss.item(), iteration)
            
            writer.add_scalar('Loss/img_pt_mse_loss', img_pt_mse_loss.item(), iteration)
            writer.add_scalar('Loss/img_mesh_mse_loss', img_mesh_mse_loss.item(), iteration)
            writer.add_scalar('Loss/mesh_pt_mse_loss', mesh_pt_mse_loss.item(), iteration)
            writer.add_scalar('Loss/mse_loss', mse_loss.item(), iteration)
            
            writer.add_scalar('Loss/center_loss', cmc_loss.item(), iteration)
            
            writer.add_scalar('Loss/loss', loss.item(), iteration)
            
            writer.add_scalar('Acc/img_acc',img_acc, iteration)
            writer.add_scalar('Acc/pt_acc',pt_acc, iteration)
            writer.add_scalar('Acc/mesh_acc',mesh_acc, iteration)


            if (iteration%args.lr_step) == 0:
                lr = args.lr * (0.1 ** (iteration // args.lr_step))
                print('New  Learning Rate:     ' + str(lr))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            # update the learning rate of the center loss
            if (iteration%args.lr_step) == 0:
                lr_center = args.lr_center * (0.1 ** (iteration // args.lr_step))
                print('New  Center LR:     ' + str(lr_center))
                for param_group in optimizer_centloss.param_groups:
                    param_group['lr'] = lr_center

            writer.add_scalar('LR/lr', args.lr * (0.1 ** (iteration // args.lr_step)), iteration)
            writer.add_scalar('LR/lr_center', args.lr_center * (0.1 ** (iteration // args.lr_step)), iteration)

            if iteration % args.per_print == 0:
                print('[%d][%d]  loss: %f  img_acc:  %f pt_acc %f mesh_acc %f time: %f  vid: %d' % (epoch, iteration, loss.item(), img_acc, pt_acc, mesh_acc, time.time() - start_time, target.size(0)))
                start_time = time.time()

            iteration = iteration + 1
            if((iteration+1) % args.per_save) ==0:
                print('----------------- Save The Network ------------------------')
                with open(args.save + str(iteration+1)+'-head_net.pkl', 'wb') as f:
                    torch.save(model, f)
                with open(args.save + str(iteration+1)+'-img_net.pkl', 'wb') as f:
                    torch.save(img_net, f)
                with open(args.save + str(iteration+1)+'-pt_net.pkl', 'wb') as f:
                    torch.save(pt_net, f)
                with open(args.save + str(iteration+1)+'-mesh_net.pkl', 'wb') as f:
                    torch.save(mesh_net, f)
            iteration = iteration + 1

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Cross Modal Retrieval for Point Cloud, Mesh, and Image Models')

    parser.add_argument('--dataset', type=str, default='ModelNet40', metavar='dataset',
                        help='ModelNet10 or ModelNet40')

    parser.add_argument('--dataset_dir', type=str, default='./dataset/', metavar='dataset_dir',
                        help='dataset_dir')

    parser.add_argument('--num_classes', type=int, default=40, metavar='num_classes',
                        help='10 or 40')

    parser.add_argument('--batch_size', type=int, default=96, metavar='batch_size',
                        help='Size of batch)')

    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of episode to train ')
    #optimizer
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')

    parser.add_argument('--lr_step', type=int,  default=20000,
                        help='how many iterations to decrease the learning rate')

    parser.add_argument('--lr_center', type=float, default=0.001, metavar='LR',
                        help='learning rate for center loss (default: 0.5)')
                                         
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')

    #DGCNN
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')

    #loss
    parser.add_argument('--weight_center', type=float, default=10, metavar='weight_center',
                        help='weight center (default: 1.0)')

    parser.add_argument('--weight_decay', type=float, default=1e-3, metavar='weight_decay',
                        help='learning rate (default: 1e-3)')

    parser.add_argument('--per_save', type=int,  default=5000,
                        help='how many iterations to save the model')

    parser.add_argument('--per_print', type=int,  default=100,
                        help='how many iterations to print the loss and accuracy')
                        
    parser.add_argument('--save', type=str,  default='./checkpoints/ModelNet40',
                        help='path to save the final model')

    parser.add_argument('--gpu_id', type=str,  default='0,1,2,3',
                        help='GPU used to train the network')

    parser.add_argument('--log', type=str,  default='log/',
                        help='path to the log information')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    torch.backends.cudnn.enabled = False
    training(args)
