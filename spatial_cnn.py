import numpy as np
import pickle
import os
from PIL import Image
import time
from tqdm import tqdm
import shutil
from random import randint
import argparse

import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

import dataloader
from action_utils import *
from network import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='UCF101 spatial stream on resnet101')
parser.add_argument('--epochs', default=500, type=int, metavar='N', help='number of total epochs')
parser.add_argument('--batch-size', default=25, type=int, metavar='N', help='mini-batch size (default: 25)')
parser.add_argument('--lr', default=5e-4, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')

parser.add_argument('--nb-classes', default=101, type=int, metavar='N', help='Number of target classes to train')
parser.add_argument('--finetune', default=0, type=int, metavar='N', help='1 to fine-tune, 0 to to train normally')

def main():
    global arg
    arg = parser.parse_args()
    print(arg)

    #Prepare DataLoader
    data_loader = dataloader.spatial_dataloader(
                        BATCH_SIZE=arg.batch_size,
                        num_workers=8,
                        path=r"/mnt/disks/datastorage/jpegs_256/",
                        ucf_list =r"/home/mlp/two-stream-action-recognition/UCF_list/",
                        ucf_split ='01', 
                        )
    
    train_loader, test_loader, test_video = data_loader.run()
    #Model 
    model = Spatial_CNN(
                        nb_epochs=arg.epochs,
                        lr=arg.lr,
                        batch_size=arg.batch_size,
                        resume=arg.resume,
                        start_epoch=arg.start_epoch,
                        evaluate=arg.evaluate,
                        train_loader=train_loader,
                        test_loader=test_loader,
                        test_video=test_video,
                        nb_classes=arg.nb_classes, # added the nb_classes
                        finetune=arg.finetune
    )
    #Training
    model.run()

class Spatial_CNN():
    # add the nb_classes
    def __init__(self, nb_epochs, lr, batch_size, resume, start_epoch, evaluate, train_loader, test_loader, test_video,nb_classes,finetune):
        self.nb_epochs=nb_epochs
        self.lr=lr
        self.batch_size=batch_size
        self.resume=resume
        self.start_epoch=start_epoch
        self.evaluate=evaluate
        self.train_loader=train_loader
        self.test_loader=test_loader
        self.best_prec1=0
        self.test_video=test_video
        
        self.nb_classes = nb_classes
        self.finetune = finetune

    def build_model(self):
        print('==> Build model and setup loss and optimizer')
        #build model
        # added in the nb_classes argument
        self.model = resnet101(pretrained= True, channel=3,nb_classes=self.nb_classes,p=0).cuda()
        #Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.optimizer = torch.optim.SGD(self.model.parameters(), self.lr, momentum=0.9)
        #self.optimizer = torch.optim.Adam(self.model.parameters(),self.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=3,verbose=True)
    
    def resume_and_evaluate(self):
        if self.resume:
            if os.path.isfile(self.resume):
                print("==> loading checkpoint '{}'".format(self.resume))
                checkpoint = torch.load(self.resume)
                ## to finetune, or to not fine, tis the question
                if not self.finetune:
                  print("In Resume Training Mode\n")
                  self.start_epoch = checkpoint['epoch']
                  self.best_prec1 = checkpoint['best_prec1']
                  self.model.load_state_dict(checkpoint['state_dict'])
                  self.optimizer.load_state_dict(checkpoint['optimizer'])
                  print("==> loaded checkpoint '{}' (epoch {}) (best_prec1 {})".format(self.resume, checkpoint['epoch'], self.best_prec1))
                # Add in new state dict
                else:
                  print("--In Finetune mode--")
                  pretrained_dict = checkpoint['state_dict']
                  new_model_dict = self.model.state_dict()
                  
                  print("Delete last layer weights")
                  
                  del pretrained_dict["fc_custom.weight"]
                  del pretrained_dict["fc_custom.bias"]
                  
                  print("Update last layer weights with ImageNet pretrained weights")
                  pretrained_dict["fc_custom.weight"] = new_model_dict["fc_custom.weight"].clone()
                  pretrained_dict["fc_custom.bias"] = new_model_dict["fc_custom.bias"].clone()          
                  print("Sanity Check On Number of Classes : ",pretrained_dict["fc_custom.weight"].size()[0])
                  self.model.load_state_dict(pretrained_dict)
                   
                  
            else:
                print("==> no checkpoint found at '{}'".format(self.resume))
        if self.evaluate:
            self.epoch = 0
            prec1, val_loss = self.validate_1epoch()
            return
 
    def run(self):
        self.build_model()
        self.resume_and_evaluate()
        cudnn.benchmark = True
        
        for self.epoch in range(self.start_epoch, self.nb_epochs):
            # MLP
            self.train_1epoch()
            prec1, val_loss = self.validate_1epoch()
            is_best = prec1 > self.best_prec1
            #lr_scheduler
            self.scheduler.step(val_loss)
            # save model
            if is_best:
                self.best_prec1 = prec1
                with open('record/spatial/spatial_video_preds.pickle','wb') as f:
                    pickle.dump(self.dic_video_level_preds,f)
                f.close()
            
            save_checkpoint({
                'epoch': self.epoch,
                'state_dict': self.model.state_dict(),
                'best_prec1': self.best_prec1,
                'optimizer' : self.optimizer.state_dict()
            },is_best,'record/spatial/checkpoint.pth.tar','record/spatial/model_best.pth.tar')

    def train_1epoch(self):
        print('==> Epoch:[{0}/{1}][training stage]'.format(self.epoch, self.nb_epochs))
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        #switch to train mode
        self.model.train()    
        end = time.time()
        # mini-batch training
        progress = tqdm(self.train_loader)
        for i, (data_dict,label) in enumerate(progress):

    
            # measure data loading time
            data_time.update(time.time() - end)
            
            label = label.cuda()
            target_var = Variable(label).cuda()

            # compute output
            output = Variable(torch.zeros(len(data_dict['img1']),101).float()).cuda()

            #MLP
            dic_len = len(data_dict)

            for i in range(dic_len):
                key = 'img'+str(i)
                data = data_dict[key]
                input_var = Variable(data).cuda()
                output += self.model(input_var)

            loss = self.criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, label, topk=(1, 5))
            
            # MLP
            losses.update(loss.data.item(), data.size(0))
            top1.update(prec1.item(), data.size(0))
            top5.update(prec5.item(), data.size(0))

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
        
        info = {'Epoch':[self.epoch],
                'Batch Time':[round(batch_time.avg,3)],
                'Data Time':[round(data_time.avg,3)],
                'Loss':[round(losses.avg,5)],
                'Prec@1':[round(top1.avg,4)],
                'Prec@5':[round(top5.avg,4)],
                'lr': self.optimizer.param_groups[0]['lr']
                }
        record_info(info, 'record/spatial/rgb_train.csv','train')

    def validate_1epoch(self):
        print('==> Epoch:[{0}/{1}][validation stage]'.format(self.epoch, self.nb_epochs))
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        # switch to evaluate mode
        self.model.eval()
        self.dic_video_level_preds={}
        end = time.time()
        progress = tqdm(self.test_loader)
        with torch.no_grad():
            for i, (keys,data,_) in enumerate(progress):

                #label = label#.cuda(async=True)
                
                # MLP
                # warning error that volatile=True no longer applicable
                #data_var = Variable(data, volatile=True)#.cuda(async=True)
                #label_var = Variable(label, volatile=True)#.cuda(async=True)
                
                data_var = data.cuda()
                
                # compute output
                output = self.model(data_var)
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                #Calculate video level prediction
                preds = output.data.cpu().numpy()
                nb_data = preds.shape[0]
                for j in range(nb_data):
                    #print(keys[j])
                    # May be redundant . . .
                    videoName = keys[j].split('/',1)[0]
                    # MLP
                    #print(videoName)
                    if videoName not in self.dic_video_level_preds.keys():
                        self.dic_video_level_preds[videoName] = preds[j,:]
                    else:
                        self.dic_video_level_preds[videoName] += preds[j,:]

        video_top1, video_top5, video_loss = self.frame2_video_level_accuracy()
            

        info = {'Epoch':[self.epoch],
                'Batch Time':[round(batch_time.avg,3)],
                # MLP change video_loss.round(5)
                'Loss':[video_loss.round(5)],
                'Prec@1':[round(video_top1,3)],
                'Prec@5':[round(video_top5,3)]}
        record_info(info, 'record/spatial/rgb_test.csv','test')
        return video_top1, video_loss

    def frame2_video_level_accuracy(self):
            
        correct = 0
        video_level_preds = np.zeros((len(self.dic_video_level_preds),101))
        video_level_labels = np.zeros(len(self.dic_video_level_preds))
        ii=0
        for name in sorted(self.dic_video_level_preds.keys()):
        
            preds = self.dic_video_level_preds[name]
            label = int(self.test_video[name])-1
                
            video_level_preds[ii,:] = preds
            video_level_labels[ii] = label
            ii+=1         
            if np.argmax(preds) == (label):
                correct+=1

        #top1 top5
        video_level_labels = torch.from_numpy(video_level_labels).long()
        video_level_preds = torch.from_numpy(video_level_preds).float()
            
        top1,top5 = accuracy(video_level_preds, video_level_labels, topk=(1,5))
        loss = self.criterion(Variable(video_level_preds).cuda(), Variable(video_level_labels).cuda())     
        #loss = self.criterion(Variable(video_level_preds), Variable(video_level_labels))     

                            
        top1 = float(top1.numpy())
        top5 = float(top5.numpy())
            
        print(' * Video level Prec@1 {top1:.3f}, Video level Prec@5 {top5:.3f}'.format(top1=top1, top5=top5))
        return top1,top5,loss.data.cpu().numpy()







if __name__=='__main__':
    main()