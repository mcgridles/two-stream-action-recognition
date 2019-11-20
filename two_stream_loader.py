# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 18:09:40 2019

@author: lpott
"""

from network import *
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

def load_models(spatial_path,temporal_path):
    """ spatial_path : path to the best model parameters for spatial
        temporal_path : path to the best model parameters for temporal
    """
    state_dict_temporal = torch.load(r"/home/mlp/two-stream-action-recognition/record/motion/model_best.pth.tar") 
    state_dict_spatial = torch.load(r"/home/mlp/two-stream-action-recognition/record/spatial/model_best.pth.tar")

    temporal_net = resnet101(pretrained=True,channel=20,nb_classes=4)
    spatial_net = resnet101(pretrained=True,channel=3,nb_classes=4)
    
    temporal_net.load_state_dict(state_dict_temporal['state_dict'])
    spatial_net.load_state_dict(state_dict_spatial['state_dict'])
    
    return temporal_net.cuda(),spatial_net.cuda()
    
def load_temporal_images(path,videoname):
    """ path : folder path to tvl1 images (not including the u/v)
        videoname : the video filename
        
        returns : imgs_v : list full of v optical flow
                  imgs_u : list full of u optical flow
    """
    u_image_path = os.path.join(path,'u',videoname)
    v_image_path = os.path.join(path,'v',videoname)
    
    tsfm = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                ])
    
    files_u = sorted(os.listdir(u_image_path))    
    files_v = sorted(os.listdir(v_image_path))
 
    imgs_u = []
    imgs_v = []
    for file in files_u:
        imgs_u.append(tsfm(Image.open(os.path.join(path,'u',videoname,file))))
        
        
    imgs_v = []
    for file in files_v:
        imgs_v.append(tsfm(Image.open(os.path.join(path,'v',videoname,file))))
    
    return imgs_u,imgs_v
    
    
def load_spatial_images(path,videoname):
    """ path : folder path to RGB images
        videoname : the video filename
        
        returns : imgs : list full of spatial images
    """
    spatial_image_path = os.path.join(path,videoname)
    
    
    tsfm = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ])
    
    files = sorted(os.listdir(spatial_image_path))

    imgs = []
    for file in files:
        imgs.append(tsfm(Image.open(os.path.join(path,videoname,file))))
    
    return imgs

def load_normal_spatial_images(path,videoname):
    """ path : folder path to RGB images
        videoname : the video filename
        
        returns : imgs : list full of spatial images
    """
    spatial_image_path = os.path.join(path,videoname)
    
    files = sorted(os.listdir(spatial_image_path))
    
    imgs = []
    for file in files:
        imgs.append(cv2.resize(plt.imread(os.path.join(path,videoname,file)),(224,224)))
    
    return imgs

def load_normal_temporal_images(path,videoname):
    """ path : folder path to tvl1 images (not including the u/v)
        videoname : the video filename
        
        returns : imgs_v : list full of v optical flow
                  imgs_u : list full of u optical flow
    """
    u_image_path = os.path.join(path,'u',videoname)
    v_image_path = os.path.join(path,'v',videoname)
    
    tsfm = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                ])
    
    files_u = sorted(os.listdir(u_image_path))    
    files_v = sorted(os.listdir(v_image_path))
 
    imgs_u = []
    imgs_v = []
    for file in files_u:
        imgs_u.append(cv2.resize(plt.imread(os.path.join(path,'u',videoname,file)),(224,224)))
        
    imgs_v = []
    for file in files_v:
        imgs_v.append(cv2.resize(plt.imread(os.path.join(path,'v',videoname,file)),(224,224)))
    
    return imgs_u,imgs_v



if __name__ == '__main__':
    print("I MADE IT YAY")
    