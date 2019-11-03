# -*- coding: utf-8 -*-
import glob
import numpy as np
import os
import re

import argparse

parser = argparse.ArgumentParser(description='Training .txt file generator')
parser.add_argument('--train', default=1, type=int, metavar='N', help='1 for train.txt file, 0 for test.txt file')
parser.add_argument('--classindexpath', default=r'/home/mlp/two-stream-action-recognition/UCF_list/classInd.txt', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--videofolderpath', default=r'/mnt/disks/datastorage/videos/rgb/', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--groupblock', default='[2][0]', type=str, help='Enter regex expression for numbers of groups i.e. [0-1][0-5] for groups 0-15')

def main():
    global arg
    arg = parser.parse_args()
    #print("Starting to parse...")
    #print(arg)
    
    video_path = arg.videofolderpath
    class_ind_path = arg.classindexpath
    
    group_block = arg.groupblock
    train = arg.train
    
    class_labels = {}
    with open(class_ind_path) as f:
        for line in f:
           (val, key) = line.split()
           class_labels[key] = int(val)
           
    search = os.path.join(video_path,"*_g" + group_block + "_c[0-9]*")
    folders = glob.glob(search)
    for file in folders:
        action = re.search("v_.*_g",file)[0][2:-2]
        action_folder = re.search("v_.*",file)[0]
        if train:
            action_label = class_labels[action]
            print(os.path.join(action,action_folder)+" "+str(class_labels[action]))
        else:
            print(os.path.join(action,action_folder))
            
if __name__=='__main__':
    main()