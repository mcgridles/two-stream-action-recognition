# -*- coding: utf-8 -*-
import glob
import numpy as np
import os
import re
import pickle

import argparse

parser = argparse.ArgumentParser(description='Training .txt file generator')
parser.add_argument('--videofolderpath', default=r'/mnt/disks/datastorage/videos/rgb', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')

def main():
    global arg
    arg = parser.parse_args()
    #print("Starting to parse...")
    #print(arg)
    
    video_path = arg.videofolderpath
    
    video_files = os.listdir(video_path)           
    
    frame_count = {}
    for file in video_files:

        full_path = os.path.join(video_path,file)
        frame_count[file[2:]] = len(os.listdir(full_path))
    
    print(frame_count)
    pickle_file = open('frame_count.pickle','wb')
    pickle.dump(frame_count,pickle_file)
    pickle_file.close()
if __name__=='__main__':
    main()