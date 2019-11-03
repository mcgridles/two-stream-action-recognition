import pickle
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from split_train_test_video import *
from skimage import io, color, exposure

import re

class spatial_dataset(Dataset):  
    def __init__(self, dic, root_dir, mode, transform=None):
        
        print("Spatial {:s} Dataset Initialized".format(mode))
 
        # keys contains list of "videoname [frames in video]" for train split
        self.keys = list(dic.keys())
        
        # keys in value contains list of annotations for each video for train split
        self.values= list(dic.values())
        
        self.root_dir = root_dir
        self.mode =mode
        self.transform = transform

    def __len__(self):
        return len(self.keys)

    def load_ucf_image(self,video_name, index):
        if re.search("HandStandPushups",video_name):
            video_name = re.sub("Stand","stand",video_name)
            path = self.root_dir +  "v_" + video_name + "/"
        else:
            path = self.root_dir + "v_" + video_name + "/"

        frame_num = "0" * (6-len(str(index)))
        img = Image.open(path + "frame" + frame_num + str(index)+'.jpg')
        
        # MLP
        if self.transform is not None:
            transformed_img = self.transform(img)
           
        else:
            transformed_img=img
        
        img.close()

        return transformed_img

    def __getitem__(self, idx):

        if self.mode == 'train':
            video_name, nb_clips = self.keys[idx].split(' ')
            nb_clips = int(nb_clips)
            clips = []
            clips.append(np.random.randint(1, nb_clips/3))
            clips.append(np.random.randint(nb_clips/3, nb_clips*2/3))
            clips.append(np.random.randint(nb_clips*2/3, nb_clips+1))
            
        elif self.mode == 'val':
            video_name, index = self.keys[idx].split(' ')
            index =abs(int(index))
       
        else:
            raise ValueError('There are only train and val mode')

        label = self.values[idx]
        label = int(label)-1
        
        if self.mode=='train':
            data ={}
            for i in range(len(clips)):
                key = 'img'+str(i)
                index = clips[i]
                data[key] = self.load_ucf_image(video_name, index)
                    
            sample = (data, label)
        elif self.mode=='val':
            data = self.load_ucf_image(video_name,index)
            sample = (video_name, data, label)
        else:
            raise ValueError('There are only train and val mode')
           
        return sample

class spatial_dataloader():
    def __init__(self, BATCH_SIZE, num_workers, path, ucf_list, ucf_split):
        print("Spatial Dataloader Initialized")

        self.BATCH_SIZE=BATCH_SIZE
        self.num_workers=num_workers
        self.data_path=path
        self.frame_count ={}
        
        # split the training and testing videos
        splitter = UCF101_splitter(path=ucf_list,split=ucf_split)
        
        # self.train_video is dictionary of videoname : label
        # i.e. "HandStandPushups_g21_c06: 37,
        # self.test video is similar... 
        self.train_video, self.test_video = splitter.split_video()

    def load_frame_count(self):
        #print '==> Loading frame number of each video'
        path = r'/home/mlp/two-stream-action-recognition/dataloader/dic/frame_count.pickle'
        
        # dictionary of video : number of frames
        # i.e "ApplyLipstick_g02_c02.avi" : 112
        with open(path,'rb') as file:
            dic_frame = pickle.load(file)
        file.close()

        for line in dic_frame:
            # extract just the video name i.e. "ApplyEyeMakeup_g10_c06
            videoname = re.search("_.*\.",line)
            if videoname is not None:
              videoname = videoname[0][1:-1]
            else:
              videoname = line

            if re.search("HandstandPushups",videoname):
                videoname = re.sub("stand","Stand",videoname)
            
            # dictionary of video : number of frames
            # i.e "ApplyLipstick_g02_c02" : 112
            self.frame_count[videoname]=dic_frame[line]
        

    def run(self):
        self.load_frame_count()
        self.get_training_dic()
        self.val_sample20()
        train_loader = self.train()
        val_loader = self.validate()

        return train_loader, val_loader, self.test_video

    def get_training_dic(self):
        #print '==> Generate frame numbers of each training video'
        self.dic_training={}
        # generate dictionary with keys contains list of 
        # "videoname [frames in video]" for train split
        # and values with annotation
        # i.e. 'ApplyEyeMakeup_g08_c03 136': 1

        for video in self.train_video:
            nb_frame = self.frame_count[video]-10+1
            key = video+' '+ str(nb_frame)
            self.dic_training[key] = self.train_video[video]
                    
    def val_sample20(self):
        print('==> sampling testing frames')
        self.dic_testing={}
        # generates dictionary with keys containing list of 
        # "videoname [frame to be sampled] : annotation
        # spaces out the [frame to be sampled] in videos to get 19 frames per video
        # i.e. 'ApplyEyeMakeup_g01_c01 1': 1
        for video in self.test_video:
            nb_frame = self.frame_count[video]-10+1
            interval = int(nb_frame/19)
            for i in range(19):
                frame = i*interval
                key = video+' '+str(frame+1)
                self.dic_testing[key] = self.test_video[video]      

    def train(self):
        training_set = spatial_dataset(dic=self.dic_training, root_dir=self.data_path, mode='train', transform = transforms.Compose([
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ]))
        print('==> Training data :',len(training_set),'frames')
        #print(training_set[1][0]['img1'].size())

        train_loader = DataLoader(
            dataset=training_set, 
            batch_size=self.BATCH_SIZE,
            shuffle=True,
            num_workers=self.num_workers)
        return train_loader

    def validate(self):
        validation_set = spatial_dataset(dic=self.dic_testing, root_dir=self.data_path, mode='val', transform = transforms.Compose([
                transforms.Resize([224,224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ]))
        
        print('==> Validation data :',len(validation_set),'frames')
        #print(validation_set[1][1].size())

        val_loader = DataLoader(
            dataset=validation_set, 
            batch_size=self.BATCH_SIZE, 
            shuffle=False,
            num_workers=self.num_workers)
        
        return val_loader





if __name__ == '__main__':
    dataloader = spatial_dataloader(BATCH_SIZE=1, num_workers=1, 
                                path=r"/mnt/disks/datastorage/videos/rgb/", 
                                ucf_list=r"/home/mlp/two-stream-action-recognition/UCF_list/",
                                ucf_split='01')
    train_loader,val_loader,test_video = dataloader.run()