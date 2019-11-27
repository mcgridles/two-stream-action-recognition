import os
import time
import torch
import torchvision.transforms as transforms

from action_utils import *
from network import *


class AddTransform(object):
    
    def __call__(self, x):
        return x / 255 + 128./255
    

class SpatialCNN:
    """
    Spatial network for two stream action recognition.

    Arguments:
        --spatial_weights (str) -> Path to spatial weights checkpoint
        --image_size (tuple(int)) -> Desired image size, default=(224,224)
        --cuda (bool) -> Attempt to use CUDA if true
        --number_gpus (int) -> Number of GPUs to use, default=-1
    """

    def __init__(self, args):
        self.model = None
        self.weights = args.spatial_weights
        self.img_size = list(args.image_size[:2]) + [3]
        self.args = args
        self.rgb = torch.FloatTensor(11, 3, 224, 224)
        self.frame_idx = 0

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ])

        self.load()

    def load(self):
        with TimerBlock('Building spatial model') as block:
            # Build model
            if self.args.cuda and self.args.number_gpus > 0:
                block.log('Initializing CUDA')
                self.model = resnet101(pretrained=True, channel=3, nb_classes=self.args.nb_classes).cuda()
            else:
                self.model = resnet101(pretrained=True, channel=3, nb_classes=self.args.nb_classes)

            # Load weights
            if os.path.isfile(self.weights):
                block.log('Loading weights {}'.format(self.weights))
                checkpoint = torch.load(self.weights)
                self.model.load_state_dict(checkpoint['state_dict'])

                epoch = checkpoint['epoch']
                best_prec = checkpoint['best_prec1']
                block.log("Loaded checkpoint '{}' (epoch {}) (best_prec {})".format(self.weights, epoch, round(best_prec, 2)))
            else:
                block.log("No checkpoint found at '{}'".format(self.weights))
                exit(1)

            self.model.eval()

    def __call__(self, img):
        self.rgb[-1, :, :, :] = self.transform(img)
        preds = None
        
        if self.frame_idx >= 10:
            with torch.no_grad():
                img = self.rgb[0, :, :, :].unsqueeze(0)
                if self.args.cuda and self.args.number_gpus > 0:
                    img = img.cuda()

                output = self.model(img)
                preds = output.data.cpu().numpy()
                
        self.rgb = self.roll_tensor(-1)
        self.frame_idx += 1
        
        return preds
    
    def roll_tensor(self, n):
        return torch.cat((self.rgb[-n:, :, :], self.rgb[:-n, :, :]))


class MotionCNN:
    """
    Temporal network for two stream action recognition.

    Arguments:
        --motion_weights (str) -> Path to motion weights checkpoint
        --image_size (tuple(int)) -> Desired image size, default=(224,224)
        --cuda (bool) -> Attempt to use CUDA if true
        --number_gpus (int) -> Number of GPUs to use, default=-1
    """
    
    def __init__(self, args):
        self.model = None
        self.weights = args.motion_weights
        self.img_size = [20] + list(args.image_size[:2])
        self.args = args
        self.flow = torch.FloatTensor(2*10, 224, 224)
        self.frame_idx = 0
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            AddTransform()
            ])

        self.load()

    def load(self):
        with TimerBlock('Building temporal model') as block:
            # Build model
            if self.args.cuda and self.args.number_gpus > 0:
                block.log('Initializing CUDA')
                self.model = resnet101(pretrained=True, channel=20, nb_classes=self.args.nb_classes).cuda()
            else:
                self.model = resnet101(pretrained=True, channel=20, nb_classes=self.args.nb_classes)

            if os.path.isfile(self.weights):
                block.log("Loading weights '{}'".format(self.weights))
                checkpoint = torch.load(self.weights)
                self.model.load_state_dict(checkpoint['state_dict'])

                epoch = checkpoint['epoch']
                best_prec = checkpoint['best_prec1']
                block.log("Loaded checkpoint '{}' (epoch {}) (best_prec1 {})".format(self.weights, epoch, round(best_prec, 2)))
            else:
                block.log("No checkpoint found at '{}'".format(self.weights))
                exit(1)

            self.model.eval()

    def __call__(self, of):
        self.flow[-2, :, :] = self.transform(of[0])
        self.flow[-1, :, :] = self.transform(of[1])
        preds = None
        
        if self.frame_idx >= 9:
            with torch.no_grad():
                flow = self.flow.unsqueeze(0)
                if self.args.cuda and self.args.number_gpus > 0:
                    flow = flow.cuda()

                output = self.model(flow)
                preds = output.data.cpu().numpy()
        
        self.flow = self.roll_tensor(-2)
        self.frame_idx += 1

        return preds
    
    def roll_tensor(self, n):
        return torch.cat((self.flow[-n:, :, :], self.flow[:-n, :, :]))
