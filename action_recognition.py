import os
import time
import cv2
import torch
import torchvision.transforms as transforms

from utils import *
from network import *


class SpatialCNN:
    """
    Spatial network for two stream action recognition.
    """

    def __init__(self, weights_path, img_size=(224,224)):
        self.model = None
        self.weights = weights_path
        self.img_size = list(img_size) + [3]

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ])

        self.load()

    def load(self):
        with TimerBlock('Building spatial model') as block:
            # Build model
            self.model = resnet101(pretrained=True, channel=3).cuda()

            # Load weights
            if os.path.isfile(self.weights):
                block.log('Loading weights {}'.format(self.weights))
                checkpoint = torch.load(self.weights)
                self.model.load_state_dict(checkpoint['state_dict'])

                epoch = checkpoint['epoch']
                best_prec = checkpoint['best_prec1']
                block.log("Loaded checkpoint '{}' (epoch {}) (best_prec {})".format(self.weights, epoch, best_prec))
            else:
                block.log("No checkpoint found at '{}'".format(self.weights))
                exit(1)

            self.model.eval()

    def run_async(self, img_queue, pred_queue):
        with TimerBlock('Starting spatial network') as block:
            while True:
                img = img_queue.get(block=True)
                if type(img) != np.ndarray:
                    # Break out of loop when signal received
                    block.log('Spatial network exiting')
                    break

                preds = self.run(img)
                pred_queue.put(preds)

    def run(self, img):
        img = np.resize(img, self.img_size)
        img = self.transform(img).unsqueeze(0)

        with torch.no_grad():
            output = self.model(img.cuda())
            preds = output.data.cpu().numpy()

        return preds


class MotionCNN:
    """
    Temporal network for two stream action recognition.
    """
    
    def __init__(self, weights_path, img_size=(224,224)):
        self.model = None
        self.weights = weights_path
        self.img_size = [20] + list(img_size)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            ])

        self.load()

    def load(self):
        with TimerBlock('Building temporal model') as block:
            # Build model
            self.model = resnet101(pretrained=True, channel=20).cuda()

            if os.path.isfile(self.weights):
                block.log("Loading weights '{}'".format(self.weights))
                checkpoint = torch.load(self.weights)
                self.model.load_state_dict(checkpoint['state_dict'])

                epoch = checkpoint['epoch']
                best_prec = checkpoint['best_prec1']
                block.log("Loaded checkpoint '{}' (epoch {}) (best_prec1 {})".format(self.weights, epoch, best_prec))
            else:
                block.log("No checkpoint found at '{}'".format(self.weights))
                exit(1)

            self.model.eval()

    def run_async(self, flow_queue, pred_queue):
        with TimerBlock('Performing temporal inference') as block:
            while True:
                flow = flow_queue.get(block=True)
                if type(flow) != np.ndarray:
                    # Break out of loop when signal received
                    block.log('Temporal network exiting')
                    break

                preds = self.run(flow)
                pred_queue.put(preds)

    def run(self, flow):
        flow = np.resize(flow, self.img_size)
        for i in range(flow.shape[0]):
            flow[i,:,:] = self.transform(np.uint8(flow[i,:,:]))
        flow = torch.from_numpy(flow).unsqueeze(0)

        with torch.no_grad():
            output = self.model(flow.type(torch.FloatTensor).cuda())
            preds = output.data.cpu().numpy()

        return preds
