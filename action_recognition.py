import os
import time
import cv2
import torch
import torchvision.transforms as transforms

from utils import *
from network import *


class SpatialCNN:
	def __init__(self, weights_path, img_size=(224,224)):
		self.model = None
        self.weights = weights_path
        self.img_size = img_size

        self.transform = transforms.Compose([
        	transforms.RandomCrop(224),
    		transforms.ToTensor(),
    		transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    		])

        self.load()

    def load(self):
    	print('==> Build spatial model')

        # Build model
        self.model = resnet101(pretrained=True, channel=3).cuda()

        # Load weights
        if os.path.isfile(self.weights):
        	print("==> loading weights '{}'".format(self.weights))
	        checkpoint = torch.load(self.weights)
	        self.model.load_state_dict(checkpoint['state_dict'])

	        epoch = checkpoint['epoch']
	        best_prec = checkpoint['best_prec1']
	        print("==> loaded checkpoint '{}' (epoch {}) (best_prec {})".format(self.weights, epoch, best_prec))
	    else:
	    	print("==> no checkpoint found at '{}'".format(self.weights))
	    	exit(1)

	    self.model.eval()

    def run(self, img):
    	prev_time = time.time()

    	img = cv2.resize(self.transform(img), self.img_size)
    	with torch.no_grad():
    		output = self.model(img)
    		preds = output.data.cpu().numpy()

    		elapsed_time = time.time() - prev_time
            prev_time = time.time()

        return preds


class MotionCNN:
	def __init__(self, weights_path):
        self.model = None
        self.weights = weights_path

        self.load()

    def load(self):
    	print('==> Build temporal model')

        # Build model
        self.model = resnet101(pretrained=True, channel=3).cuda()

        if os.path.isfile(self.weights):
        	print("==> loading weights '{}'".format(self.weights))
        	checkpoint = torch.load(self.weights)
        	self.model.load_state_dict(checkpoint['state_dict'])

            epoch = checkpoint['epoch']
            best_prec = checkpoint['best_prec1']
            print("==> loaded checkpoint '{}' (epoch {}) (best_prec1 {})".format(self.weights, epoch, best_prec))
        else:
        	print("==> no checkpoint found at '{}'".format(self.weights))
        	exit(1)

        self.model.eval()

    def run(self, flow):
    	prev_time = time.time()

    	flow = cv2.resize(flow, self.img_size)
    	with torch.no_grad():
    		output = self.model(img)
    		preds = output.data.cpu().numpy()

    		elapsed_time = time.time() - prev_time
            prev_time = time.time()

        return preds
