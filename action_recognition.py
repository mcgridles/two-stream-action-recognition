import os
import torch
import torchvision.transforms as transforms

from action_utils import *
from network import *


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

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(224,224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ])

        self.load()

    def load(self):
        with TimerBlock('Building spatial model') as block:
            # Build model
            if self.args.cuda and self.args.number_gpus > 0:
                self.model = resnet101(pretrained=True, channel=3).cuda()
            else:
                self.model = resnet101(pretrained=True, channel=3)

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
            if self.args.cuda and self.args.number_gpus > 0:
                img = img.cuda()

            output = self.model(img)
            preds = output.data.cpu().numpy()

        return preds


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

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(224,224),
            transforms.ToTensor(),
            ])

        self.load()

    def load(self):
        with TimerBlock('Building temporal model') as block:
            # Build model
            if self.args.cuda and self.args.number_gpus > 0:
                self.model = resnet101(pretrained=True, channel=20).cuda()
            else:
                self.model = resnet101(pretrained=True, channel=20)

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
        with TimerBlock('Starting temporal network') as block:
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
        flow = flow.type(torch.FloatTensor)

        with torch.no_grad():
            if self.args.cuda and self.args.number_gpus > 0:
                flow = flow.cuda()

            output = self.model(flow)
            preds = output.data.cpu().numpy()

        return preds
