import matplotlib.pyplot as plt
import pickle
import numpy as np
import torch
import dataloader
import os
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

from action_utils import *

if __name__ == '__main__':

    rgb_preds='record/spatial/spatial_video_preds.pickle'
    with open(rgb_preds,'rb') as f:
        rgb =pickle.load(f)
    f.close()

    dataloader = dataloader.spatial_dataloader(BATCH_SIZE=1, num_workers=1, 
                                    path='/mnt/disks/datastorage/videos/rgb/', 
                                    ucf_list=os.path.join(os.path.abspath(__file__), 'UCF_list'),
                                    ucf_split='01')
    train_loader,val_loader,test_video = dataloader.run()

    video_level_preds = np.zeros(len(rgb.keys()))
    video_level_labels = np.zeros(len(rgb.keys()))
    for ii, name in enumerate(sorted(rgb.keys())):
        r = rgb[name]

        label = int(test_video[name])-1
                    
        video_level_preds[ii] = np.argmax(r)
        video_level_labels[ii] = label

    video_level_labels = torch.from_numpy(video_level_labels).long()
    video_level_preds = torch.from_numpy(video_level_preds).float()

    cm = confusion_matrix(video_level_labels, video_level_preds)

    classes = set(video_level_labels + video_level_preds)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    plt.show()
