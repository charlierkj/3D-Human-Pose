import os
import numpy as np
import json

import torch
from torch import nn
import torch.utils.data as td
import torchvision as tv


class MultiView_SynData(td.Dataset):

    def __init__(self, path, num_camera=4):
        self.basepath = path
        self.subj = os.listdir(self.basepath)
        self.framelist = []
        self.cameras = []
        self.camparams = []

        for camera_idx in range(num_camera):
            self.cameras.append("camera_%d" % (camera_idx + 1))

        for (subj_idx, subj_name) in enumerate(self.subj):
            subj_path = os.path.join(self.basepath, subj_name)
            files = os.listdir(subj_path)
            for f in files:
                f_path = os.path.join(subj_path, f)
                if not os.path.isdir(f_path):
                    if f in self.cameras:
                        
                self.framelist.append([subj_idx, int(f)])
            

    def __len__(self):

    def __getitem(self, idx):
        sample = self.framelist[idx]
        subj, frame = sample[0], sample[1]
        frame_path = os.path.join(self.basepath, subj, frame)
        
