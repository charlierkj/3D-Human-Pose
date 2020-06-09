import os
import numpy as np
import json

import torch
from torch import nn
import torch.utils.data as td
import torchvision as tv

from PIL import Image

from camera_utils import *

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class MultiView_SynData(td.Dataset):

    def __init__(self, path, num_camera=4, invalid_joints=None, bbox=None, ori_form=1):
        """
        invalid_joints: tuple of indices for invalid joints; associated joints will not be used in evaluation.
        bbox: [upper_left_x, upper_left_y, lower_right-x, lower_right_y].
        ori_form: original form of how the synthetic data is stored, currently takes value 0 or 1.
        """
        self.basepath = path
        self.form = ori_form
        self.subj = os.listdir(self.basepath)
        self.framelist = []
        self.camera_names = []
        self.cameras = {}
        self.len = 0

        self.invalid_jnts = () if invalid_joints is None else invalid_joints

        # torchvision transforms
        self.transform = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
            ])

        # joint names
        self.joints_name = [
            "foot_r",
            "calf_r",
            "thigh_r",
            "thigh_l",
            "calf_l",
            "foot_l",
            "pelvis",
            "spine_02",
            "neck_01",
            "head",
            "hand_r",
            "lowerarm_r",
            "upperarm_r",
            "upperarm_l",
            "lowerarm_l",
            "hand_l",
            "head"
            ]
        self.num_jnts = len(self.joints_name)

        # original form: 0
        if self.form == 0:
            # load camera names
            for camera_idx in range(num_camera):
                self.camera_names.append("%d" % (camera_idx + 1))

            for (subj_idx, subj_name) in enumerate(self.subj):
                subj_path = os.path.join(self.basepath, subj_name)
                files = os.listdir(subj_path)
                cams = {} # dict to store camera params for each subject
                for f in files:
                    f_path = os.path.join(subj_path, f)
                    if not os.path.isdir(f_path):
                        f_fn = os.path.splitext(f)[0] # filename without extension
                        if ('camera' in f_fn) and (f_fn.split('_')[1] in self.camera_names):
                            # load cameras
                            camera_idx = self.camera_names.index(f_fn.split('_')[1])
                            with open(f_path) as json_file:
                                camera_dict = json.load(json_file)

                            pos = camera_dict['location']
                            rot = camera_dict['rotation']
                            width = camera_dict['width']
                            height = camera_dict['height']
                            fov = camera_dict['fov']

                            cam = Camera(pos[0], pos[1], pos[2],
                                         rot[2], rot[0], rot[1],
                                         width, height, width/2)
                            cams[camera_idx] = cam

                    else:
                        # record frame list
                        self.framelist.append([subj_idx, int(f), bbox])
                        self.len += 1
                self.cameras[subj_idx] = cams

        # original form: 1
        elif self.form == 1:
            for camera_idx in range(num_camera):
                self.camera_names.append("%04d" % camera_idx)

            for (subj_idx, subj_name) in enumerate(self.subj):
                subj_path = os.path.join(self.basepath, subj_name)
                anims = os.listdir(subj_path) # animation list
                cams_subj = {} # dict to store camera params for each subject, where keys are anim indices
                for anim_name in anims:
                    anim_idx = int(anim_name.split('_')[1]) # animation index
                    anim_path = os.path.join(subj_path, anim_name)
                    files = os.listdir(anim_path)
                    cams = {} # dict to store camera params for each subject with each animations
                    for f in files:
                        f_path = os.path.join(anim_path, f)
                        if not os.path.isdir(f_path):
                            f_fn = os.path.splitext(f)[0] # filename without extension
                            if ('camera' in f_fn) and (f_fn.split('_')[1] in self.camera_names):
                                # load cameras
                                camera_idx = self.camera_names.index(f_fn.split('_')[1])
                                with open(f_path) as json_file:
                                    camera_dict = json.load(json_file)

                                pos = camera_dict['location']
                                rot = camera_dict['rotation']
                                width = camera_dict['width']
                                height = camera_dict['height']
                                fov = camera_dict['fov']

                                cam = Camera(pos[0], pos[1], pos[2],
                                             rot[2], rot[0], rot[1],
                                             width, height, width/2)
                                cams[camera_idx] = cam
                            elif 'skeleton' in f_fn:
                                # record frame list
                                frame_idx = f_fn.split('_')[1]
                                self.framelist.append([subj_idx, anim_idx, int(frame_idx), bbox])
                                self.len += 1
                                
                        cams_subj[anim_idx] = cams
                    self.cameras[subj_idx] = cams_subj

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        sample = self.framelist[idx]

        data = {}
        images = [] # list of image tensors
        cameras = [] # list of camera instances

        # original form: 0
        if self.form == 0:
            subj_idx, frame, bbox = sample[0], sample[1], sample[2]
            subj = self.subj[subj_idx]
            frame_path = os.path.join(self.basepath, subj, '%06d' % frame)
            skeleton_path = os.path.join(frame_path, 'skeleton.json')

            # load data
            for camera_idx, camera_name in enumerate(self.camera_names):
                image_path = os.path.join(frame_path, 'view_%s.png' % camera_name)
                assert os.path.isfile(image_path)
                image = Image.open(image_path) # RGB
                cam = self.cameras[subj_idx][camera_idx]
            
                if bbox is not None:
                    image = image.crop(bbox)
                    cam.update_after_crop(bbox)

                image_tensor = self.transform(image)
                images.append(image_tensor)
                cameras.append(cam)

        # original form: 1
        elif self.form == 1:
            subj_idx, anim_idx, frame, bbox = sample[0], sample[1], sample[2], sample[3]
            subj = self.subj[subj_idx]
            anim_path = os.path.join(self.basepath, subj, 'anim_%03d' % anim_idx)
            skeleton_path = os.path.join(anim_path, 'skeleton_%06d.json' % frame)

            # load data
            for camera_idx, camera_name in enumerate(self.camera_names):
                image_path = os.path.join(anim_path, camera_name, '%06d.jpg' % frame)
                assert os.path.isfile(image_path)
                image = Image.open(image_path) # RGB
                cam = self.cameras[subj_idx][anim_idx][camera_idx]
            
                if bbox is not None:
                    image = image.crop(bbox)
                    cam.update_after_crop(bbox)

                image_tensor = self.transform(image)
                images.append(image_tensor)
                cameras.append(cam)


        images = torch.stack(images, dim=0) # tensor of size (num_views x 3 x h x w) 
        data['images'] = images
        data['cameras'] = cameras

        with open(skeleton_path) as skeleton_json:
            skeleton = json.load(skeleton_json)

        all_joints = {joint["Name"].lower(): joint["KpWorld"] for joint in skeleton} # lowercase
        x_keypts = np.array([all_joints[jnt]['X'] for jnt in self.joints_name])
        y_keypts = np.array([all_joints[jnt]['Y'] for jnt in self.joints_name])
        z_keypts = np.array([all_joints[jnt]['Z'] for jnt in self.joints_name])
        keypoints = np.hstack((x_keypts.reshape(self.num_jnts,1),\
                               y_keypts.reshape(self.num_jnts,1),\
                               z_keypts.reshape(self.num_jnts,1)))
        keypts_tensor = torch.from_numpy(keypoints)
        data['joints_3d_gt'] = keypts_tensor # joints groundtruth, tensor of size n x 3

        keypts_valid = torch.ones(self.num_jnts, 1)
        keypts_valid[self.invalid_jnts, :] = 0
        data['joints_3d_valid'] = keypts_valid # binary tensor of size n x 1
        
        return data
            
        
