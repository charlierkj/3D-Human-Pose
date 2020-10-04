import os
import numpy as np
import json

import torch
from torch import nn
import torch.utils.data as td
import torchvision as tv

from PIL import Image

import imgaug as ia
import imgaug.augmenters as iaa

from utils.ue import *
import datasets.utils as datasets_utils
import utils.visualize as visualize


class MultiView_SynData(td.Dataset):

    def __init__(self,
                 path="../mocap_syndata/multiview_data",
                 num_camera=4,
                 load_joints=17,
                 invalid_joints=(),
                 bbox=None,
                 image_shape=[384, 384], 
                 train=False,
                 test=False, 
                 with_aug=False):
        """
        invalid_joints: tuple of indices for invalid joints; associated joints will not be used in evaluation.
        bbox: [upper_left_x, upper_left_y, lower_right-x, lower_right_y].
        image_size: [width, height].
        with_aug: use or not image augmentation.
        """
        assert train or test

        train_subj = ['S0map', 'S0rand', 'S0real', \
                      'S1map', 'S1rand', 'S1real', \
                      'S2map', 'S2rand', 'S2real', \
                      'S3map', 'S3rand', 'S3real']
        
        test_subj = ['S4map', 'S4rand', 'S4real', \
                     'S5map', 'S5rand', 'S5real']

        if train:
            self.subj = train_subj
        elif test:
            self.subj = test_subj
        
        self.basepath = path
        # self.subj = sorted(os.listdir(self.basepath))
        self.framelist = []
        self.camera_names = []
        self.cameras = {}
        self.len = 0

        self.invalid_jnts = () if invalid_joints is None else invalid_joints

        self.image_shape = image_shape

        # torchvision transforms
        # self.transform = tv.transforms.Compose([
        #     tv.transforms.ToTensor(),
        #     tv.transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        #     ])

        # joint names 
        self.joints_name = datasets_utils.get_joints_name(load_joints)

        self.num_jnts = len(self.joints_name)
        assert self.num_jnts == load_joints

        for camera_idx in range(num_camera):
            self.camera_names.append("%04d" % camera_idx)

        for (subj_idx, subj_name) in enumerate(self.subj):
            subj_path = os.path.join(self.basepath, subj_name)
            anims = sorted(os.listdir(subj_path)) # animation list
            cams_subj = {} # dict to store camera params for each subject, where keys are anim indices
            for anim_name in anims:
                anim_idx = int(anim_name.split('_')[1]) # animation index
                anim_path = os.path.join(subj_path, anim_name)
                files = sorted(os.listdir(anim_path))
                cams = {} # dict to store camera params for each subject with each animations
                for f in files:
                    f_path = os.path.join(anim_path, f)
                    if not os.path.isdir(f_path):
                        f_fn = os.path.splitext(f)[0] # filename without extension
                        if ('camera' in f_fn) and (f_fn.split('_')[1] in self.camera_names):
                            # load cameras
                            camera_idx = self.camera_names.index(f_fn.split('_')[1])
                            cam = datasets_utils.load_camera(f_path)
                            cams[camera_idx] = cam
                        elif 'skeleton' in f_fn:
                            # record frame list
                            frame_idx = f_fn.split('_')[1]
                            self.framelist.append([subj_idx, anim_idx, int(frame_idx), bbox])
                            self.len += 1
                                
                    cams_subj[anim_idx] = cams
                self.cameras[subj_idx] = cams_subj

        # augmentation
        self.aug = None
        if train and with_aug:
            sometimes = lambda aug: iaa.Sometimes(0.5, aug)
            self.aug = iaa.Sequential(
                [
                    sometimes(iaa.Affine(
                        scale={"x": (0.75, 1.25), "y": (0.75, 1.25)}, # scale to 75-125% of image height/width (each axis independently)
                        translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)}, # translate by -5 to +5 relative to height/width (per axis)
                        rotate=(-30, 30), # rotate by -30 to +30 degrees
                        shear=(-20, 20), # shear by -20 to +20 degrees
                        order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                        cval=(0, 255), # if the mode is constant, then use a random brightness for the newly created pixels
                        mode='constant' # use any available mode to fill newly created pixels, see API or scikit-image for which modes are available
                    )),
                    sometimes(iaa.AdditiveGaussianNoise(scale=0.5*2, per_channel=0.5)),
                    sometimes(iaa.GaussianBlur(sigma=(1.0, 5.0))),
                    sometimes(iaa.ContrastNormalization((0.8, 1.25), per_channel=0.5)) # improve or worsen the contrast
                ],
                random_order=True
            )

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        sample = self.framelist[idx]

        data = {}
        images = [] # list of image tensors
        cameras = [] # list of camera instances

        subj_idx, anim_idx, frame, bbox = sample[0], sample[1], sample[2], sample[3]
        subj = self.subj[subj_idx]
        anim_path = os.path.join(self.basepath, subj, 'anim_%03d' % anim_idx)
        skeleton_path = os.path.join(anim_path, 'skeleton_%06d.json' % frame)

        # load data
        for camera_idx, camera_name in enumerate(self.camera_names):
            image_path = os.path.join(anim_path, camera_name, '%06d.jpg' % frame)
            image_tensor = datasets_utils.load_image(image_path, bbox, self.image_shape)
            cam = self.cameras[subj_idx][anim_idx][camera_idx]
            
            if bbox is not None:
                cam.update_after_crop(bbox)
                cam.update_after_resize(self.image_shape)

            images.append(image_tensor)
            cameras.append(cam)


        images = torch.stack(images, dim=0) # tensor of size (num_views x 3 x h x w) 
        data['images'] = images
        #data['cameras'] = cameras
        
        proj_mats = torch.stack([torch.from_numpy(cam.get_P()) for cam in cameras], dim=0) # num_views x 3 x 4
        data['proj_mats'] = proj_mats

        keypoints = datasets_utils.load_joints(self.joints_name, skeleton_path)
        #preds_path = os.path.join("results/mocap_syndata/preds", self.subj[subj_idx], "anim_%03d" % anim_idx, "joints_3d.npy")
        #print(os.path.abspath(preds_path))
        #joints_3d_pred = np.load(preds_path)
        #keypoints[self.invalid_jnts, :] = joints_3d_pred[frame, self.invalid_jnts, :] # load preds as gt for head joints
        keypts_tensor = torch.from_numpy(keypoints)
        data['joints_3d_gt'] = keypts_tensor # joints groundtruth, tensor of size n x 3

        keypts_valid = torch.ones(self.num_jnts, 1)
        keypts_valid[self.invalid_jnts, :] = 0
        data['joints_3d_valid'] = keypts_valid # binary tensor of size n x 1

        keypts_2d = visualize.proj_to_2D_batch(proj_mats.unsqueeze(0), keypts_tensor.unsqueeze(0))
        keypts_2d = keypts_2d.squeeze(0)
        data['joints_2d_gt'] = keypts_2d # 2d joints groundtruth, tensor of size (num_views x num_joints x 2)

        # augmentation
        if self.aug is not None:
            data['images'], data['joints_2d_gt'] = self.augment(images, keypts_2d)

        data['info'] = '%s_%03d_%06d' % (self.subj[subj_idx], anim_idx, frame)

        return data

    def augment(self, images, keypts_2d):
        images_np = images.numpy()
        images_np = np.moveaxis(images_np, 1, -1) # num_views x C x H x W -> num_views x H x W x C
        keypts_2d_np = keypts_2d.numpy() # num_views x num_joints x 2
        #images_np = images_np[..., (2, 1, 0)] # BGR -> RGB
        #images_np = np.clip(255 * (images_np * visualize.IMAGENET_STD + visualize.IMAGENET_MEAN), 0, 255).astype(np.uint8) # denormalize
        images_np_aug = np.zeros(shape=images_np.shape)
        keypts_2d_np_aug = np.zeros(shape=keypts_2d_np.shape)
        num_views = images.shape[0]
        for view_idx in range(num_views):
            img, k = self.aug(image=images_np[view_idx], \
                              keypoints=np.expand_dims(keypts_2d_np[view_idx], axis=0))
            images_np_aug[view_idx, :, :, :] = img
            keypts_2d_np_aug[view_idx, :, :] = k.squeeze(0)
        images_np_aug = np.moveaxis(images_np_aug, -1, 1) # num_views x H x W x C -> num_views x C x H x W
        images_aug = torch.from_numpy(images_np_aug)
        keypts_2d_aug = torch.from_numpy(keypts_2d_np_aug)
        return images_aug, keypts_2d_aug



class MultiView_SynHand(MultiView_SynData):

    def __init__(self,
                 path="../mocap_syndata/multiview_data",
                 num_camera=4,
                 load_joints=21,
                 invalid_joints=(),
                 bbox=None,
                 image_shape=[384, 384], 
                 train=False,
                 test=False, 
                 with_aug=False):
        """
        invalid_joints: tuple of indices for invalid joints; associated joints will not be used in evaluation.
        bbox: [upper_left_x, upper_left_y, lower_right-x, lower_right_y].
        image_size: [width, height].
        with_aug: use or not image augmentation.
        """
        super(MultiView_SynHand, self).__init__(path,
                                                num_camera,
                                                load_joints,
                                                invalid_joints,
                                                bbox,
                                                image_shape,
                                                train,
                                                test,
                                                with_aug)

        # right / left hand joints name
        self.joints_name_rh = self.joints_name[0:21]
        self.joints_name_lh = self.joints_name[21:42]

        # right / left forearm joints name, used for finding hand bbox
        self.joints_name_rf = ["lowerarm_r", "hand_r"]
        self.joints_name_lf = ["lowerarm_l", "hand_l"]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # currently only support for 2D
        
        sample = self.framelist[idx]

        cameras = [] # list of camera instances

        subj_idx, anim_idx, frame, _ = sample[0], sample[1], sample[2], sample[3]
        subj = self.subj[subj_idx]
        anim_path = os.path.join(self.basepath, subj, 'anim_%03d' % anim_idx)
        skeleton_path = os.path.join(anim_path, 'skeleton_%06d.json' % frame)

        # 3D keypoints groundtruth
        keypts_rh = torch.from_numpy(datasets_utils.load_joints(self.joints_name_rh, skeleton_path))
        keypts_lh = torch.from_numpy(datasets_utils.load_joints(self.joints_name_lh, skeleton_path))
        keypts_rf = torch.from_numpy(datasets_utils.load_joints(self.joints_name_rf, skeleton_path))
        keypts_lf = torch.from_numpy(datasets_utils.load_joints(self.joints_name_lf, skeleton_path))

        # load camera
        for camera_idx, camera_name in enumerate(self.camera_names):
            cam = self.cameras[subj_idx][anim_idx][camera_idx]
            cameras.append(cam)
        proj_mats = torch.stack([torch.from_numpy(cam.get_P()) for cam in cameras], dim=0) # num_views x 3 x 4

        # projected 2D keypoints groundtruth (unprocessed)
        keypts_2d_rh = visualize.proj_to_2D_batch(proj_mats.unsqueeze(0), keypts_rh.unsqueeze(0))
        keypts_2d_rh = keypts_2d_rh.squeeze(0) # num_views x 21 x 2
        keypts_2d_lh = visualize.proj_to_2D_batch(proj_mats.unsqueeze(0), keypts_lh.unsqueeze(0))
        keypts_2d_lh = keypts_2d_lh.squeeze(0)
        keypts_2d_rf = visualize.proj_to_2D_batch(proj_mats.unsqueeze(0), keypts_rf.unsqueeze(0))
        keypts_2d_rf = keypts_2d_rf.squeeze(0) # num_views x 2 x 2
        keypts_2d_lf = visualize.proj_to_2D_batch(proj_mats.unsqueeze(0), keypts_lf.unsqueeze(0))
        keypts_2d_lf = keypts_2d_lf.squeeze(0)

        # generate bboxes
        bbox_rh = torch.zeros((4, 4)) # hardcoded
        bbox_lh = torch.zeros((4, 4))
        center_r = keypts_2d_rf[:, 1, :] + 0.15 * (keypts_2d_rf[:, 1, :] - keypts_2d_rf[:, 0, :]) # num_views x 2
        center_l = keypts_2d_lf[:, 1, :] + 0.15 * (keypts_2d_lf[:, 1, :] - keypts_2d_lf[:, 0, :]) # num_views x 2
        w_rh = keypts_2d_rh[:, :, 0].max(-1)[0] - keypts_2d_rh[:, :, 0].min(-1)[0] # num_views
        h_rh = keypts_2d_rh[:, :, 1].max(-1)[0] - keypts_2d_rh[:, :, 1].min(-1)[0] # num_views
        B_rh = 2.2 * torch.max(w_rh, h_rh) # num_views
        w_lh = keypts_2d_lh[:, :, 0].max(-1)[0] - keypts_2d_lh[:, :, 0].min(-1)[0] # num_views
        h_lh = keypts_2d_lh[:, :, 1].max(-1)[0] - keypts_2d_lh[:, :, 1].min(-1)[0] # num_views
        B_lh = 2.2 * torch.max(w_lh, h_lh) # num_views
        bbox_rh[:, 0] = (center_r[:, 0] - B_rh / 2).type(torch.int)
        bbox_rh[:, 1] = (center_r[:, 1] - B_rh / 2).type(torch.int)
        bbox_rh[:, 2] = (center_r[:, 0] + B_rh / 2).type(torch.int)
        bbox_rh[:, 3] = (center_r[:, 1] + B_rh / 2).type(torch.int)
        bbox_lh[:, 0] = (center_l[:, 0] - B_lh / 2).type(torch.int)
        bbox_lh[:, 1] = (center_l[:, 1] - B_lh / 2).type(torch.int)
        bbox_lh[:, 2] = (center_l[:, 0] + B_lh / 2).type(torch.int)
        bbox_lh[:, 3] = (center_l[:, 1] + B_lh / 2).type(torch.int)

        # process 2D keypoint groundtruth
        keypts_2d_rh[:, 0] = keypts_2d_rh[:, 0] - bbox_rh[:, 0]
        keypts_2d_rh[:, 1] = keypts_2d_rh[:, 1] - bbox_rh[:, 1]
        keypts_2d_rh[:, 0] = keypts_2d_rh[:, 0] * self.image_shape[0] / (bbox_rh[:, 2] - bbox_rh[:, 0])
        keypts_2d_rh[:, 1] = keypts_2d_rh[:, 1] * self.image_shape[1] / (bbox_rh[:, 3] - bbox_rh[:, 1])
        keypts_2d_lh[:, 0] = keypts_2d_lh[:, 0] - bbox_lh[:, 0]
        keypts_2d_lh[:, 1] = keypts_2d_lh[:, 1] - bbox_lh[:, 1]
        keypts_2d_lh[:, 0] = keypts_2d_lh[:, 0] * self.image_shape[0] / (bbox_lh[:, 2] - bbox_lh[:, 0])
        keypts_2d_lh[:, 1] = keypts_2d_lh[:, 1] * self.image_shape[1] / (bbox_lh[:, 3] - bbox_lh[:, 1])
        keypts_2d_lh[:, 0] = self.image_shape[0] - keypts_2d_lh[:, 0] # flip left to right
        keypts_2d = torch.cat((keypts_2d_rh, keypts_2d_lh), dim=0) # 2*num_views x 21 x 2
        
        # load cropped and resized images
        images_rh = []
        images_lh = []
        for camera_idx, camera_name in enumerate(self.camera_names):
            image_path = os.path.join(anim_path, camera_name, '%06d.jpg' % frame)
            image_rh = datasets_utils.load_image(image_path, bbox_rh[camera_idx, :], self.image_shape)
            image_lh = datasets_utils.load_image(image_path, bbox_lh[camera_idx, :], self.image_shape, flip=True)
            images_rh.append(image_rh)
            images_lh.append(image_lh)

        images = torch.stack(images_rh + images_lh, dim=0) # tensor of size (2*num_views x 3 x h x w) 

        # augmentation
        if self.aug is not None:
            images, keypts_2d = self.augment(images, keypts_2d)

        # info = '%s_%03d_%06d' % (self.subj[subj_idx], anim_idx, frame)

        return images, keypts_2d

    def augment(self, images, keypts_2d):
        return super(MultiView_SynHand, self).augment(images, keypts_2d)         
        
