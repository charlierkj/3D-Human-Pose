import os, json
import numpy as np
import torch
import torchvision as tv
import torch.utils.data as td
from PIL import Image

from camera_utils import *
import visualize


Joints_SynData = [
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
    "head",
    "ball_r",
    "ball_l",
    "middle_03_r",
    "thumb_03_r",
    "thumb_03_l",
    "middle_03_l"
    ] # need modify


def load_joints(joints_name, skeleton_path):
    # return numpy array of size num_joints x 3.
    num_jnts = len(joints_name)
    with open(skeleton_path) as skeleton_json:
        skeleton = json.load(skeleton_json)
        
    all_joints = {joint["Name"].lower(): joint["KpWorld"] for joint in skeleton} # lowercase
    x_keypts = np.array([all_joints[jnt]['X'] for jnt in joints_name])
    y_keypts = np.array([all_joints[jnt]['Y'] for jnt in joints_name])
    z_keypts = np.array([all_joints[jnt]['Z'] for jnt in joints_name])
    keypoints = np.hstack((x_keypts.reshape(num_jnts,1),\
                           y_keypts.reshape(num_jnts,1),\
                           z_keypts.reshape(num_jnts,1)))
    return keypoints


def load_image(image_path, bbox=None, size=None):
    # return tensor
    assert os.path.isfile(image_path)
    image = Image.open(image_path) # RGB
    if bbox is not None:
        image = image.crop(bbox)
    if size is not None:
        image = image.resize(size)
    transform = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(visualize.IMAGENET_MEAN, visualize.IMAGENET_STD)
        ])
    image_tensor = transform(image)
    return image_tensor


def load_camera(camera_file):
    with open(camera_file) as camera_json:
        camera_dict = json.load(camera_json)

    pos = camera_dict['location']
    rot = camera_dict['rotation']
    width = camera_dict['width']
    height = camera_dict['height']
    fov = camera_dict['fov']

    cam = Camera(pos[0], pos[1], pos[2],
                 rot[2], rot[0], rot[1],
                 width, height, width/2)
    return cam

    
def collate_fn(batch):
    # mainly used to convert numpy to tensor and concatenate,
    # with making sure the data type of tensors is float.
    samples = list(d for d in batch if d is not None)
    if len(samples) == 0:
        print("current batch is empty")
        return None

    images_batch = torch.stack([sample['images'] for sample in samples], dim=0) # batch_size x num_views x 3 x h x w
    proj_mats_batch = torch.stack([torch.stack([torch.from_numpy(cam.get_P()) for cam in sample['cameras']], dim=0)\
                                   for sample in samples], dim=0) # batch_size x num_views x 3 x 4
    joints_3d_gt_batch = torch.stack([sample['joints_3d_gt'] for sample in samples], dim=0) # batch_size x num_joints x 3
    joints_3d_valid_batch = torch.stack([sample['joints_3d_valid'] for sample in samples], dim=0) # batch_size x num_joints x 1

    images_batch, proj_mats_batch, joints_3d_gt_batch, joints_3d_valid_batch \
                  = images_batch.type(torch.float32), proj_mats_batch.type(torch.float32), joints_3d_gt_batch.type(torch.float32), joints_3d_valid_batch.type(torch.float32)
    info_batch = [[int(s) for s in sample['info'].split('_')] for sample in samples]
    return images_batch, proj_mats_batch, joints_3d_gt_batch, joints_3d_valid_batch, info_batch


def syndata_loader(dataset, batch_size=1, shuffle=False):
    dataloader = td.DataLoader(dataset,
                               batch_size=batch_size,
                               shuffle=shuffle,
                               collate_fn = collate_fn,
                               pin_memory=True)
    return dataloader
    
