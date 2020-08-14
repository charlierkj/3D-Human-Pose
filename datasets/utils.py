import os, json
import numpy as np
import torch
import torchvision as tv
import torch.utils.data as td
from PIL import Image

from camera_utils import *
import visualize

from utils.img import image_batch_to_torch

# Synthetic Dataset
Joints_SynData = [
    "foot_r", #0
    "calf_r", #1
    "thigh_r", #2
    "thigh_l", #3
    "calf_l", #4
    "foot_l", #5
    "pelvis", #6
    "spine_02", #7
    "neck_01", #8
    "head", #9
    "hand_r", #10
    "lowerarm_r", #11
    "upperarm_r", #12
    "upperarm_l", #13
    "lowerarm_l", #14
    "hand_l", #15
    "head", #16
    "ball_r", #17
    "ball_l", #18
    "middle_03_r", #19
    "thumb_03_r", #20
    "thumb_03_l", #21
    "middle_03_l" #22
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

    
def syndata_collate_fn(batch):
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
                               collate_fn = syndata_collate_fn,
                               pin_memory=True)
    return dataloader



# Human3.6M Dataset
# Reference: https://github.com/karfly/learnable-triangulation-pytorch
def human36m_collate_fn(items):
        #items = list(filter(lambda x: x is not None, items))
        items = list(x for x in items if x is not None)
        if len(items) == 0:
            print("All items in batch are None")
            return None

        batch = dict()
        total_n_views = min(len(item['images']) for item in items)

        indexes = np.arange(total_n_views)
    
        batch['images'] = np.stack([np.stack([item['images'][i] for item in items], axis=0) for i in indexes], axis=0).swapaxes(0, 1)
        batch['detections'] = np.array([[item['detections'][i] for item in items] for i in indexes]).swapaxes(0, 1)
        batch['cameras'] = [[item['cameras'][i] for item in items] for i in indexes]

        batch['keypoints_3d'] = [item['keypoints_3d'] for item in items]
        # batch['cuboids'] = [item['cuboids'] for item in items]
        batch['indexes'] = [item['indexes'] for item in items]

        try:
            batch['pred_keypoints_3d'] = np.array([item['pred_keypoints_3d'] for item in items])
        except:
            pass

        return human36m_prepare_batch(batch)


def human36m_prepare_batch(batch):
    # images
    images_batch = []
    for image_batch in batch['images']:
        image_batch = image_batch_to_torch(image_batch)
        image_batch = image_batch.to(device)
        images_batch.append(image_batch)

    images_batch = torch.stack(images_batch, dim=0)

    # 3D keypoints
    keypoints_3d_batch_gt = torch.from_numpy(np.stack(batch['keypoints_3d'], axis=0)[:, :, :3]).float().to(device)

    # 3D keypoints validity
    keypoints_3d_validity_batch_gt = torch.from_numpy(np.stack(batch['keypoints_3d'], axis=0)[:, :, 3:]).float().to(device)

    # projection matricies
    proj_matricies_batch = torch.stack([torch.stack([torch.from_numpy(camera.projection) for camera in camera_batch], dim=0) for camera_batch in batch['cameras']], dim=0).transpose(1, 0)  # shape (batch_size, n_views, 3, 4)
    proj_matricies_batch = proj_matricies_batch.float().to(device)

    return images_batch, proj_matricies_batch, keypoints_3d_batch_gt, keypoints_3d_validity_batch_gt, None


def human36m_loader(dataset, batch_size=1, shuffle=False):
    dataloader = td.DataLoader(dataset,
                               batch_size=batch_size,
                               shuffle=shuffle,
                               collate_fn = human36m_collate_fn,
                               pin_memory=True)
    return dataloader
