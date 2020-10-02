import numpy as np

import torch
from torch import nn

import utils.visualize as visualize
from utils.op import render_points_as_2d_gaussians


class PCK(nn.Module):
    """
    2D Percentage of Correct Keypoints.
    default: PCK@0.2
    """
    def __init__(self, thresh=0.2):
        super(PCK, self).__init__()
        self.thresh = thresh

    def forward(self, joints_2d_pred, joints_2d_gt_batch, joints_2d_valid_batch):
        num_views = joints_2d_pred.shape[1]
        num_joints = min(joints_2d_pred.shape[2], joints_2d_gt_batch.shape[2])
        joints_2d_valid_batch = self.get_joints_validity(joints_2d_gt_batch, joints_2d_valid_batch, num_joints)
        bbox_w = joints_2d_gt_batch[..., 0].max(-1, keepdim=True)[0] - joints_2d_gt_batch[..., 0].min(-1, keepdim=True)[0] # batch_size x num_views x 1
        bbox_h = joints_2d_gt_batch[..., 1].max(-1, keepdim=True)[0] - joints_2d_gt_batch[..., 1].min(-1, keepdim=True)[0] # batch_size x num_views x 1
        torso_diam = torch.max(bbox_w, bbox_h) # batch_size x num_views x 1
        # print(torso_diam)
        diff = joints_2d_pred[:, :, 0:num_joints, :] - joints_2d_gt_batch[:, :, 0:num_joints, :]
        dist = torch.norm(diff, dim=-1) # batch_size x num_views x num_joints
        detected = ((dist < self.thresh * torso_diam) * joints_2d_valid_batch).sum(dtype=torch.float32)
        # total_joints = num_views * (joints_3d_valid_batch == 1).sum()
        total_joints = (joints_2d_valid_batch == 1).sum()
        return detected, total_joints

    def get_joints_validity(self, joints_2d_gt_batch, joints_2d_valid_batch, num_joints):
        # zero-coordinate indicated invisibility
        joints_2d_gt_zero = (joints_2d_gt_batch == 0)
        joints_2d_visible = ~(joints_2d_gt_zero[:, :, :, 0] & joints_2d_gt_zero[:, :, :, 1])
        joints_2d_valid_batch = joints_2d_valid_batch[:, :, 0:num_joints, :]
        joints_2d_valid_batch = (joints_2d_valid_batch.squeeze(-1) & joints_2d_visible)
        if num_joints == 16: # mpii, not include thorax
            joints_2d_valid_batch[:, :, 7] = False
        return joints_2d_valid_batch # batch_size x num_views x num_joints


class PCKh(nn.Module):
    """
    2D Percentage of Correct Keypoints (threshold on head length).
    default: PCKh@0.5
    """
    def __init__(self, thresh=0.5):
        super(PCKh, self).__init__()
        self.thresh = thresh

    def forward(self, joints_2d_pred, joints_2d_gt_batch, joints_2d_valid_batch):
        num_views = joints_2d_pred.shape[1]
        num_joints = min(joints_2d_pred.shape[2], joints_2d_gt_batch.shape[2])
        joints_2d_valid_batch = self.get_joints_validity(joints_2d_gt_batch, joints_2d_valid_batch, num_joints)
        head_length = torch.norm(joints_2d_gt_batch[:, :, 9, :] - joints_2d_gt_batch[:, :, 16, :], dim=-1, keepdim=True) # batch_size x num_views x 1
        diff = joints_2d_pred[:, :, 0:num_joints, :] - joints_2d_gt_batch[:, :, 0:num_joints, :]
        dist = torch.norm(diff, dim=-1) # batch_size x num_views x num_joints
        detected = ((dist < self.thresh * head_length) * joints_2d_valid_batch).sum(dtype=torch.float32)
        # total_joints = num_views * (joints_3d_valid_batch == 1).sum()
        total_joints = (joints_2d_valid_batch == 1).sum()
        return detected, total_joints

    def get_joints_validity(self, joints_2d_gt_batch, joints_2d_valid_batch, num_joints):
        # zero-coordinate indicated invisibility
        joints_2d_gt_zero = (joints_2d_gt_batch == 0)
        joints_2d_visible = ~(joints_2d_gt_zero[:, :, :, 0] & joints_2d_gt_zero[:, :, :, 1])
        joints_2d_valid_batch = joints_2d_valid_batch[:, :, 0:num_joints, :]
        joints_2d_valid_batch = (joints_2d_valid_batch.squeeze(-1) & joints_2d_visible)
        if num_joints == 16: # mpii, not include thorax
            joints_2d_valid_batch[:, :, 7] = False
        return joints_2d_valid_batch # batch_size x num_views x num_joints


class PCK3D(nn.Module):
    """
    3D Percentage of Correct Keypoints.
    default threshold: 150mm
    """
    def __init__(self, thresh=150):
        super(PCK3D, self).__init__()
        self.thresh = thresh

    def forward(self, joints_3d_pred, joints_3d_gt_batch, joints_3d_valid_batch):
        diff = joints_3d_pred - joints_3d_gt_batch
        dist = torch.norm(diff, dim=-1) # batch_size x num_joints
        detected = (dist < self.thresh).sum(dtype=torch.float32)
        total_joints = (joints_3d_valid_batch == 1).sum()
        return detected, total_joints

