import numpy as np

import torch
from torch import nn

import utils.visualize as visualize
from utils.op import render_points_as_2d_gaussians


class HeatmapMSELoss(nn.Module):
    def __init__(self, config):
        super(HeatmapMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.image_shape = config.dataset.image_shape # [w, h]

    def forward(self, heatmaps_pred, proj_mats_batch, joints_3d_gt_batch, joints_3d_valid_batch):
        batch_size = heatmaps_pred.shape[0]
        num_views = heatmaps_pred.shape[1]
        heatmap_shape = tuple(heatmaps_pred.shape[3:]) # [h, w]
        ratio_w = heatmap_shape[1] / self.image_shape[0]
        ratio_h = heatmap_shape[0] / self.image_shape[1]
        heatmaps_gt = torch.zeros_like(heatmaps_pred)
        joints_2d_gt_batch = visualize.proj_to_2D_batch(proj_mats_batch, joints_3d_gt_batch)
        joints_2d_gt_batch_scaled = torch.zeros_like(joints_2d_gt_batch)
        joints_2d_gt_batch_scaled[:, :, :, 0] = joints_2d_gt_batch[:, :, :, 0] * ratio_w
        joints_2d_gt_batch_scaled[:, :, :, 1] = joints_2d_gt_batch[:, :, :, 1] * ratio_h
        for batch_idx in range(batch_size):
            for view_idx in range(num_views):
                joints_2d_gt_scaled = joints_2d_gt_batch_scaled[batch_idx, view_idx, ...]
                sigmas = torch.ones_like(joints_2d_gt_scaled) # unit str
                heatmaps_gt[batch_idx, view_idx, ...] = render_points_as_2d_gaussians(joints_2d_gt_scaled, sigmas, heatmap_shape)
        loss = self.criterion(heatmaps_pred, heatmaps_gt)
        return loss * heatmap_shape[0] * heatmap_shape[1]


class PCK(nn.Module):
    def __init__(self, thresh=0.2):
        super(PCK, self).__init__()
        self.thresh = thresh

    def forward(self, joints_2d_pred, proj_mats_batch, joints_3d_gt_batch, joints_3d_valid_batch):
        num_views = joints_2d_pred.shape[1]
        joints_2d_gt_batch = visualize.proj_to_2D_batch(proj_mats_batch, joints_3d_gt_batch)
        bbox_w = joints_2d_gt_batch[..., 0].max(-1, keepdim=True)[0] - joints_2d_gt_batch[..., 0].min(-1, keepdim=True)[0] # batch_size x num_views x 1
        bbox_h = joints_2d_gt_batch[..., 1].max(-1, keepdim=True)[0] - joints_2d_gt_batch[..., 1].min(-1, keepdim=True)[0] # batch_size x num_views x 1
        torso_diam = torch.max(bbox_w, bbox_h) # batch_size x num_views x 1
        # print(torso_diam)
        diff = joints_2d_pred - joints_2d_gt_batch
        dist = torch.norm(diff, dim=-1) # batch_size x num_views x num_joints
        detected = (dist < self.thresh * torso_diam).sum()
        total_joints = num_views * (joints_3d_valid_batch == 1).sum()
        return detected, total_joints


class KeypointsMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, keypoints_pred, keypoints_gt, keypoints_binary_validity):
        dimension = keypoints_pred.shape[-1]
        loss = torch.sum((keypoints_gt - keypoints_pred) ** 2 * keypoints_binary_validity)
        loss = loss / (dimension * max(1, torch.sum(keypoints_binary_validity).item()))
        return loss

class KeypointsMSESmoothLoss(nn.Module):
    def __init__(self, threshold=400):
        super().__init__()

        self.threshold = threshold

    def forward(self, keypoints_pred, keypoints_gt, keypoints_binary_validity):
        dimension = keypoints_pred.shape[-1]
        diff = (keypoints_gt - keypoints_pred) ** 2 * keypoints_binary_validity
        diff[diff > self.threshold] = torch.pow(diff[diff > self.threshold], 0.1) * (self.threshold ** 0.9)
        loss = torch.sum(diff) / (dimension * max(1, torch.sum(keypoints_binary_validity).item()))
        return loss


class KeypointsMAELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, keypoints_pred, keypoints_gt, keypoints_binary_validity):
        dimension = keypoints_pred.shape[-1]
        loss = torch.sum(torch.abs(keypoints_gt - keypoints_pred) * keypoints_binary_validity)
        loss = loss / (dimension * max(1, torch.sum(keypoints_binary_validity).item()))
        return loss


class KeypointsL2Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, keypoints_pred, keypoints_gt, keypoints_binary_validity):
        loss = torch.sum(torch.sqrt(torch.sum((keypoints_gt - keypoints_pred) ** 2 * keypoints_binary_validity, dim=2)))
        loss = loss / max(1, torch.sum(keypoints_binary_validity).item())
        return loss


class VolumetricCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, coord_volumes_batch, volumes_batch_pred, keypoints_gt, keypoints_binary_validity):
        loss = 0.0
        n_losses = 0

        batch_size = volumes_batch_pred.shape[0]
        for batch_i in range(batch_size):
            coord_volume = coord_volumes_batch[batch_i]
            keypoints_gt_i = keypoints_gt[batch_i]

            coord_volume_unsq = coord_volume.unsqueeze(0)
            keypoints_gt_i_unsq = keypoints_gt_i.unsqueeze(1).unsqueeze(1).unsqueeze(1)

            dists = torch.sqrt(((coord_volume_unsq - keypoints_gt_i_unsq) ** 2).sum(-1))
            dists = dists.view(dists.shape[0], -1)

            min_indexes = torch.argmin(dists, dim=-1).detach().cpu().numpy()
            min_indexes = np.stack(np.unravel_index(min_indexes, volumes_batch_pred.shape[-3:]), axis=1)

            for joint_i, index in enumerate(min_indexes):
                validity = keypoints_binary_validity[batch_i, joint_i]
                loss += validity[0] * (-torch.log(volumes_batch_pred[batch_i, joint_i, index[0], index[1], index[2]] + 1e-6))
                n_losses += 1


        return loss / n_losses
