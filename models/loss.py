import numpy as np

import torch
from torch import nn


class HeatmapMSELoss(nn.Module):
    def __init__(self, image_shape):
        super(HeatmapMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.image_shape = image_shape

    def forward(self, heatmaps_pred, proj_mats_batch, joints_3d_gt_batch, joints_3d_valid_batch):
        batch_size = heatmaps_pred.shape[0]
        num_views = heatmaps_pred.shape[1]
        num_joints = heatmaps_pred.shape[2]
        heatmap_shape = tuple(heatmaps_pred.shape[3:])
        heatmaps_gt = torch.zeros_like(heatmaps_pred)
        proj_mats_batch = proj_mats_batch.view(-1, 3, 4)
        for batch_idx in range(batch_size):
            joints_3d_gt = joints_3d_gt_batch[batch_idx, ...]
            for view_idx in range(num_views):
                proj_mat = proj_mats_batch[batch_idx, view_idx, ...]
                joints_2d_gt = visualize.proj_to_2D(proj_mat, joints_3d_gt.T)
                joints_2d_gt = joints_2d_gt.T
                heatmaps_gt[batch_idx, view_idx, ...] = render_points_as_2d_gaussians(joints_2d_gt, 1, heatmap_shape)
        loss = self.criterion(heatmaps_pred, heatmaps_gt)
        return loss

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
