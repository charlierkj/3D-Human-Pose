# Reference: https://github.com/karfly/learnable-triangulation-pytorch

from copy import deepcopy
import numpy as np
import pickle
import random

from scipy.optimize import least_squares

import torch
from torch import nn

from utils import op, multiview

from models import pose_resnet


class AlgebraicTriangulationNet(nn.Module):
    def __init__(self, config, device='cuda:0'):
        super().__init__()

        self.use_confidences = config.model.use_confidences

        config.model.backbone.alg_confidences = False
        config.model.backbone.vol_confidences = False

        if self.use_confidences:
            config.model.backbone.alg_confidences = True

        self.backbone = pose_resnet.get_pose_net(config.model.backbone, device=device)

        self.heatmap_softmax = config.model.heatmap_softmax
        self.heatmap_multiplier = config.model.heatmap_multiplier


    def forward(self, images, proj_matricies):
        device = images.device
        batch_size, n_views = images.shape[:2]

        # reshape n_views dimension to batch dimension
        images = images.view(-1, *images.shape[2:])

        # forward backbone and integral
        if self.use_confidences:
            heatmaps, _, alg_confidences, _ = self.backbone(images)
        else:
            heatmaps, _, _, _ = self.backbone(images)
            alg_confidences = torch.ones(batch_size * n_views, heatmaps.shape[1]).type(torch.float).to(device)

        heatmaps_before_softmax = heatmaps.view(batch_size, n_views, *heatmaps.shape[1:])
        keypoints_2d, heatmaps = op.integrate_tensor_2d(heatmaps * self.heatmap_multiplier, self.heatmap_softmax)

        # reshape back
        images = images.view(batch_size, n_views, *images.shape[1:])
        heatmaps = heatmaps.view(batch_size, n_views, *heatmaps.shape[1:])
        keypoints_2d = keypoints_2d.view(batch_size, n_views, *keypoints_2d.shape[1:])
        alg_confidences = alg_confidences.view(batch_size, n_views, *alg_confidences.shape[1:])

        # norm confidences
        alg_confidences = alg_confidences / alg_confidences.sum(dim=1, keepdim=True)
        alg_confidences = alg_confidences + 1e-5  # for numerical stability

        # calcualte shapes
        image_shape = tuple(images.shape[3:])
        batch_size, n_views, n_joints, heatmap_shape = heatmaps.shape[0], heatmaps.shape[1], heatmaps.shape[2], tuple(heatmaps.shape[3:])

        # upscale keypoints_2d, because image shape != heatmap shape
        keypoints_2d_transformed = torch.zeros_like(keypoints_2d)
        keypoints_2d_transformed[:, :, :, 0] = keypoints_2d[:, :, :, 0] * (image_shape[1] / heatmap_shape[1])
        keypoints_2d_transformed[:, :, :, 1] = keypoints_2d[:, :, :, 1] * (image_shape[0] / heatmap_shape[0])
        keypoints_2d = keypoints_2d_transformed

        # if not projection matrices are provided, only infer in 2D
        if proj_matricies is None:
            keypoints_3d = None
        else:
            # triangulate
            try:
                keypoints_3d = multiview.triangulate_batch_of_points(
                    proj_matricies, keypoints_2d,
                    confidences_batch=alg_confidences
                )
            except RuntimeError as e:
                print("Error: ", e)

                print("confidences =", confidences_batch_pred)
                print("proj_matricies = ", proj_matricies)
                print("keypoints_2d_batch_pred =", keypoints_2d_batch_pred)
                exit()

        return keypoints_3d, keypoints_2d, heatmaps, alg_confidences

