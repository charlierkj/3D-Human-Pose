import numpy as np
import torch

def estimate_campose(keypoints_2d_l, keypoints_2d_r, \
                     K_l, K_r, \
                     confidences_l, confidences_r):
    """
    keypoints_2d: 2D joints estimation in two cameras. (size n x 2)
    K: intrinsic matrix (size 3 x 3)
    confidences: size n x 1
    """
    kpts_2d_l_homo = torch.cat([keypoints_2d_l, torch.ones((keypoints_2d_l.shape[0], 1), dtype=keypoints_2d_l.dtype, device=keypoints_2d_l.device)], dim=1)
    kpts_2d_r_homo = torch.cat([keypoints_2d_r, torch.ones((keypoints_2d_r.shape[0], 1), dtype=keypoints_2d_r.dtype, device=keypoints_2d_r.device)], dim=1)
    A = kpts_2d_l_homo.view(-1, 1).repeat(1, 3).view(-1, 9) * kpts_2d_r_homo.repeat(1, 3)
    A *= confidences_l.view(-1, 1) * confidences_r.view(-1, 1)

    u, s, vh = torch.svd(A.view(-1, 9))

    f = -vh[:, -1]
    F = f.view(3, 3) # fundamental matrix
    E = K_l.T @ F @ K_r # essential matrix
    
    U, S, Vh = torch.svd(E.view(3, 3))

    R = U @ torch.Tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]]).type(torch.float64) @ Vh.T # rotation matrix (3 x 3)
    t = U[:, -1] # translation (3,)
    return R, t
    
