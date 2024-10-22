import numpy as np
import torch
import matplotlib.pyplot as plt

from datasets.human36m import Human36MMultiViewDataset
from datasets.mpii import Mpii
import datasets.utils as datasets_utils
from models.loss import KeypointsL2Loss
from models.metric import PCK, PCKh

import consistency

from utils.multiview import triangulate_point_from_multiple_views_linear_torch
from utils.visualize import proj_to_2D_batch


def evaluate_one_scene(joints_3d_pred_path, scene_folder, invalid_joints=(9, 16), path=True):
    """joints_3d_pred_path: the path where npy file is stored.
    scene_folder: the folder under which the input images and groundtruth jsons are stored.
    path: indicate whether 'joints_3d_pred_path' is a path or not - True(is path), False(not path, is array)."""
    if path:
        joints_3d_pred = np.load(joints_3d_pred_path)
    else:
        joints_3d_pred = joints_3d_pred_path
    num_frames, num_joints = joints_3d_pred.shape[0], joints_3d_pred.shape[1]
    joints_3d_valid = np.ones(shape=(num_frames, num_joints, 1))
    joints_3d_valid[:, invalid_joints, :] = 0
    joints_3d_gt = np.empty(shape=(num_frames, num_joints, 3))
    for frame_idx in range(num_frames):
        joints_name = datasets_utils.get_joints_name(num_joints)
        skeleton_path = os.path.join(scene_folder, 'skeleton_%06d.json' % frame_idx)
        joints_3d_gt[frame_idx, :, :] = datasets_utils.load_joints(joints_name, skeleton_path)
    error_frames = evaluate_one_batch(joints_3d_pred, joints_3d_gt, joints_3d_valid) # size of num_frames
    error_mean = float(error_frames.mean())
    return error_mean    
    
def evaluate_one_batch(joints_3d_pred, joints_3d_gt_batch, joints_3d_valid_batch):
    """MPJPE (mean per joint position error) in mm"""
    if isinstance(joints_3d_pred, np.ndarray):
        error_batch = np.sqrt(((joints_3d_pred - joints_3d_gt_batch) ** 2).sum(2))
        error_batch = joints_3d_valid_batch * np.expand_dims(error_batch, axis=2)
        error_batch = error_batch.mean((1, 2))
    elif torch.is_tensor(joints_3d_pred):
        error_batch = torch.sqrt(((joints_3d_pred - joints_3d_gt_batch) ** 2).sum(2)) # batch_size x num_joints
        error_batch = joints_3d_valid_batch * error_batch.unsqueeze(2) # batch_size x num_joints x 1
        error_batch = error_batch.mean((1, 2)) # mean error per sample in batch
    return error_batch


def eval_one_batch(metric, joints_3d_pred, joints_2d_pred, \
                   proj_mats_batch, joints_3d_gt_batch, joints_3d_valid_batch, joints_2d_gt_batch):
    error = 0
    detected = 0
    if isinstance(metric, PCK):
        #detected, num_samples = metric(joints_2d_pred, proj_mats_batch, \
        #                               joints_3d_gt_batch, joints_3d_valid_batch)
        joints_2d_valid_batch = torch.ones(*joints_2d_pred.shape[0:3], 1).type(torch.bool).to(joints_2d_pred.device)
        detected, num_samples, detected_per_joint, num_per_joint = metric(joints_2d_pred, joints_2d_gt_batch, joints_2d_valid_batch)
    elif isinstance(metric, KeypointsL2Loss):
        error = metric(joints_3d_pred, joints_3d_gt_batch, joints_3d_valid_batch).item()
        num_samples = joints_3d_pred.shape[0]
    return detected, error, num_samples, detected_per_joint, num_per_joint # either total_joints, or total_frames


def eval_pseudo_labels(dataset='human36m', p=0.2, separate=True, triangulate=False): # hardcoded

    if dataset == 'mpii' and triangulate:
        raise ValueError("MPII dataset is not multiview, please use multivew dataset.")
    
    print("Loading data ...")
    if dataset == 'human36m':
        train_set = dataset = Human36MMultiViewDataset(
            h36m_root="../learnable-triangulation-pytorch/data/human36m/processed/",
            train=True,
            image_shape=[384, 384],
            labels_path="../learnable-triangulation-pytorch/data/human36m/extra/human36m-multiview-labels-GTbboxes.npy",
            with_damaged_actions=True,
            retain_every_n_frames=10,
            scale_bbox=1.6,
            kind="human36m",
            undistort_images=True,
            ignore_cameras=[],
            crop=True,
        )
        train_loader = datasets_utils.human36m_loader(train_set, \
                                                      # batch_size=64, \
                                                      batch_size=4, \
                                                      shuffle=False, \
                                                      num_workers=4)
        pseudo_labels = np.load("pseudo_labels/human36m_train_every_10_frames.npy", allow_pickle=True).item()
        thresh = 1
        
    elif dataset == 'mpii':
        train_set = Mpii(
            image_path="../mpii_images",
            anno_path="../pytorch-pose/data/mpii/mpii_annotations.json",
            inp_res=384,
            out_res=96,
            is_train=True
        )
        train_loader = datasets_utils.mpii_loader(train_set, \
                                                  batch_size=256, \
                                                  shuffle=False, \
                                                  num_workers=4)
        pseudo_labels = np.load("pseudo_labels/mpii_train.npy", allow_pickle=True).item()
        thresh = 0.5
        
    print("Data loaded.")    
    score_thresh = consistency.get_score_thresh(pseudo_labels, p, separate=separate)

    total_joints = 0
    total_detected_pck = 0
    total_detected_pckh = 0
    errors = torch.empty(0)
    for iter_idx, (images_batch, proj_mats_batch, joints_3d_gt_batch, joints_3d_valid_batch, joints_2d_gt_batch, indexes) in enumerate(train_loader):
        print(iter_idx)
        if images_batch is None:
            continue

        joints_2d_pseudo, joints_2d_valid_batch = \
                          consistency.get_pseudo_labels(pseudo_labels, indexes, images_batch.shape[1], score_thresh)

        if triangulate:
            num_valid_before = joints_2d_valid_batch.sum()
            joints_2d_pseudo, joints_2d_valid_batch = triangulate_pseudo_labels(proj_mats_batch, joints_2d_pseudo, joints_2d_valid_batch)
            num_valid_after = joints_2d_valid_batch.sum()
            print("Number of valid labels: before: %d, after: %d" % (num_valid_before, num_valid_after))

        detected_pck, num_jnts, _, _ = PCK()(joints_2d_pseudo, joints_2d_gt_batch, joints_2d_valid_batch)
        detected_pckh, _, _, _ = PCKh(thresh=thresh)(joints_2d_pseudo, joints_2d_gt_batch, joints_2d_valid_batch)
        diff = torch.sqrt(torch.sum((joints_2d_pseudo - joints_2d_gt_batch)**2, dim=-1, keepdims=True))
        error_2d = diff[joints_2d_valid_batch]
        errors = torch.cat((errors, error_2d))

        total_joints += num_jnts
        total_detected_pck += detected_pck
        total_detected_pckh += detected_pckh

    errors = errors.cpu().numpy()
    print("PCK:", total_detected_pck / total_joints)
    print("PCKh:", total_detected_pckh / total_joints)
    print("Error(2D):", errors.mean())

    # plot histogram
    plt.hist(errors, bins=100, density=True)
    plt.title("2D error distribution in pseudo labels")
    plt.xlabel("2D error (in pixel)")
    plt.ylabel("density")
    plt.savefig("figs/errors_pseudo_labels_%s.png" % dataset)
    plt.close()


def triangulate_pseudo_labels(proj_mats_batch, points_batch, points_valid_batch, confidences_batch=None):
    batch_size = points_batch.shape[0]
    num_joints = points_batch.shape[2]
    points_batch_tr = points_batch.clone()
    points_valid_batch_tr = points_valid_batch.clone()
    for batch_i in range(batch_size):
        proj_mats = proj_mats_batch[batch_i] # num_views x 3 x 4
        points = points_batch[batch_i] # num_views x num_joints x 2
        points_valid = points_valid_batch[batch_i] # num_views x num_joints x 1
        points_valid = points_valid.squeeze(-1) # num_views x num_joints
        num_valid = points_valid.sum(0) # num_joints
        for j in range(num_joints):
            if num_valid[j] <= 1:
                continue
            views_valid = points_valid[:, j]
            points_3d = triangulate_point_from_multiple_views_linear_torch(proj_mats[views_valid, :, :], \
                                                                           points[views_valid, j, :])
            points_3d = points_3d.view(1, 1, 3) # 1 x 1 x 3
            points_tr = proj_to_2D_batch(proj_mats[views_valid, :, :].unsqueeze(0), points_3d)
            points_tr = points_tr.view(-1, 2) # num_views x 2
            points_batch_tr[batch_i, views_valid, j, :] = points_tr
            points_valid_batch_tr[batch_i, :, j, :] = True

    return points_batch_tr, points_valid_batch_tr   

