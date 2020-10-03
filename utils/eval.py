import numpy as np
import torch

from datasets.human36m import Human36MMultiViewDataset
import datasets.utils as datasets_utils
from models.loss import KeypointsL2Loss
from models.metric import PCK, PCKh

import consistency


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
        detected, num_samples = metric(joints_2d_pred, joints_2d_gt_batch, joints_2d_valid_batch)
    elif isinstance(metric, KeypointsL2Loss):
        error = metric(joints_3d_pred, joints_3d_gt_batch, joints_3d_valid_batch).item()
        num_samples = joints_3d_pred.shape[0]
    return detected, error, num_samples # either total_joints, or total_frames


def eval_pseudo_labels(): # hardcoded
    h36m_train_set = dataset = Human36MMultiViewDataset(
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
    h36m_train_loader = datasets_utils.human36m_loader(h36m_train_set, \
                                                       batch_size=64, \
                                                       shuffle=False, \
                                                       num_workers=4)

    pseudo_labels = np.load("pseudo_labels/human36m_train.npy", allow_pickle=True).item()
    p = 0.2 # percentage
    score_thresh = consistency.get_score_thresh(pseudo_labels, p)

    total_joints = 0
    total_detected_pck = 0
    total_detected_pckh = 0
    for iter_idx, (images_batch, proj_mats_batch, joints_3d_gt_batch, joints_3d_valid_batch, joints_2d_gt_batch, indexes) in enumerate(h36m_train_loader):
        print(iter_idx)
        if images_batch is None:
            continue

        joints_2d_pseudo, joints_2d_valid_batch = \
                          consistency.get_pseudo_labels(pseudo_labels, indexes, images_batch.shape[1], score_thresh)

        detected_pck, num_jnts = PCK()(joints_2d_pseudo, joints_2d_gt_batch, joints_2d_valid_batch)

        detected_pckh, _ = PCKh(thresh=1)(joints_2d_pseudo, joints_2d_gt_batch, joints_2d_valid_batch)

        total_joints += num_jnts
        total_detected_pck += detected_pck
        total_detected_pckh += detected_pckh

    print("PCK:", total_detected_pck / total_joints)
    print("PCKh:", total_detected_pckh / total_joints)

