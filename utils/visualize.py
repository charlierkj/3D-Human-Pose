import os, json
import numpy as np
import imageio
import glob
import matplotlib.pyplot as plt

import torch
import torchvision as tv
import cv2
from PIL import Image

import datasets.utils as datasets_utils
from utils.ue import *

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

CONNECTIVITY_HUMAN36M = [
    (0, 1),
    (1, 2),
    (2, 6),
    (5, 4),
    (4, 3),
    (3, 6),
    (6, 7),
    (7, 8),
    (8, 16),
    (9, 16),
    (8, 12),
    (11, 12),
    (10, 11),
    (8, 13),
    (13, 14),
    (14, 15)
    ]

CONNECTIVITY_23 = CONNECTIVITY_HUMAN36M + \
    [
    (0, 17),
    (5, 18),
    (10, 19),
    (10, 20),
    (19, 20),
    (15, 21),
    (15, 22),
    (21, 22)
    ]

CONNECTIVITY_Face = [
    (17, 18),
    (18, 19),
    (19, 20),
    (20, 21),
    (21, 22),
    (22, 17)
    ]
    
CONNECTIVITY_Hand = [
    (10, 17),
    (17, 18),
    (10, 19),
    (19, 20),
    (10, 21),
    (21, 22),
    (10, 23),
    (23, 24),
    (10, 25),
    (25, 26),
    (15, 27),
    (27, 28),
    (15, 29),
    (29, 30),
    (15, 31),
    (31, 32),
    (15, 33),
    (33, 34),
    (15, 35),
    (35, 36)
    ]

CONNECTIVITY_Foot = [
    (0, 17),
    (17, 18),
    (5, 20),
    (20, 19)
    ]

CONNECTIVITY_21 = CONNECTIVITY_HUMAN36M + CONNECTIVITY_Foot

CONNECTIVITY_27 = CONNECTIVITY_HUMAN36M + CONNECTIVITY_Face + \
        [tuple([j if j < 17 else j + 6 for j in conn]) for conn in CONNECTIVITY_Foot]

CONNECTIVITY_37 = CONNECTIVITY_HUMAN36M + CONNECTIVITY_Hand

CONNECTIVITY_All = CONNECTIVITY_HUMAN36M + CONNECTIVITY_Face + \
        [tuple([j if j < 17 else j + 6 for j in conn]) for conn in CONNECTIVITY_Hand] + \
        [tuple([j if j < 17 else j + 26 for j in conn]) for conn in CONNECTIVITY_Foot]

def load_connectivity(num_joints):
    connectivity = []
    if num_joints == 17:
        connectivity = CONNECTIVITY_HUMAN36M
    elif num_joints == 21:
        connectivity = CONNECTIVITY_21
    elif num_joints == 23:
        connectivity = CONNECTIVITY_23
    elif num_joints == 27:
        connectivity = CONNECTIVITY_27
    elif num_joints == 37:
        connectivity = CONNECTIVITY_37
    elif num_joints == 47:
        connectivity = CONNECTIVITY_All
    else:
        raise ValueError("invalid number of joints.")
    return connectivity

def proj_to_2D(proj_mat, pts_3d):
    pts_3d_homo = to_homogeneous_coords(pts_3d) # 4 x n
    pts_2d_homo = proj_mat @ pts_3d_homo # 3 x n
    pts_2d = to_cartesian_coords(pts_2d_homo) # 2 x n
    return pts_2d

def proj_to_2D_batch(proj_mats_batch, pts_3d_batch):
    """
    proj_mats_batch: batch_size x num_views x 3 x 4.
    pts_3d_batch: batch_size x num_joints x 3.
    """
    pts_3d_homo_batch = to_homogeneous_coords_batch(pts_3d_batch) # batch_size x num_joints x 4
    if isinstance(pts_3d_homo_batch, np.ndarray) and isinstance(proj_mats_batch, np.ndarray):
        pts_3d_homo_batch = np.expand_dims(pts_3d_homo_batch, axis=1) # batch_size x 1 x num_joints x 4
        pts_2d_homo_batch = pts_3d_homo_batch @ np.swapaxes(proj_mats_batch, 2, 3) # batch_size x num_views x num_joints x 3
    elif torch.is_tensor(pts_3d_homo_batch) and torch.is_tensor(proj_mats_batch):
        pts_3d_homo_batch = pts_3d_homo_batch.unsqueeze(1) # batch_size x 1 x num_joints x 4
        pts_2d_homo_batch = pts_3d_homo_batch @ proj_mats_batch.transpose(2, 3) # batch_size x num_views x num_joints x 3
    pts_2d_batch = to_cartesian_coords_batch(pts_2d_homo_batch) # batch_size x num_views x num_joints x 2
    return pts_2d_batch

def proj_to_camspace(ext_mat, pts_3d):
    """project to 3D camera space"""
    pts_3d_homo = to_homogeneous_coords(pts_3d) # 4 x n
    pts_3d_camspace = ext_mat @ pts_3d_homo # 3 x n
    return pts_3d_camspace

def to_homogeneous_coords(pts_cart):
    """conversion from cartesian to homogeneous coordinates."""
    if isinstance(pts_cart, np.ndarray):
        return np.vstack((pts_cart, np.ones((1, pts_cart.shape[1]))))
    elif torch.is_tensor(pts_cart):
        return torch.cat((pts_cart, torch.ones((1, pts_cart.shape[1])).type(pts_cart.dtype).to(pts_cart.device)), dim=0)

def to_homogeneous_coords_batch(pts_cart_batch):
    """conversion from cartesian to homogeneous coordinates.
    pts_cart_batch: batch_size x num_joints x 3, or, batch_size x num_views x num_joints x 2.
    output: batch_size x num_joints x 4, or, batch_size x num_views x num_joints x 3.
    """
    if isinstance(pts_cart_batch, np.ndarray):
        return np.concatenate((pts_cart_batch, np.ones((*pts_cart_batch.shape[0:-1], 1))), axis=-1)
    elif torch.is_tensor(pts_cart):
        return torch.cat((pts_cart_batch, \
                          torch.ones((*pts_cart_batch.shape[0:-1], 1)).type(pts_cart_batch.dtype).to(pts_cart_batch.device)),\
                         dim=-1)

def to_cartesian_coords(pts_homo):
    """conversion from homogeneous to cartesian coodinates."""
    return pts_homo[:-1, :] / pts_homo[-1, :]

def to_cartesian_coords_batch(pts_homo_batch):
    """conversion from homogeneous to cartesian coodinates.
    pts_homo_batch: batch_size x num_joints x 4, or, batch_size x num_views x num_joints x 3.
    output: batch_size x num_joints x 3, or, batch_size x num_views x num_joints x 2.
    """
    if isinstance(pts_homo_batch, np.ndarray):
        return pts_homo_batch[..., :-1] / np.expand_dims(pts_homo_batch[..., -1], axis=-1)
    elif torch.is_tensor(pts_homo_batch):
        return pts_homo_batch[..., :-1] / pts_homo_batch[..., -1].unsqueeze(-1)


def make_gif(temp_folder, write_path, remove_imgs=False):
    write_folder = os.path.split(write_path)[0]
    if write_folder != '':
        os.makedirs(write_folder, exist_ok=True)
    with imageio.get_writer(write_path, mode='I') as writer:
        for image in sorted(glob.glob(os.path.join(temp_folder, '*.png'))):
            writer.append_data(imageio.imread(image))
            if remove_imgs:
                os.remove(image)
    writer.close()

def make_vid(temp_folder, write_path, img_format='png', fps=30, size=None, remove_imgs=False):
    imgs_list_sorted = sorted(glob.glob(os.path.join(temp_folder, '*.%s' % img_format)))
    if size is None:
        img_0 = cv2.imread(imgs_list_sorted[0])
        height, width, channels = img_0.shape
        size = (width, height)
    write_folder = os.path.split(write_path)[0]
    if write_folder != '':
        os.makedirs(write_folder, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vid_writer = cv2.VideoWriter(write_path, fourcc, fps, size)
    for image in imgs_list_sorted:
        vid_writer.write(cv2.imread(image))
        if remove_imgs:
            os.remove(image)
    vid_writer.release()

def draw_pose_2D(jnts_2d, ax, point_size=2, line_width=1):
    if torch.is_tensor(jnts_2d):
        jnts_2d = jnts_2d.cpu().detach() # n x 2

    num_jnts = jnts_2d.shape[0]
    connectivity = load_connectivity(num_jnts)

    ax.scatter(jnts_2d[:, 0], jnts_2d[:, 1], c='red', s=point_size) # plot joints
    for conn in CONNECTIVITY_HUMAN36M:
        ax.plot(jnts_2d[conn, 0], jnts_2d[conn, 1], c='lime', linewidth=line_width)
    if num_jnts > 17:
        for conn in connectivity[16:]:
            ax.plot(jnts_2d[conn, 0], jnts_2d[conn, 1], c='cyan', linewidth=line_width)

def visualize_pred(images, proj_mats, joints_3d_gt, joints_3d_pred, joints_2d_pred, size=5):
    """visualize pose prediction for single data sample."""
    num_views = images.shape[0]
    num_jnts = joints_3d_gt.shape[0]
    fig, axes = plt.subplots(nrows=3, ncols=num_views, figsize=(num_views * size, 3 * size))
    axes = axes.reshape(3, num_views)

    # plot images
    if torch.is_tensor(images):
        images = images.cpu().numpy()
    images = np.moveaxis(images, 1, -1) # num_views x C x H x W -> num_views x H x W x C
    images = images[..., (2, 1, 0)] # BGR -> RGB
    images = np.clip(255 * (images * IMAGENET_STD + IMAGENET_MEAN), 0, 255).astype(np.uint8) # denormalize
    #images = images[..., (2, 1, 0)] # BGR -> RGB
    
    for view_idx in range(num_views):
        axes[0, view_idx].imshow(images[view_idx, ::])
        axes[1, view_idx].imshow(images[view_idx, ::])
        axes[2, view_idx].imshow(images[view_idx, ::])
        axes[2, view_idx].set_xlabel('view_%d' % (view_idx + 1), size='large')

    # plot groundtruth poses
    axes[0, 0].set_ylabel('groundtruth', size='large')
    for view_idx in range(num_views):
        joints_2d_gt = proj_to_2D(proj_mats[view_idx, ::], joints_3d_gt.T)
        draw_pose_2D(joints_2d_gt.T, axes[0, view_idx])

    # plot projection of predicted 3D poses
    axes[1, 0].set_ylabel('3D prediction', size='large')
    for view_idx in range(num_views):
        joints_3d_pred_proj = proj_to_2D(proj_mats[view_idx, ::], joints_3d_pred.T)
        draw_pose_2D(joints_3d_pred_proj.T, axes[1, view_idx])

    # plot predicted 2D poses
    axes[2, 0].set_ylabel('2D prediction', size='large')
    for view_idx in range(num_views):
        draw_pose_2D(joints_2d_pred[view_idx, :, :], axes[2, view_idx])

    fig.tight_layout()
    fig.canvas.draw()
    fig_np = np.array(fig.canvas.renderer._renderer)
    plt.close('all')
    return fig_np

def draw_one_scene(joints_3d_pred_path, joints_2d_pred_path, scene_folder, save_folder, \
                   cams_idx=list(range(4)), \
                   bbox=[160, 0, 1120, 960], image_shape=[384, 384], \
                   show_img=False):
    os.makedirs(save_folder, exist_ok=True)
    joints_3d_pred = np.load(joints_3d_pred_path)
    joints_2d_pred = np.load(joints_2d_pred_path)
    num_frames, num_joints = joints_3d_pred.shape[0], joints_3d_pred.shape[1]

    #bbox = [80, 0, 560, 480] # hardcoded bbox
    for frame_idx in range(num_frames):
        images = []
        proj_mats = []
        for camera_idx in cams_idx:
            camera_name = "%04d" % camera_idx
            
            # load image
            image_path = os.path.join(scene_folder, camera_name, '%06d.jpg' % frame_idx)
            image_tensor = datasets_utils.load_image(image_path, bbox, image_shape)
            images.append(image_tensor)

            # load camera
            camera_file = os.path.join(scene_folder, 'camera_%s.json' % camera_name)
            cam = datasets_utils.load_camera(camera_file)
            cam.update_after_crop(bbox)
            cam.update_after_resize(image_shape)
            proj_mat = cam.get_P() # 3 x 4
            proj_mats.append(proj_mat)

        images = torch.stack(images, dim=0)
        proj_mats = np.stack(proj_mats, axis=0)

        # load groundtruth
        joints_name = datasets_utils.get_joints_name(num_joints)
        skeleton_path = os.path.join(scene_folder, 'skeleton_%06d.json' % frame_idx)
        joints_3d_gt = datasets_utils.load_joints(joints_name, skeleton_path)

        # draw
        vis_img = visualize_pred(images, proj_mats, joints_3d_gt, joints_3d_pred[frame_idx, ::], joints_2d_pred[frame_idx, ::])
        im = Image.fromarray(vis_img)
        if show_img:
            im.show()
        # img_name = "%02d.png" % (iter_idx % subj_num_batches)
        img_path = os.path.join(save_folder, "%06d.png" % frame_idx)
        im.save(img_path)

    
