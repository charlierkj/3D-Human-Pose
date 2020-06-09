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
from camera_utils import *

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
    

def proj_to_2D(proj_mat, pts_3d):
    pts_3d_homo = to_homogeneous_coords(pts_3d) # 4 x n
    pts_2d_homo = proj_mat @ pts_3d_homo # 3 x n
    pts_2d = to_cartesian_coords(pts_2d_homo) # 2 x n
    return pts_2d

def to_homogeneous_coords(pts_cart):
    """conversion from cartesian to homogeneous coordinates."""
    if isinstance(pts_cart, np.ndarray):
        return np.vstack((pts_cart, np.ones((1, pts_cart.shape[1]))))
    elif torch.is_tensor(pts_cart):
        return torch.cat((pts_cart, torch.ones((1, pts_cart.shape[1])).to(pts_cart.device)), dim=0)

def to_cartesian_coords(pts_homo):
    """conversion from homogeneous to cartesian coodinates."""
    return pts_homo[:-1, :] / pts_homo[-1, :]

def make_gif(temp_folder, write_path, remove_imgs=False):
    write_folder = os.path.split(write_path)[0]
    os.makedirs(write_folder, exist_ok=True)
    with imageio.get_writer(write_path, mode='I') as writer:
        for image in sorted(glob.glob(os.path.join(temp_folder, '*.png'))):
            writer.append_data(imageio.imread(image))
            if remove_imgs:
                os.remove(image)
    writer.close()

def make_vid(temp_folder, write_path, fps=30, size=None, remove_imgs=False):
    imgs_list_sorted = sorted(glob.glob(os.path.join(temp_folder, '*.png')))
    if size is None:
        img_0 = cv2.imread(imgs_list_sorted[0])
        height, width, channels = img_0.shape
        size = (width, height)
    write_folder = os.path.split(write_path)[0]
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

    ax.scatter(jnts_2d[:, 0], jnts_2d[:, 1], c='red', s=point_size) # plot joints
    for conn in CONNECTIVITY_HUMAN36M:
        ax.plot(jnts_2d[conn, 0], jnts_2d[conn, 1], c='lime', linewidth=line_width)

def visualize_pred(images, proj_mats, joints_3d_gt, joints_3d_pred, size=5):
    """visualize pose prediction for single data sample."""
    num_views = images.shape[0]
    num_jnts = joints_3d_gt.shape[0]
    fig, axes = plt.subplots(nrows=2, ncols=num_views, figsize=(num_views * size, 2 * size))
    axes = axes.reshape(2, num_views)

    # plot images
    if torch.is_tensor(images):
        images = images.cpu().numpy()
    images = np.moveaxis(images, 1, -1) # num_views x C x H x W -> num_views x H x W x C
    images = np.clip(255 * (images * IMAGENET_STD + IMAGENET_MEAN), 0, 255).astype(np.uint8) # denormalize
    # images = images[..., (2, 1, 0)] # BGR -> RGB
    
    for view_idx in range(num_views):
        axes[0, view_idx].imshow(images[view_idx, ::])
        axes[1, view_idx].imshow(images[view_idx, ::])
        axes[1, view_idx].set_xlabel('view_%d' % (view_idx + 1), size='large')

    # plot groundtruth poses
    axes[0, 0].set_ylabel('groundtruth', size='large')
    for view_idx in range(num_views):
        joints_2d_gt = proj_to_2D(proj_mats[view_idx, ::], joints_3d_gt.T)
        draw_pose_2D(joints_2d_gt.T, axes[0, view_idx])

    # plot predicted poses
    axes[1, 0].set_ylabel('prediction', size='large')
    for view_idx in range(num_views):
        joints_2d_pred = proj_to_2D(proj_mats[view_idx, ::], joints_3d_pred.T)
        draw_pose_2D(joints_2d_pred.T, axes[1, view_idx])

    fig.tight_layout()
    fig.canvas.draw()
    fig_np = np.array(fig.canvas.renderer._renderer)
    plt.close('all')
    return fig_np

def draw_one_scene(joints_3d_pred_path, scene_folder, save_folder, cams_idx=list(range(4)), show_img=False):
    os.makedirs(save_folder, exist_ok=True)
    joints_3d_pred = np.load(joints_3d_pred_path)
    num_frames, num_joints = joints_3d_pred.shape[0], joints_3d_pred.shape[1]

    bbox = [80, 0, 560, 480] # hardcoded bbox
    for frame_idx in range(num_frames):
        ####### duplicate with dataset class, need to simplify later #######
        images = []
        proj_mats = []
        for camera_idx in cams_idx:
            camera_name = "%04d" % camera_idx
            
            # load image
            image_path = os.path.join(scene_folder, camera_name, '%06d.jpg' % frame_idx)
            assert os.path.isfile(image_path)
            image = Image.open(image_path) # RGB
            image = image.crop(bbox)
            transform = tv.transforms.Compose([
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
                ])
            image_tensor = transform(image)
            images.append(image_tensor)

            # load camera
            camera_file = os.path.join(scene_folder, 'camera_%s.json' % camera_name)
            with open(camera_file) as camera_json:
                camera = json.load(camera_json)

            cam = Camera(camera['location'][0], camera['location'][1], camera['location'][2],
                         camera['rotation'][2], camera['rotation'][0], camera['rotation'][1],
                         camera['width'], camera['height'], camera['width'] / 2)
            cam.update_after_crop(bbox)
            proj_mat = cam.get_P() # 3 x 4
            proj_mats.append(proj_mat)

        images = torch.stack(images, dim=0)
        proj_mats = np.stack(proj_mats, axis=0)
        ####### duplicate with dataset class, need to simplify later #######

        # load groundtruth
        joints_name = datasets_utils.Joints_SynData
        skeleton_path = os.path.join(scene_folder, 'skeleton_%06d.json' % frame_idx)
        joints_3d_gt = datasets_utils.load_joints(joints_name, skeleton_path)

        # draw
        vis_img = visualize_pred(images, proj_mats, joints_3d_gt, joints_3d_pred[frame_idx, ::])
        im = Image.fromarray(vis_img)
        if show_img:
            im.show()
        # img_name = "%02d.png" % (iter_idx % subj_num_batches)
        img_path = os.path.join(save_folder, "%06d.png" % frame_idx)
        im.save(img_path)

    
