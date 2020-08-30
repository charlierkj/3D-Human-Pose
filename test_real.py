import os, json
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from mvn.utils import cfg
from mvn.models_temp.triangulation import AlgebraicTriangulationNet
from datasets.multiview_syndata import MultiView_SynData
import datasets.utils as datasets_utils

import utils.visualize as visualize


def visualize_pred(images, proj_mats, joints_3d_pred, joints_2d_pred, size=5):
    """visualize pose prediction for single data sample."""
    num_views = images.shape[0]
    num_jnts = joints_3d_pred.shape[0]
    fig, axes = plt.subplots(nrows=2, ncols=num_views, figsize=(num_views * size, 2 * size))
    axes = axes.reshape(2, num_views)

    # plot images
    if torch.is_tensor(images):
        images = images.cpu().numpy()
    images = np.moveaxis(images, 1, -1) # num_views x C x H x W -> num_views x H x W x C
    images = np.clip(255 * (images * visualize.IMAGENET_STD + visualize.IMAGENET_MEAN), 0, 255).astype(np.uint8) # denormalize
    # images = images[..., (2, 1, 0)] # BGR -> RGB
    
    for view_idx in range(num_views):
        axes[0, view_idx].imshow(images[view_idx, ::])
        axes[1, view_idx].imshow(images[view_idx, ::])
        axes[1, view_idx].set_xlabel('view_%d' % (view_idx + 1), size='large')

    # plot projection of predicted 3D poses
    axes[0, 0].set_ylabel('3D prediction', size='large')
    for view_idx in range(num_views):
        joints_3d_pred_proj = visualize.proj_to_2D(proj_mats[view_idx, ::], joints_3d_pred.T)
        visualize.draw_pose_2D(joints_3d_pred_proj.T, axes[0, view_idx])

    # plot predicted 2D poses
    axes[1, 0].set_ylabel('2D prediction', size='large')
    for view_idx in range(num_views):
        visualize.draw_pose_2D(joints_2d_pred[view_idx, :, :], axes[1, view_idx])

    fig.tight_layout()
    fig.canvas.draw()
    fig_np = np.array(fig.canvas.renderer._renderer)
    plt.close('all')
    return fig_np


def update_K(K, bbox, size):
    left, upper, right, lower = bbox
    cx, cy = K[0, 2], K[1, 2]
    new_cx = cx - left
    new_cy = cy - upper
    K[0, 2], K[1, 2] = new_cx, new_cy

    w, h = right - left, lower - upper
    new_w, new_h = size
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    new_fx = fx * (new_w / w)
    new_fy = fy * (new_h / h)
    new_cx = cx * (new_w / w)
    new_cy = cy * (new_h / h)
    K[0, 0], K[1, 1], K[0, 2], K[1, 2] = new_fx, new_fy, new_cx, new_cy
    return K
    

if __name__ == "__main__":
    bbox = [420, 0, 1500, 1080]
    size=(480, 480)

    folder = os.path.join("data", "real", "single_human")
    output_folder = os.path.join(folder, "results")

    device = torch.device(0)
    
    config = cfg.load_config('experiments/syn_data/multiview_data_alg_test_17jnts.yaml')

    model = AlgebraicTriangulationNet(config, device=device).to(device)

    state_dict = torch.load(config.model.checkpoint)
    for key in list(state_dict.keys()):
        new_key = key.replace("module.", "")
        state_dict[new_key] = state_dict.pop(key)

    model.load_state_dict(state_dict, strict=True)

    # load camera
    proj_mats = torch.empty((1, 2, 3, 4))
    with open(os.path.join(folder, "cam0.json"), 'r') as json_file:
        cam0 = json.load(json_file)
    with open(os.path.join(folder, "cam1.json"), 'r') as json_file:
        cam1 = json.load(json_file)
    K0 = update_K(torch.Tensor(cam0["int"]), bbox, size)
    K1 = update_K(torch.Tensor(cam1["int"]), bbox, size)
    P0 = K0 @ torch.Tensor(cam0["ext"])
    P1 = K1 @ torch.Tensor(cam1["ext"])
    proj_mats[0, 0, :, :] = P0
    proj_mats[0, 1, :, :] = P1

    model.eval()
    with torch.no_grad():
        joints_3d_pred_np = np.empty(shape=(0, 17, 3))
        for frame_idx in range(480, 1001): # hardcoded
            # load image
            view0_path = os.path.join(folder, "cam0", "cam0_%06d.jpg" % frame_idx)
            view0_tensor = datasets_utils.load_image(view0_path, bbox=bbox, size=size)
            view1_path = os.path.join(folder, "cam1", "cam1_%06d.jpg" % frame_idx)
            view1_tensor = datasets_utils.load_image(view1_path, bbox=bbox, size=size)
            images = torch.stack([view0_tensor, view1_tensor], dim=0)
            images_batch = images.unsqueeze(0)

            images_batch, proj_mats = images_batch.to(device), proj_mats.to(device)
            joints_3d_pred, joints_2d_pred, heatmaps_pred, confidences_pred = model(images_batch, proj_mats)

            joints_3d_pred_np = np.vstack((joints_3d_pred_np, joints_3d_pred.detach().cpu().numpy()))

            vis_img = visualize_pred(images_batch[0], proj_mats[0], joints_3d_pred[0], joints_2d_pred[0], size=5)
            im = Image.fromarray(vis_img)
            img_path = os.path.join(output_folder, "%06d.png" % frame_idx)
            # im.save(img_path)
        
    np.save(os.path.join(output_folder, 'joints_3d.npy'), joints_3d_pred_np)

    
