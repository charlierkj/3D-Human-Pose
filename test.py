import os, json
import numpy as np
import torch
from PIL import Image

from mvn.utils import cfg
from mvn.models_temp.triangulation import AlgebraicTriangulationNet
from datasets.multiview_syndata import MultiView_SynData
import datasets.utils as datasets_utils

import visualize


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
        joints_name = datasets_utils.Joints_SynData
        skeleton_path = os.path.join(scene_folder, 'skeleton_%06d.json' % frame_idx)
        joints_3d_gt[frame_idx, :, :] = datasets_utils.load_joints(joints_name, skeleton_path)
    error_frames = evaluate_one_batch(joints_3d_pred, joints_3d_gt, joints_3d_valid) # size of num_frames
    error_mean = float(error_frames.mean())
    return error_mean    
    
def evaluate_one_batch(joints_3d_pred, joints_3d_gt_batch, joints_3d_valid_batch):
    """MPJPE (mean per joint position error) in cm"""
    if isinstance(joints_3d_pred, np.ndarray):
        error_batch = np.sqrt(((joints_3d_pred - joints_3d_gt_batch) ** 2).sum(2))
        error_batch = joints_3d_valid_batch * np.expand_dims(error_batch, axis=2)
        error_batch = error_batch.mean((1, 2))
    elif torch.is_tensor(joints_3d_pred):
        error_batch = torch.sqrt(((joints_3d_pred - joints_3d_gt_batch) ** 2).sum(2)) # batch_size x num_joints
        error_batch = joints_3d_valid_batch * error_batch.unsqueeze(2) # batch_size x num_joints x 1
        error_batch = error_batch.mean((1, 2)) # mean error per sample in batch
    return error_batch


def multiview_test(model, dataloader, device, save_folder, show_img=False, make_gif=False, make_vid=False):
    frames_per_scene = 60
    
    model.to(device)
    model.eval()

    #scene_names = []
    subj_names = []
    anim_names = []
    metrics = {}
    with torch.no_grad():
        for iter_idx, (images_batch, proj_mats_batch, joints_3d_gt_batch, joints_3d_valid_batch, info_batch) in enumerate(dataloader):
            if images_batch is None:
                continue

            images_batch = images_batch.to(device)
            proj_mats_batch = proj_mats_batch.to(device)
            joints_3d_gt_batch = joints_3d_gt_batch.to(device)
            joints_3d_valid_batch = joints_3d_valid_batch.to(device)

            batch_size = images_batch.shape[0]
            
            joints_3d_pred, joints_2d_pred, heatmaps_pred, confidences_pred = model(images_batch, proj_mats_batch)

            [subj_idx, anim_idx, frame] = info_batch[0] 
            if frame == 0:
                subj_name = 'S%d' % subj_idx
                anim_name = 'anim_%03d' % anim_idx
                print(subj_name, anim_name)
                if subj_name not in subj_names:
                    subj_names.append(subj_name)
                if anim_name not in anim_names:
                    anim_names.append(anim_name)
                
                joints_3d_pred_np = np.empty([0,] + list(joints_3d_pred.detach().cpu().numpy().shape[1::]))
                joints_2d_pred_np = np.empty([0,] + list(joints_2d_pred.detach().cpu().numpy().shape[1::]))
                heatmaps_pred_np = np.empty([0,] + list(heatmaps_pred.detach().cpu().numpy().shape[1::]))
                confidences_pred_np = np.empty([0,] + list(confidences_pred.detach().cpu().numpy().shape[1::]))

                preds_folder = os.path.join(save_folder, 'preds', 'S%d' % subj_idx, 'anim_%03d' % anim_idx)
                os.makedirs(preds_folder, exist_ok=True)

            joints_3d_pred_np = np.concatenate((joints_3d_pred_np, joints_3d_pred.detach().cpu().numpy()), axis=0)
            joints_2d_pred_np = np.concatenate((joints_2d_pred_np, joints_2d_pred.detach().cpu().numpy()), axis=0)
            heatmaps_pred_np = np.concatenate((heatmaps_pred_np, heatmaps_pred.detach().cpu().numpy()), axis=0)
            confidences_pred_np = np.concatenate((confidences_pred_np, confidences_pred.detach().cpu().numpy()), axis=0)

            if (frame + batch_size) == frames_per_scene:
                # save intermediate results
                print('saving intermediate results...')
                np.save(os.path.join(preds_folder, 'joints_3d.npy'), joints_3d_pred_np) # numpy array of size (num_frames, num_joints, 3)
                np.save(os.path.join(preds_folder, 'joints_2d.npy'), joints_2d_pred_np) # numpy array of size (num_frames, num_views, num_joints, 2)
                #np.save(os.path.join(preds_folder, 'heatmaps.npy'), heatmaps_pred_np) # numpy array of size (num_frames, num_views, num_joints, 120, 120)
                np.save(os.path.join(preds_folder, 'confidences.npy'), confidences_pred_np) # numpy array of size (num_frames, num_views, num_joints)

    # save evaluations and visualizations
    for subj_name in subj_names:
        metrics_subj = {}
        for anim_name in anim_names:
            scene_folder = os.path.join(dataloader.dataset.basepath, subj_name, anim_name) # subj currently hardcoded
            # evaluate
            joints_3d_pred_path = os.path.join(save_folder, 'preds', subj_name, anim_name, 'joints_3d.npy')
            error_per_scene = evaluate_one_scene(joints_3d_pred_path, scene_folder, invalid_joints=(9, 16))
            metrics_subj[anim_name] = error_per_scene

            # save images
            print('saving result images...')
            imgs_folder = os.path.join(save_folder, 'imgs', subj_name, anim_name)
            visualize.draw_one_scene(joints_3d_pred_path, scene_folder, imgs_folder, show_img=show_img)

            # save gifs/videos (optioanl)
            if make_gif:
                print('saving result gifs...')
                gif_name = os.path.join(save_folder, 'gifs', subj_name, '%s.gif' % anim_name)
                visualize.make_gif(imgs_folder, gif_name)

            if make_vid:
                print('saving result videos...')
                vid_name = os.path.join(save_folder, 'vids', subj_name, '%s.mp4' % anim_name)
                visualize.make_vid(imgs_folder, vid_name)

        metrics[subj_name] = metrics_subj
            
    # save evaluation
    print('saving evaluation results...')
    metrics_path = os.path.join(save_folder, 'metrics.json')
    with open(metrics_path, 'w') as metrics_json:
        json.dump(metrics, metrics_json)

    
if __name__ == "__main__":

    device = torch.device(0)
    
    config = cfg.load_config('experiments/syn_data/multiview_data_2_alg.yaml')

    model = AlgebraicTriangulationNet(config, device=device).to(device)

    state_dict = torch.load(config.model.checkpoint)
    for key in list(state_dict.keys()):
        new_key = key.replace("module.", "")
        state_dict[new_key] = state_dict.pop(key)

    model.load_state_dict(state_dict, strict=True)

    print("Loading data..")
    data_path = 'data/test_03_temp/multiview_data'
    dataset = MultiView_SynData(data_path, invalid_joints=(9, 16), bbox=[80, 0, 560, 480], ori_form=1)
    dataloader = datasets_utils.syndata_loader(dataset, batch_size=4)

    save_folder = os.path.join(os.getcwd(), 'results/test_03_temp')
    multiview_test(model, dataloader, device, save_folder, make_vid=True)
