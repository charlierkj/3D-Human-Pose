import os, json
import numpy as np
import matplotlib.pyplot as plt

from test import evaluate_one_scene, evaluate_one_batch
import datasets.utils as datasets_utils
import visualize


def plot_3D_error_per_subject(dataset_folder, result_folder, write_path):
    preds_folder = os.path.join(result_folder, 'preds')
    subj_names = os.listdir(preds_folder)
    errors = []
    for subj_name in subj_names:
        error_subj = 0
        subj_folder = os.path.join(preds_folder, subj_name)
        anim_names = os.listdir(subj_folder)
        num_anims = len(anim_names)
        for anim_name in anim_names:
            anim_folder = os.path.join(subj_folder, anim_name)
            joints_3d_pred_path = os.path.join(anim_folder, 'joints_3d.npy')
            scene_folder = os.path.join(dataset_folder, subj_name, anim_name)
            error_scene = evaluate_one_scene(joints_3d_pred_path, scene_folder)
            error_subj += error_scene
        error_subj /= num_anims
        errors.append(error_subj)
    # plot
    plt.bar(range(len(subj_names)), errors, tick_label=subj_names)
    plt.savefig(write_path)

def plot_2D_error_per_subject(dataset_folder, result_folder, write_path, reject_camera=[])
    preds_folder = os.path.join(result_folder, 'preds')
    subj_names = os.listdir(preds_folder)
    errors = []
    for subj_name in subj_names:
        error_subj = 0
        subj_folder = os.path.join(preds_folder, subj_name)
        anim_names = os.listdir(subj_folder)
        num_anims = len(anim_names)
        for anim_name in anim_names:
            anim_folder = os.path.join(subj_folder, anim_name)
            joints_2d_pred_path = os.path.join(anim_folder, 'joints_3d.npy')
            joints_2d_pred = np.load(joints_2d_pred_path)
            scene_folder = os.path.join(dataset_folder, subj_name, anim_name)

            num_frames, num_views, num_joints, _ = joints_2d_pred.shape
            invalid_camera = (0,)
            invalid_joints = (9, 16) # hardcoded
            joints_2d_valid = np.ones(shape=(num_frames, num_views, num_joints, 2))
            joints_2d_valid[:, invalid_camera, invalid_joints, :] = 0
            joints_2d_gt = np.empty(shape=(num_frames, num_views, num_joints, 2))

            proj_mats = []
            for camera_idx in range(num_views):
                camera_file = os.path.join(scene_folder, 'camera_%04d.json' % camera_idx)
                cam = datasets_utils.load_camera(camera_file)
                bbox = [80, 0, 560, 480] # hardcoded
                cam.update_after_crop(bbox)
                proj_mat = cam.get_P()
                proj_mats.append(proj_mat)

            for frame_idx in range(num_frames):
                joints_name = datasets_utils.Joints_SynData
                skeleton_path = os.path.join(scene_folder, 'skeleton_%06d.json' % frame_idx)
                joints_3d_gt = datasets_utils.load_joints(joints_name, skeleton_path)
                for camera_idx in range(num_views):
                    jnt_2d = visualize.proj_to_2D(proj_mats[camera_idx], joints_3d_gt.T) # need rename
                    joints_2d_gt[frame_idx, camera_idx, :, :] = jnt_2d.T

            error_scene = joints_2d_valid * (joints_2d_pred - joints_2d_gt)
            error_scene = np.mean(np.sqrt(np.sum(error_scene**2, axis=3)))
            error_subj += error_scene
            
        error_subj /= num_anims
        errors.append(error_subj)
    # plot
    plt.bar(range(len(subj_names)), errors, tick_label=subj_names)
    plt.savefig(write_path)

if __name__ == "__main__":

    dataset_folder = '../mocap_syndata/multiview_data'
    result_folder = 'results/mocap_syndata'

    plot_2D_error_per_subject(dataset_folder, result_folder, write_path='figs/2D_per_subj.png')
