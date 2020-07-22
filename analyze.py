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
    write_folder = os.path.split(write_path)[0]
    if write_folder != '':
        os.makedirs(write_folder, exist_ok=True)
    plt.savefig(write_path)

def plot_2D_error_per_subject(dataset_folder, result_folder, write_path, reject_camera=[]):
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
            joints_2d_pred_path = os.path.join(anim_folder, 'joints_2d.npy')
            joints_2d_pred = np.load(joints_2d_pred_path)
            joints_2d_pred[:, :, :, 1] = 480 - joints_2d_pred[:, :, :, 1] # hardcoded
            scene_folder = os.path.join(dataset_folder, subj_name, anim_name)
            
            num_frames, num_views, num_joints, _ = joints_2d_pred.shape
            invalid_camera = (0,)
            invalid_joints = (9, 16) # hardcoded
            joints_2d_valid = np.ones(shape=(num_frames, num_views, num_joints, 2))
            joints_2d_valid[:, invalid_camera, :, :] = 0
            joints_2d_valid[:, :, invalid_joints, :] = 0
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
            error_scene = np.sum(np.sqrt(np.sum(error_scene**2, axis=3))) / (num_frames * 3 * 15) # hardcoded
            error_subj += error_scene
            
        error_subj /= num_anims
        errors.append(error_subj)
    # plot
    plt.bar(range(len(subj_names)), errors, tick_label=subj_names)
    plt.ylabel('error (pixel)')
    plt.xlabel('subject')
    write_folder = os.path.split(write_path)[0]
    if write_folder != '':
        os.makedirs(write_folder, exist_ok=True)
    plt.savefig(write_path)


def plot_2D_error_per_joint(dataset_folder, result_folder, write_path, reject_camera=[]):
    preds_folder = os.path.join(result_folder, 'preds')
    subj_names = os.listdir(preds_folder)
    error_scene = np.zeros((60, 4, 17)) # hardcoded
    for subj_name in subj_names:
        error_subj = 0
        subj_folder = os.path.join(preds_folder, subj_name)
        anim_names = os.listdir(subj_folder)
        num_anims = len(anim_names)
        for anim_name in anim_names:
            anim_folder = os.path.join(subj_folder, anim_name)
            joints_2d_pred_path = os.path.join(anim_folder, 'joints_2d.npy')
            joints_2d_pred = np.load(joints_2d_pred_path)
            joints_2d_pred[:, :, :, 1] = 480 - joints_2d_pred[:, :, :, 1] # hardcoded
            scene_folder = os.path.join(dataset_folder, subj_name, anim_name)
            
            num_frames, num_views, num_joints, _ = joints_2d_pred.shape
            invalid_camera = (0,)
            invalid_joints = (9, 16) # hardcoded
            joints_2d_valid = np.ones(shape=(num_frames, num_views, num_joints, 2))
            joints_2d_valid[:, invalid_camera, :, :] = 0
            joints_2d_valid[:, :, invalid_joints, :] = 0
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

            error_scene += np.sqrt(np.sum((joints_2d_valid * (joints_2d_pred - joints_2d_gt))**2, axis=3))
            
    errors = np.sum(error_scene, axis=(0, 1)) / (6 * 55 * 60 * 3) # hardcoded
    # print(errors)
    # plot
    plt.bar(range(17), errors, tick_label=range(17))
    plt.ylabel('error (pixel)')
    plt.xlabel('joint index')
    write_folder = os.path.split(write_path)[0]
    if write_folder != '':
        os.makedirs(write_folder, exist_ok=True)
    plt.savefig(write_path)


def plot_2D_error_occlusion(dataset_folder, result_folder, write_path, reject_camera=[]):
    preds_folder = os.path.join(result_folder, 'preds')
    subj_names = os.listdir(preds_folder)
    error_noocclusion = np.zeros((17, 2)) # hardcoded, (counts, accum_error)
    error_occlusion = np.zeros((17, 2)) # hardcoded, (counts, accum_error)
    for subj_name in subj_names:
        error_subj = 0
        subj_folder = os.path.join(preds_folder, subj_name)
        anim_names = os.listdir(subj_folder)
        num_anims = len(anim_names)
        for anim_name in anim_names:
            anim_folder = os.path.join(subj_folder, anim_name)
            joints_2d_pred_path = os.path.join(anim_folder, 'joints_2d.npy')
            joints_2d_pred = np.load(joints_2d_pred_path)
            joints_2d_pred[:, :, :, 1] = 480 - joints_2d_pred[:, :, :, 1] # hardcoded
            scene_folder = os.path.join(dataset_folder, subj_name, anim_name)
            
            num_frames, num_views, num_joints, _ = joints_2d_pred.shape

            proj_mats = []
            ext_mats = []
            for camera_idx in range(num_views):
                camera_file = os.path.join(scene_folder, 'camera_%04d.json' % camera_idx)
                cam = datasets_utils.load_camera(camera_file)
                bbox = [80, 0, 560, 480] # hardcoded
                cam.update_after_crop(bbox)
                proj_mat = cam.get_P()
                proj_mats.append(proj_mat)
                ext_mat = cam.get_extM()
                ext_mats.append(ext_mat)

            for frame_idx in range(num_frames):
                joints_name = datasets_utils.Joints_SynData
                skeleton_path = os.path.join(scene_folder, 'skeleton_%06d.json' % frame_idx)
                joints_3d_gt = datasets_utils.load_joints(joints_name, skeleton_path)
                for camera_idx in range(1, num_views):
                    depth_npy_path = os.path.join(scene_folder, '%04d' % camera_idx, '%06d.npy' % frame_idx)
                    depth_npy = np.load(depth_npy_path)
                    jnt_2d = visualize.proj_to_2D(proj_mats[camera_idx], joints_3d_gt.T) # need rename
                    joints_2d_gt = jnt_2d.T
                    jnt_3d_camspace = visualize.proj_t0_camspace(ext_mats[camera_idx], joints_3d_gt.T)
                    depth_gt = jnt_3d_camspace[2, :]
                    for joint_idx in [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15]:
                        x, y = int(joints_2d_gt[joint_idx, 0]), int(joints_2d_gt[joint_idx, 1]) 

            error_scene += np.sqrt(np.sum((joints_2d_valid * (joints_2d_pred - joints_2d_gt))**2, axis=3))
            
    errors = np.sum(error_scene, axis=(0, 1)) / (6 * 55 * 60 * 3) # hardcoded
    print(errors)
    # plot
    plt.bar(range(17), errors, tick_label=range(17))
    plt.ylabel('error (pixel)')
    plt.xlabel('joint index')
    write_folder = os.path.split(write_path)[0]
    if write_folder != '':
        os.makedirs(write_folder, exist_ok=True)
    plt.savefig(write_path)


def plot_confidences_2D_error(dataset_folder, result_folder, write_path, reject_camera=[]):
    preds_folder = os.path.join(result_folder, 'preds')
    subj_names = os.listdir(preds_folder)
    confidences = []
    errors = []
    for subj_name in subj_names:
        error_subj = 0
        subj_folder = os.path.join(preds_folder, subj_name)
        anim_names = os.listdir(subj_folder)
        num_anims = len(anim_names)
        for anim_name in anim_names:
            anim_folder = os.path.join(subj_folder, anim_name)
            joints_2d_pred_path = os.path.join(anim_folder, 'joints_2d.npy')
            joints_2d_pred = np.load(joints_2d_pred_path)
            joints_2d_pred[:, :, :, 1] = 480 - joints_2d_pred[:, :, :, 1] # hardcoded
            confidences_scene_path = os.path.join(anim_folder, 'confidences.npy')
            confidences_scene = np.load(confidences_scene_path)
            scene_folder = os.path.join(dataset_folder, subj_name, anim_name)
            
            num_frames, num_views, num_joints, _ = joints_2d_pred.shape
            invalid_camera = (0,)
            invalid_joints = (9, 16) # hardcoded
            joints_2d_valid = np.ones(shape=(num_frames, num_views, num_joints, 2))
            joints_2d_valid[:, invalid_camera, :, :] = 0
            joints_2d_valid[:, :, invalid_joints, :] = 0
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

            error_scene = np.sqrt(np.sum((joints_2d_valid * (joints_2d_pred - joints_2d_gt))**2, axis=3))

            # reject invalids
            valid_jnts = (0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15)
            error_scene = error_scene[:, 1:, :]
            error_scene = error_scene[:, :, valid_jnts]
            confidences_scene = confidences_scene[:, 1:, :]
            confidences_scene = confidences_scene[:, :, valid_jnts]
            for e in error_scene.flatten():
                errors.append(e)
            for c in confidences_scene.flatten():
                confidences.append(c)
                
    # plot
    plt.scatter(errors, confidences, s=0.5)
    # print(np.min(errors), np.max(errors))
    # print(np.min(confidences), np.max(confidences))
    plt.ylabel('confidences')
    plt.xlabel('error (pixel)')
    write_folder = os.path.split(write_path)[0]
    if write_folder != '':
        os.makedirs(write_folder, exist_ok=True)
    plt.savefig(write_path)



if __name__ == "__main__":

    dataset_folder = '../mocap_syndata/multiview_data'
    result_folder = 'results/mocap_syndata'

    plot_2D_error_per_subject(dataset_folder, result_folder, write_path='figs/2D_per_subj.png')
    plot_2D_error_per_joint(dataset_folder, result_folder, write_path='figs/2D_per_joint.png')
    plot_confidences_2D_error(dataset_folder, result_folder, write_path='figs/confidences_2D_error.png')
