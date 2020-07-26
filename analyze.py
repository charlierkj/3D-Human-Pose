import os, json
import numpy as np
import matplotlib.pyplot as plt

from test import evaluate_one_scene, evaluate_one_batch
import datasets.utils as datasets_utils
import visualize


def load_preds(result_folder, invalid_joints=(9, 16)):
    valid_joints = tuple(filter(lambda x : x not in list(invalid_joints), list(range(17))))
    preds_folder = os.path.join(result_folder, 'preds')
    subj_names = sorted(os.listdir(preds_folder))
    joints_3d_pred_all = []
    joints_2d_pred_all = []
    confidences_all = []
    for subj_name in subj_names:
        joints_3d_pred_subj = []
        joints_2d_pred_subj = []
        confidences_subj = []
        subj_folder = os.path.join(preds_folder, subj_name)
        anim_names = sorted(os.listdir(subj_folder))
        for anim_name in anim_names:
            anim_folder = os.path.join(subj_folder, anim_name)
            joints_3d_pred_path = os.path.join(anim_folder, 'joints_3d.npy')
            joints_2d_pred_path = os.path.join(anim_folder, 'joints_2d.npy')
            confidences_path = os.path.join(anim_folder, 'confidences.npy')
            #scene_folder = os.path.join(dataset_folder, subj_name, anim_name)
            joints_3d_pred = np.load(joints_3d_pred_path)
            joints_2d_pred = np.load(joints_2d_pred_path)
            confidences = np.load(confidences_path)
            joints_3d_pred_subj.append(joints_3d_pred)
            joints_2d_pred_subj.append(joints_2d_pred)
            confidences_subj.append(confidences)
        joints_3d_pred_all.append(joints_3d_pred_subj)
        joints_2d_pred_all.append(joints_2d_pred_subj)
        confidences_all.append(confidences_subj)
    joints_3d_pred_all = np.array(joints_3d_pred_all) # size: (num_subjs, num_anims, num_frames, 17, 3)
    joints_2d_pred_all = np.array(joints_2d_pred_all) # size: (num_subjs, num_anims, num_frames, num_views, 17, 2)
    confidences_all = np.array(confidences_all) # size: (num_subjs, num_anims, num_frames, num_views, 17)
    joints_3d_pred_all = joints_3d_pred_all[:, :, :, valid_joints, :]
    joints_2d_pred_all = joints_2d_pred_all[:, :, :, :, valid_joints, :]
    confidences_all = confidences_all[:, :, :, :, valid_joints]
    return joints_3d_pred_all, joints_2d_pred_all, confidences_all


def load_groundtruths(dataset_folder, bbox=[80, 0, 560, 480], invalid_joints=(9, 16)):
    valid_joints = tuple(filter(lambda x : x not in list(invalid_joints), list(range(17))))
    joints_3d_gt_all = []
    joints_2d_gt_all = []
    subj_names = sorted(os.listdir(dataset_folder))
    for subj_name in subj_names:
        joints_3d_gt_subj = []
        joints_2d_gt_subj = []
        subj_folder = os.path.join(dataset_folder, subj_name)
        anim_names = sorted(os.listdir(subj_folder))
        for anim_name in anim_names:
            num_frames = 60 # hardcoded
            num_views = 4 # hardcoded
            joints_3d_gt_anim = np.empty(shape=(num_frames, 17, 3))
            joints_2d_gt_anim = np.empty(shape=(num_frames, num_views, 17, 2))
            anim_folder = os.path.join(subj_folder, anim_name)

            # load camera projection matrices
            proj_mats = []
            for camera_idx in range(num_views):
                camera_file = os.path.join(anim_folder, 'camera_%04d.json' % camera_idx)
                cam = datasets_utils.load_camera(camera_file)
                cam.update_after_crop(bbox)
                proj_mat = cam.get_P()
                proj_mats.append(proj_mat)

            for frame_idx in range(num_frames):
                joints_name = datasets_utils.Joints_SynData
                skeleton_path = os.path.join(anim_folder, 'skeleton_%06d.json' % frame_idx)
                # load 3d gt
                joints_3d_gt = datasets_utils.load_joints(joints_name, skeleton_path)
                joints_3d_gt_anim[frame_idx, :, :] = joints_3d_gt
                # load 2d gt
                for camera_idx in range(num_views):
                    joints_2d_gt = visualize.proj_to_2D(proj_mats[camera_idx], joints_3d_gt.T)
                    joints_2d_gt_anim[frame_idx, camera_idx, :, :] = joints_2d_gt.T
                    
            joints_3d_gt_subj.append(joints_3d_gt_anim)
            joints_2d_gt_subj.append(joints_2d_gt_anim)
            
        joints_3d_gt_all.append(joints_3d_gt_subj)
        joints_2d_gt_all.append(joints_2d_gt_subj)
        
    joints_3d_gt_all = np.array(joints_3d_gt_all) # size: (num_subjs, num_anims, num_frames, 17, 3)
    joints_2d_gt_all = np.array(joints_2d_gt_all) # size: (num_subjs, num_anims, num_frames, num_views, 17, 2)
    joints_3d_gt_all = joints_3d_gt_all[:, :, :, valid_joints, :]
    joints_2d_gt_all = joints_2d_gt_all[:, :, :, :, valid_joints, :]
    return joints_3d_gt_all, joints_2d_gt_all


def plot_3D_error_per_subject(joints_3d_pred_all, joints_3d_gt_all, write_path):
    diff = joints_3d_pred_all - joints_3d_gt_all
    errors = np.sqrt(np.sum(diff ** 2, axis=4))
    error_per_subj = np.mean(errors, axis=(1, 2, 3))
    std = np.std(errors, axis=(1, 2, 3))
    # plot
    plt.bar(range(len(error_per_subj)), error_per_subj, yerr=std, tick_label=range(len(error_per_subj)))
    plt.xlabel('subject index')
    plt.ylabel('MPJPE (cm)')
    write_folder = os.path.split(write_path)[0]
    if write_folder != '':
        os.makedirs(write_folder, exist_ok=True)
    plt.savefig(write_path)
    plt.close()


def plot_2D_error_per_subject(joints_2d_pred_all, joints_2d_gt_all, write_path):
    diff = joints_2d_pred_all - joints_2d_gt_all
    errors = np.sqrt(np.sum(diff ** 2, axis=5))
    error_per_subj = np.mean(errors, axis=(1, 2, 3, 4))
    std = np.std(errors, axis=(1, 2, 3, 4))
    # plot
    plt.bar(range(len(error_per_subj)), error_per_subj, yerr=std, tick_label=range(len(error_per_subj)))
    plt.xlabel('subject index')
    plt.ylabel('2D error (pixel)')
    write_folder = os.path.split(write_path)[0]
    if write_folder != '':
        os.makedirs(write_folder, exist_ok=True)
    plt.savefig(write_path)
    plt.close()


def plot_3D_error_per_joint(joints_3d_pred_all, joints_3d_gt_all, write_path):
    diff = joints_3d_pred_all - joints_3d_gt_all
    errors = np.sqrt(np.sum(diff ** 2, axis=4))
    error_per_joint = np.mean(errors, axis=(0, 1, 2))
    std = np.std(errors, axis=(0, 1, 2))
    # plot
    plt.bar(range(len(error_per_joint)), error_per_joint, yerr=std, tick_label=range(len(error_per_joint)))
    plt.xlabel('joint index')
    plt.ylabel('MPJPE (cm)')
    write_folder = os.path.split(write_path)[0]
    if write_folder != '':
        os.makedirs(write_folder, exist_ok=True)
    plt.savefig(write_path)
    plt.close()


def plot_2D_error_per_joint(joints_2d_pred_all, joints_2d_gt_all, write_path):
    diff = joints_2d_pred_all - joints_2d_gt_all
    errors = np.sqrt(np.sum(diff ** 2, axis=5))
    error_per_joint = np.mean(errors, axis=(0, 1, 2, 3))
    std = np.std(errors, axis=(0, 1, 2, 3))
    # plot
    plt.bar(range(len(error_per_joint)), error_per_joint, yerr=std, tick_label=range(len(error_per_joint)))
    plt.xlabel('joint index')
    plt.ylabel('2D error (pixel)')
    write_folder = os.path.split(write_path)[0]
    if write_folder != '':
        os.makedirs(write_folder, exist_ok=True)
    plt.savefig(write_path)
    plt.close()


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
                    jnt_3d_camspace = visualize.proj_to_camspace(ext_mats[camera_idx], joints_3d_gt.T)
                    depth_gt = jnt_3d_camspace[2, :]
                    for joint_idx in [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15]:
                        x, y = int(joints_2d_gt[joint_idx, 0]+80), int(joints_2d_gt[joint_idx, 1]) # hardcoded
                        if x > 0 and x < 640 and y > 0 and y < 480: # hardcoded
                            dz = depth_npy[y, x]
                            diff = abs(depth_gt[joint_idx] - dz)
                            error = np.linalg.norm(joints_2d_pred[frame_idx, camera_idx, joint_idx, :] - joints_2d_gt[joint_idx, :])
                            if diff < 20:
                                error_noocclusion[joint_idx, 0] += 1
                                error_noocclusion[joint_idx, 1] += error
                            else:
                                error_occlusion[joint_idx, 0] += 1
                                error_occlusion[joint_idx, 1] += error
            
    error_occlusion[:, 1] /=  error_occlusion[:, 0]
    error_noocclusion[:, 1] /=  error_noocclusion[:, 0]
    print(error_occlusion[:, 1])
    print(error_noocclusion[:, 1])
    valid_joints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15]
    # plot
    width = 4
    plt.bar(np.array(valid_joints)-width, error_noocclusion[valid_joints, 1], width=width, label="no occlusion", color='blue')
    plt.bar(np.array(valid_joints)+width, error_occlusion[valid_joints, 1], width=width, label="occlusion", color='red')
    plt.ylabel('error (pixel)')
    plt.xlabel('joint index')
    plt.legend()
    write_folder = os.path.split(write_path)[0]
    if write_folder != '':
        os.makedirs(write_folder, exist_ok=True)
    plt.savefig(write_path)


def plot_corr_confidence_vs_error(joints_2d_pred_all, joints_2d_gt_all, confidences_all, write_path):
    diff = joints_2d_pred_all - joints_2d_gt_all
    errors = np.sqrt(np.sum(diff ** 2, axis=5)) # size: (num_subj, num_anim, num_frames, num_views, num_joints)

    num_joints = errors.shape[-1]
    errors_flat = errors.reshape(-1, num_joints)
    confidences_flat = confidences_all.reshape(-1, num_joints)
                
    # plot
    for joint_idx in range(num_joints):
        plt.scatter(errors_flat[:, joint_idx], confidences_flat[:, joint_idx], s=0.5, label="joint %d" % joint_idx)
    plt.ylabel('confidence')
    plt.xlabel('2D error (pixel)')
    plt.legend()
    write_folder = os.path.split(write_path)[0]
    if write_folder != '':
        os.makedirs(write_folder, exist_ok=True)
    plt.savefig(write_path)



if __name__ == "__main__":

    #dataset_folder = '../mocap_syndata/multiview_data'
    #result_folder = 'results/mocap_syndata'
    dataset_folder = 'data/test_02/multiview_data'
    result_folder = 'results/test_02'

    joints_3d_pred_all, joints_2d_pred_all, confidences_all = load_preds(result_folder)
    joints_3d_gt_all, joints_2d_gt_all = load_groundtruths(dataset_folder)

    print(joints_3d_pred_all.shape)
    print(joints_3d_gt_all.shape)
    print(joints_2d_pred_all.shape)
    print(joints_2d_gt_all.shape)
    print(confidences_all.shape)

    plot_3D_error_per_subject(joints_3d_pred_all, joints_3d_gt_all, write_path='figs/3D_per_subj.png')
    plot_2D_error_per_subject(joints_2d_pred_all, joints_2d_gt_all, write_path='figs/2D_per_subj.png')
    plot_3D_error_per_joint(joints_3d_pred_all, joints_3d_gt_all, write_path='figs/3D_per_joint.png')
    plot_2D_error_per_joint(joints_2d_pred_all, joints_2d_gt_all, write_path='figs/2D_per_joint.png')
    plot_corr_confidence_vs_error(joints_2d_pred_all, joints_2d_gt_all, confidences_all, write_path='figs/confidences_2D_error.png')

    #plot_2D_error_per_subject(dataset_folder, result_folder, write_path='figs/2D_per_subj.png')
    #plot_2D_error_per_joint(dataset_folder, result_folder, write_path='figs/2D_per_joint.png')
    #plot_confidences_2D_error(dataset_folder, result_folder, write_path='figs/confidences_2D_error.png')
    #plot_2D_error_occlusion(dataset_folder, result_folder, write_path='figs/occlusion.png')
