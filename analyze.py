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


def load_groundtruths(dataset_folder, bbox=[80, 0, 560, 480], invalid_joints=(9, 16), \
                      with_occlusion=False, occlusion_thresh=20):
    valid_joints = tuple(filter(lambda x : x not in list(invalid_joints), list(range(17))))
    joints_3d_gt_all = []
    joints_2d_gt_all = []
    occlusion_gt_all = []
    subj_names = sorted(os.listdir(dataset_folder))
    for subj_name in subj_names:
        joints_3d_gt_subj = []
        joints_2d_gt_subj = []
        occlusion_gt_subj = []
        subj_folder = os.path.join(dataset_folder, subj_name)
        anim_names = sorted(os.listdir(subj_folder))
        for anim_name in anim_names:
            num_frames = 60 # hardcoded
            num_views = 4 # hardcoded
            num_joints = 17 # hardcoded
            joints_3d_gt_anim = np.empty(shape=(num_frames, 17, 3))
            joints_2d_gt_anim = np.empty(shape=(num_frames, num_views, 17, 2))
            occlusion_gt_anim = np.zeros(shape=(num_frames, num_views, 17), dtype=bool)
            anim_folder = os.path.join(subj_folder, anim_name)

            # load camera projection matrices and extrinsic matrices
            proj_mats = []
            ext_mats = []
            for camera_idx in range(num_views):
                camera_file = os.path.join(anim_folder, 'camera_%04d.json' % camera_idx)
                cam = datasets_utils.load_camera(camera_file)
                cam.update_after_crop(bbox)
                proj_mat = cam.get_P()
                proj_mats.append(proj_mat)
                ext_mat = cam.get_extM()
                ext_mats.append(ext_mat)                

            for frame_idx in range(num_frames):
                joints_name = datasets_utils.Joints_SynData[0:17] # hardcoded
                skeleton_path = os.path.join(anim_folder, 'skeleton_%06d.json' % frame_idx)
                # load 3d gt
                joints_3d_gt = datasets_utils.load_joints(joints_name, skeleton_path)
                joints_3d_gt_anim[frame_idx, :, :] = joints_3d_gt
                # load 2d gt
                for camera_idx in range(num_views):
                    joints_2d_gt = visualize.proj_to_2D(proj_mats[camera_idx], joints_3d_gt.T)
                    joints_2d_gt = joints_2d_gt.T
                    joints_2d_gt_anim[frame_idx, camera_idx, :, :] = joints_2d_gt
                    
                    # load occlusion gt when required
                    if with_occlusion:
                        depth_map_path = os.path.join(anim_folder, '%04d' % camera_idx, '%06d.npy' % frame_idx)
                        depth_map = np.load(depth_map_path)
                        joints_3d_gt_camspace = visualize.proj_to_camspace(ext_mats[camera_idx], joints_3d_gt.T)
                        joints_depth_gt = joints_3d_gt_camspace[2, :]
                        for joint_idx in range(num_joints):
                            x, y = int(joints_2d_gt[joint_idx, 0] + bbox[0]), int(joints_2d_gt[joint_idx, 1] + bbox[1])
                            if x > 0 and x < 640 and y > 0 and y < 480: # hardcoded
                                dz = depth_map[y, x]
                                diff = abs(joints_depth_gt[joint_idx] - dz)
                                if diff > occlusion_thresh:
                                    occlusion_gt_anim[frame_idx, camera_idx, joint_idx] = True
                                    
            joints_3d_gt_subj.append(joints_3d_gt_anim)
            joints_2d_gt_subj.append(joints_2d_gt_anim)
            occlusion_gt_subj.append(occlusion_gt_anim)
            
        joints_3d_gt_all.append(joints_3d_gt_subj)
        joints_2d_gt_all.append(joints_2d_gt_subj)
        occlusion_gt_all.append(occlusion_gt_subj)
        
    joints_3d_gt_all = np.array(joints_3d_gt_all) # size: (num_subjs, num_anims, num_frames, 17, 3)
    joints_2d_gt_all = np.array(joints_2d_gt_all) # size: (num_subjs, num_anims, num_frames, num_views, 17, 2)
    occlusion_gt_all = np.array(occlusion_gt_all, dtype=bool) # size: (num_subjs, num_anims, num_frames, num_views, 17)
    joints_3d_gt_all = joints_3d_gt_all[:, :, :, valid_joints, :]
    joints_2d_gt_all = joints_2d_gt_all[:, :, :, :, valid_joints, :]
    occlusion_gt_all = occlusion_gt_all[:, :, :, :, valid_joints]
    if with_occlusion:
        return joints_3d_gt_all, joints_2d_gt_all, occlusion_gt_all
    else:
        return joints_3d_gt_all, joints_2d_gt_all


def load_params(dataset_folder):
    params_all = []
    subj_names = sorted(os.listdir(dataset_folder))
    for subj_name in subj_names:
        params_subj = []
        subj_folder = os.path.join(dataset_folder, subj_name)
        anim_names = sorted(os.listdir(subj_folder))
        for anim_name in anim_names:
            num_frames = 60 # hardcoded
            num_views = 4 # hardcoded
            params_anim = np.empty(shape=(num_frames, num_views, 3)) # (distance, rel-azimuth, elevation)
            anim_folder = os.path.join(subj_folder, anim_name)
            scene_path = os.path.join(anim_folder, 'scene.json')
            with open(scene_path) as scene_json:
                scene = json.load(scene_json)
            human_loc = np.array(scene["human"]["location"])
            human_rot = scene["human"]["rotation"][1]
            for camera_idx in range(num_views):
                camera_loc = np.array(scene["cameras"][camera_idx]["location"])
                camera_az = scene["cameras"][camera_idx]["rotation"][1] - 180
                camera_el = - scene["cameras"][camera_idx]["rotation"][0]
                dist = np.linalg.norm(camera_loc - human_loc)
                rel_az = (camera_az - human_rot) % 360
                el = camera_el
                params_anim[:, camera_idx, 0] = dist
                params_anim[:, camera_idx, 1] = rel_az
                params_anim[:, camera_idx, 2] = el

            params_subj.append(params_anim)

        params_all.append(params_subj)

    params_all = np.array(params_all) # size: (num_subj, num_anim, num_frames, num_views, 3)
    return params_all


def plot_3D_error_per_subject(joints_3d_pred_all, joints_3d_gt_all, write_path, with_PCK=False, PCK_thresh=5):
    diff = joints_3d_pred_all - joints_3d_gt_all
    errors = np.sqrt(np.sum(diff ** 2, axis=4))
    error_per_subj = np.mean(errors, axis=(1, 2, 3))
    std = np.std(errors, axis=(1, 2, 3))
    # plot
    if with_PCK:
        pck_per_subj = np.mean(errors <= PCK_thresh, axis=(1 ,2, 3))
        
        fig, axs = plt.subplots(2)
        axs[0].bar(range(len(error_per_subj)), error_per_subj, yerr=std, tick_label=range(len(error_per_subj)))
        axs[0].set_ylabel('MPJPE (cm)')

        axs[1].bar(range(len(pck_per_subj)), pck_per_subj, tick_label=range(len(pck_per_subj)))
        axs[1].set_ylabel('PCK3D (threshold = %.f cm)' % PCK_thresh)
        axs[1].set_xlabel('subject index')

    else:
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


def plot_3D_error_per_joint(joints_3d_pred_all, joints_3d_gt_all, write_path, with_PCK=False, PCK_thresh=5):
    diff = joints_3d_pred_all - joints_3d_gt_all
    errors = np.sqrt(np.sum(diff ** 2, axis=4))
    error_per_joint = np.mean(errors, axis=(0, 1, 2))
    std = np.std(errors, axis=(0, 1, 2))
    # plot
    if with_PCK:
        pck_per_joint = np.mean(errors <= PCK_thresh, axis=(0, 1, 2))
        
        fig, axs = plt.subplots(2)
        axs[0].bar(range(len(error_per_joint)), error_per_joint, yerr=std, tick_label=range(len(error_per_joint)))
        axs[0].set_ylabel('MPJPE (cm)')

        axs[1].bar(range(len(pck_per_joint)), pck_per_joint, tick_label=range(len(pck_per_joint)))
        axs[1].set_ylabel('PCK3D (threshold = %.1f cm)' % PCK_thresh)
        axs[1].set_xlabel('joint index')

    else:
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


def plot_3D_error_per_joint_vs_occlusion(joints_3d_pred_all, joints_3d_gt_all, occlusion_gt_all, write_path):
    diff = joints_3d_pred_all - joints_3d_gt_all
    errors = np.sqrt(np.sum(diff ** 2, axis=4))
    num_occlusions = np.sum(occlusion_gt_all, axis=3)

    bar_width = 0.15
    for num in range(int(num_occlusions.min()), int(num_occlusions.max())+1):
        valid = (num_occlusions == num)
        sum_error = np.sum(valid * errors, axis=(0, 1, 2))
        num_valid = np.sum(valid, axis=(0, 1, 2))
        error_per_joint = sum_error / num_valid
        
        # plot
        plt.bar(np.arange(len(error_per_joint)) + num * bar_width, error_per_joint, \
                width=bar_width, label="%d view occluded" % num)
    plt.xticks(np.arange(len(error_per_joint)))
    plt.xlabel('joint index')
    plt.ylabel('MPJPE (cm)')
    plt.legend()    
    write_folder = os.path.split(write_path)[0]
    if write_folder != '':
        os.makedirs(write_folder, exist_ok=True)
    plt.savefig(write_path)
    plt.close()
    

def plot_2D_error_per_joint_vs_occlusion(joints_2d_pred_all, joints_2d_gt_all, occlusion_gt_all, write_path):
    diff = joints_2d_pred_all - joints_2d_gt_all
    errors = np.sqrt(np.sum(diff ** 2, axis=5))
    sum_error_occlusion = np.sum(occlusion_gt_all * errors, axis=(0, 1, 2, 3))
    sum_error_noocclusion = np.sum(~occlusion_gt_all * errors, axis=(0, 1, 2, 3))
    num_occlusion = np.sum(occlusion_gt_all, axis=(0, 1, 2, 3))
    num_noocclusion = np.sum(~occlusion_gt_all, axis=(0, 1, 2, 3))
    error_per_joint_occlusion = sum_error_occlusion / num_occlusion
    error_per_joint_noocclusion = sum_error_noocclusion / num_noocclusion
    # plot
    bar_width = 0.4
    plt.bar(np.arange(len(error_per_joint_occlusion)), error_per_joint_occlusion, \
            width=bar_width, label="occlusion")
    plt.bar(np.arange(len(error_per_joint_noocclusion)) + bar_width, error_per_joint_noocclusion, \
            width=bar_width, label="no occlusion")
    plt.xticks(np.arange(len(error_per_joint_occlusion)))
    # plt.set_xticklabels = (range(len(error_per_joint_occlusion)))
    plt.xlabel('joint index')
    plt.ylabel('2D error (pixel)')
    plt.legend()
    write_folder = os.path.split(write_path)[0]
    if write_folder != '':
        os.makedirs(write_folder, exist_ok=True)
    plt.savefig(write_path)
    plt.close()


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
    plt.close()


def plot_2D_error_per_joint_vs_dist(joints_2d_pred_all, joints_2d_gt_all, params_all, write_path, interval=10):
    dist_min = 170
    dist_max = 250
    dist_bins = int((dist_max - dist_min) / interval)
    num_joints = joints_2d_pred_all.shape[4]
    results = np.empty(shape=(num_joints, dist_bins))
    params_dist = np.repeat(np.expand_dims(params_all[:, :, :, :, 0], axis=4), num_joints, axis=4)
    # size: (num_subj, num_anim, num_frames, num_views, num_joints)
    
    diff = joints_2d_pred_all - joints_2d_gt_all
    errors = np.sqrt(np.sum(diff ** 2, axis=5))
    tick_names = []
    for b in range(dist_bins):
        b_min = dist_min + b * interval
        b_max = dist_min + (b + 1) * interval
        valid = (params_dist >= b_min) & (params_dist < b_max)
        sum_error = np.sum(valid * errors, axis=(0, 1, 2, 3))
        num_valid = np.sum(valid, axis=(0, 1, 2, 3))
        error_per_joint = sum_error / num_valid
        results[:, b] = error_per_joint
        tick_names.append("%.1f~%.1f" % (b_min / 100, b_max / 100))
    
    # plot
    for joint_idx in range(num_joints):
        plt.errorbar(range(len(results[joint_idx, :])), results[joint_idx, :], label="joint %d" % joint_idx)
    plt.xticks(np.arange(dist_bins), tick_names)
    plt.xlabel('distance to subject (m)')
    plt.ylabel('2D error (pixel)')
    plt.legend()
    write_folder = os.path.split(write_path)[0]
    if write_folder != '':
        os.makedirs(write_folder, exist_ok=True)
    plt.savefig(write_path)
    plt.close()


def plot_2D_error_per_joint_vs_az(joints_2d_pred_all, joints_2d_gt_all, params_all, write_path, interval=45):
    az_bins = int(360 / interval)
    num_joints = joints_2d_pred_all.shape[4]
    results = np.empty(shape=(num_joints, az_bins))
    params_az = np.repeat(np.expand_dims(params_all[:, :, :, :, 1], axis=4), num_joints, axis=4)
    # size: (num_subj, num_anim, num_frames, num_views, num_joints)
    
    diff = joints_2d_pred_all - joints_2d_gt_all
    errors = np.sqrt(np.sum(diff ** 2, axis=5))
    tick_names = []
    for b in range(az_bins):
        b_min = (b * interval - interval / 2) % 360
        b_max = (b * interval + interval / 2) % 360
        if b == 0:
            valid = np.logical_or((params_az >= b_min), (params_az < b_max))
        else:
            valid = (params_az >= b_min) & (params_az < b_max)
        sum_error = np.sum(valid * errors, axis=(0, 1, 2, 3))
        num_valid = np.sum(valid, axis=(0, 1, 2, 3))
        error_per_joint = sum_error / num_valid
        results[:, b] = error_per_joint
        tick_names.append("%.1f" % (b * interval))
    
    # plot
    for joint_idx in range(num_joints):
        plt.errorbar(range(len(results[joint_idx, :])), results[joint_idx, :], label="joint %d" % joint_idx)
    plt.xticks(np.arange(az_bins), tick_names)
    plt.xlabel('relative azimuth (deg)')
    plt.ylabel('2D error (pixel)')
    plt.legend()
    write_folder = os.path.split(write_path)[0]
    if write_folder != '':
        os.makedirs(write_folder, exist_ok=True)
    plt.savefig(write_path)
    plt.close()


if __name__ == "__main__":

    dataset_folder = '../mocap_syndata/multiview_data'
    result_folder = 'results/mocap_syndata'
    #dataset_folder = 'data/test_03/multiview_data'
    #result_folder = 'results/test_03'

    joints_3d_pred_all, joints_2d_pred_all, confidences_all = load_preds(result_folder)
    print("Predictions loading done.")
    joints_3d_gt_all, joints_2d_gt_all, occlusion_gt_all = load_groundtruths(dataset_folder, with_occlusion=True)
    print("Groundtruths loading done.")
    params_all = load_params(dataset_folder)
    print("Parameters loading done.")

    # occlusion_gt_all = np.random.choice(a=[True, False], size=confidences_all.shape)

    # num_subj, num_anim, num_frames, num_views, _ = confidences_all.shape
    # params_all = np.concatenate((np.random.uniform(low=170, high=250, size=(num_subj, num_anim, num_frames, num_views, 1)),
    #                             np.random.uniform(low=0, high=360, size=(num_subj, num_anim, num_frames, num_views, 1)),
    #                             np.random.uniform(low=-15, high=15, size=(num_subj, num_anim, num_frames, num_views, 1))),
    #                            axis=4)

    print(joints_3d_pred_all.shape)
    print(joints_3d_gt_all.shape)
    print(joints_2d_pred_all.shape)
    print(joints_2d_gt_all.shape)
    print(confidences_all.shape)
    print(occlusion_gt_all.shape)
    print(params_all.shape)

    plot_3D_error_per_subject(joints_3d_pred_all, joints_3d_gt_all, write_path='figs/3D_per_subj.png', with_PCK=True)
    plot_2D_error_per_subject(joints_2d_pred_all, joints_2d_gt_all, write_path='figs/2D_per_subj.png')
    plot_3D_error_per_joint(joints_3d_pred_all, joints_3d_gt_all, write_path='figs/3D_per_joint.png', with_PCK=True)
    plot_2D_error_per_joint(joints_2d_pred_all, joints_2d_gt_all, write_path='figs/2D_per_joint.png')
    plot_corr_confidence_vs_error(joints_2d_pred_all, joints_2d_gt_all, confidences_all, write_path='figs/confidences_2D_error.png')
    plot_2D_error_per_joint_vs_occlusion(joints_2d_pred_all, joints_2d_gt_all, occlusion_gt_all, write_path='figs/2D_per_joint_occlusion.png')
    plot_3D_error_per_joint_vs_occlusion(joints_3d_pred_all, joints_3d_gt_all, occlusion_gt_all, write_path='figs/3D_per_joint_occlusion.png')
    plot_2D_error_per_joint_vs_dist(joints_2d_pred_all, joints_2d_gt_all, params_all, write_path='figs/2D_per_joint_dist.png')
    plot_2D_error_per_joint_vs_az(joints_2d_pred_all, joints_2d_gt_all, params_all, write_path='figs/2D_per_joint_az.png')

    #plot_2D_error_per_subject(dataset_folder, result_folder, write_path='figs/2D_per_subj.png')
    #plot_2D_error_per_joint(dataset_folder, result_folder, write_path='figs/2D_per_joint.png')
    #plot_confidences_2D_error(dataset_folder, result_folder, write_path='figs/confidences_2D_error.png')
    #plot_2D_error_occlusion(dataset_folder, result_folder, write_path='figs/occlusion.png')
