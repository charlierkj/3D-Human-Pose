import os, sys, shutil
import numpy as np
import json
import matplotlib.pyplot as plt

from camera_utils import *


joints_name = [
        "foot_r",
        "calf_r",
        "thigh_r",
        "thigh_l",
        "calf_l",
        "foot_l",
        "pelvis",
        "spine_02",
        "neck_01", # thorax
        "head_end", # head
        "hand_r",
        "lowerarm_r",
        "upperarm_r",
        "upperarm_l",
        "lowerarm_l",
        "hand_l",
        "head" # neck/nose
        ]


def generate_label(path):
    data_path = os.path.join(path, "multiview_data")
    write_path = os.path.join(path, "extra", "syn_data_labels_bboxes.npy")
    if not os.path.exists(os.path.join(path, "extra")):
        os.mkdir(os.path.join(path, "extra"))

    subj_folders = os.listdir(data_path)
    num_subj = len(subj_folders)
    
    retval = {
        'subject_names': ['S%d' % (i + 1) for i in range(num_subj)],
        'camera_names': ['1', '2', '3', '4'],
        'action_names':['Action']
        }
    retval['cameras'] = np.empty(
        (len(retval['subject_names']), len(retval['camera_names'])),
        dtype=[
            ('R', np.float32, (3,3)),
            ('t', np.float32, (3,1)),
            ('K', np.float32, (3,3)),
            ('dist', np.float32, 5)
            ]
        )

    table_dtype = np.dtype([
        ('subject_idx', np.int8),
        ('action_idx', np.int8),
        ('frame_idx', np.int16),
        ('keypoints', np.float32, (17,3)),
        ('bbox_by_camera_tlbr', np.int16, (len(retval['camera_names']),4))
        ])
    retval['table'] = []

    for (subj_idx, subj_folder) in enumerate(subj_folders):
        subj_path = os.path.join(data_path, subj_folder)
        # fill retval['cameras']
        for camera_idx in range(len(retval['camera_names'])):
            camera_file = os.path.join(subj_path, "camera_%s.json" % retval['camera_names'][camera_idx])
            with open(camera_file) as json_file:
                camera_dict = json.load(json_file)
            pos = camera_dict['location']
            rot = camera_dict['rotation']
            width = camera_dict['width']
            height = camera_dict['height']
            fov = camera_dict['fov']

            cam = Camera(pos[0], pos[1], pos[2], rot[2], rot[0], rot[1], width, height, width/2)
            K = cam.K # intrinsics
            R_c2w = cam.R
            R_w2c = R_c2w.T # rotation
            T_c2w = cam.T
            T_w2c = -R_w2c @ T_c2w # translation
        
            camera_retval = retval['cameras'][subj_idx, camera_idx]
            camera_retval['K'] = K
            camera_retval['R'] = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]) @ R_w2c # rows re-aligned
            camera_retval['t'] = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]) @ T_w2c # rows re-aligned
            camera_retval['dist'] = np.zeros(5)

        # hardcoded bounding boxes
        bbox = np.array([[0, 80, 480, 560]])
        bboxes = np.tile(bbox, (len(retval['camera_names']), 1))

        # fill retval['table']
        for f in os.listdir(subj_path):
            f_path = os.path.join(subj_path, f)
            if not os.path.isdir(f_path):
                continue

            skeleton_path = os.path.join(f_path, "skeleton.json")
            with open(skeleton_path) as json_file:
                skeleton = json.load(json_file)

            all_joints = {joint["Name"].lower(): joint["KpWorld"] for joint in skeleton} # lowercase
            num_jnts = len(joints_name)
            x_keypts = np.array([all_joints[jnt]['X'] for jnt in joints_name])
            y_keypts = np.array([all_joints[jnt]['Y'] for jnt in joints_name])
            z_keypts = np.array([all_joints[jnt]['Z'] for jnt in joints_name])
            keypoints = np.hstack((x_keypts.reshape(num_jnts,1),\
                                   y_keypts.reshape(num_jnts,1),\
                                   z_keypts.reshape(num_jnts,1)))
            
            table_segment = np.empty(1, dtype=table_dtype)
            table_segment['subject_idx'] = subj_idx
            table_segment['action_idx'] = 0
            table_segment['frame_idx'] = int(f)
            table_segment['keypoints'] = keypoints
            table_segment['bbox_by_camera_tlbr'] = bboxes

            retval['table'].append(table_segment)

    retval['table'] = np.concatenate(retval['table'])
    assert retval['table'].ndim == 1

    print("Total frames in the dataset: ", len(retval['table']))
    np.save(write_path, retval)
    return

def reorganize_imgs(path):
    data_path = os.path.join(path, "multiview_data")
    subj_folders = os.listdir(data_path)
    num_subj = len(subj_folders)
    
    for (subj_idx, subj_folder) in enumerate(subj_folders):
        subj_path = os.path.join(data_path, subj_folder)
        write_path = os.path.join(path, "processed", "S%d" % (subj_idx + 1), "Action", "imageSequence-undistorted")
        os.makedirs(write_path, exist_ok=True)
        for i in range(1, 5):
            if not os.path.exists(os.path.join(write_path, "%d" % i)):
                os.mkdir(os.path.join(write_path, "%d" % i))

        for f in os.listdir(subj_path):
            f_path = os.path.join(subj_path, f)
            if not os.path.isdir(f_path):
                continue

            dest_filename = "img_%06d.jpg" % (int(f) + 1)
            for i in range(1, 5):
                src_path = os.path.join(f_path, "view_%d.png" % i)
                dest_path = os.path.join(write_path, "%d" % i, dest_filename)
                shutil.copy(src_path, dest_path)

def plot_joints_2D(img_path, skeleton_file, camera_file):
    with open(skeleton_file) as skeleton_json:
        skeleton = json.load(skeleton_json)

    all_joints = {joint["Name"].lower(): joint["KpWorld"] for joint in skeleton} # lowercase
    num_jnts = len(joints_name)
    X = np.array([all_joints[jnt]['X'] for jnt in joints_name]).reshape(1,num_jnts)
    Y = np.array([all_joints[jnt]['Y'] for jnt in joints_name]).reshape(1,num_jnts)
    Z = np.array([all_joints[jnt]['Z'] for jnt in joints_name]).reshape(1,num_jnts)
    pts_3d = np.vstack((X, Y, Z)) # 3 x n
    pts_3d_homo = np.vstack((pts_3d, np.ones((1,num_jnts)))) # 4 x n    

    with open(camera_file) as camera_json:
        camera = json.load(camera_json)

    cam = Camera(camera['location'][0], camera['location'][1], camera['location'][2],
                 camera['rotation'][2], camera['rotation'][0], camera['rotation'][1],
                 camera['width'], camera['height'], camera['width'] / 2)
    proj_mat = cam.get_P() # 3 x 4
    pts_2d_homo = proj_mat @ pts_3d_homo # 3 x n
    pts_2d = np.zeros((2, pts_2d_homo.shape[1])) # 2 x n
    pts_2d[0, :] = pts_2d_homo[0, :] / pts_2d_homo[2, :]
    pts_2d[1, :] = pts_2d_homo[1, :] / pts_2d_homo[2, :]

    img = plt.imread(img_path)
    plt.imshow(img)
    plt.scatter(pts_2d[0, :], pts_2d[1, :])
    plt.show()

    
if __name__ == "__main__":
    ds_path = "data/multiview_data_2"
    generate_label(ds_path)
    reorganize_imgs(ds_path)

    plot_joints_2D("data/multiview_data_2/multiview_data/person0000-anim0000/000000/view_2.png",
                   "data/multiview_data_2/multiview_data/person0000-anim0000/000000/skeleton.json",
                   "data/multiview_data_2/multiview_data/person0000-anim0000/camera_2.json")   
    
