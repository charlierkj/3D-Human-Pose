import os, sys
import numpy as np
import json


def get_K(w, h, fov):
    """get 3x3 intrinsic matrix."""
    K = np.array([
        [fov, 0, w/2],
        [0, fov, h/2],
        [0, 0, 1]
        ])
    return K

def get_R(roll, pitch, yaw):
    """get 3x3 rotation matrix (camera w.r.t. world)."""
    roll = roll / 180.0 * np.pi
    pitch = pitch / 180.0 * np.pi
    yaw = yaw / 180.0 * np.pi

    Rroll = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
        ])

    Rpitch = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
        ])

    Ryaw = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
        ])

    R = Ryaw @ Rpitch @ Rroll
    return R

def get_T(x, y, z):
    """get 3x1 translation vector (camera w.r.t. world)."""
    T = np.array([
        [x],
        [y],
        [z]
        ])
    return T

def get_P(x, y, z, roll, pitch, yaw, w, h, fov):
    """get 3x4 projection matrix."""
    K = get_K(w, h, fov)
    R = get_R(roll, pitch, yaw)
    T = get_T(x, y, z)
    P = K @ np.hstack((R.T, -R.T @ T))
    return P
    

def generate_label(path):
    retval = {
        'subject_names': ['S1'],
        'camera_names': ['1', '2', '3', '4'],
        'action_names':['Standing']
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

    data_path = os.path.join(path, "multiview_data")
    write_path = os.path.join(path, "extra", "syn_data_labels_bboxes.npy")
    if not os.path.exists(os.path.join(path, "extra")):
        os.mkdir(os.path.join(path, "extra"))

    # fill retval['cameras']
    for camera_idx in range(4):
        camera_file = os.path.join(data_path, "camera_%d.json" % (camera_idx + 1))
        with open(camera_file) as json_file:
            camera_dict = json.load(json_file)
        pos = camera_dict['location']
        rot = camera_dict['rotation']
        width = camera_dict['width']
        height = camera_dict['height']
        fov = camera_dict['fov']

        K = get_K(width, height, fov) # intrinsics
        R_c2w = get_R(rot[2], rot[0], rot[1])
        R_w2c = R_c2w.T # rotation
        T_c2w = get_T(pos[0], pos[1], pos[2])
        T_w2c = -R_w2c @ T_c2w # translation
        
        camera_retval = retval['cameras'][0, camera_idx]
        camera_retval['K'] = K
        camera_retval['R'] = R_w2c
        camera_retval['t'] = T_w2c
        camera_retval['dist'] = np.zeros(5)

    # hardcode bounding boxes
    bbox = np.array([[0, 80, 480, 560]])
    bboxes = np.tile(bbox, (len(retval['camera_names']), 1))

    # fill retval['table']
    joints_name = [
        "foot_r",
        "calf_r",
        "thigh_r",
        "thigh_l",
        "calf_l",
        "foot_l",
        "pelvis",
        "spine_02",
        "head",
        "head_end",
        "hand_r",
        "lowerarm_r",
        "upperarm_r",
        "upperarm_l",
        "lowerarm_l",
        "hand_l",
        "neck_01"
        ]

    for f in os.listdir(data_path):
        f_path = os.path.join(data_path, f)
        if not os.path.isdir(f_path):
            continue

        skeleton_path = os.path.join(f_path, "skeleton.json")
        with open(skeleton_path) as json_file:
            skeleton = json.load(json_file)

        all_joints = {joint["Name"]: joint["KpWorld"] for joint in skeleton}
        x_keypts = np.array([all_joints[jnt]['X'] for jnt in joints_name])
        y_keypts = np.array([all_joints[jnt]['Y'] for jnt in joints_name])
        z_keypts = np.array([all_joints[jnt]['Z'] for jnt in joints_name])
        keypoints = np.hstack((x_keypts.reshape(17,1),\
                               y_keypts.reshape(17,1),\
                               z_keypts.reshape(17,1)))
            
        table_segment = np.empty(1, dtype=table_dtype)
        table_segment['subject_idx'] = 0
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
    return


if __name__ == "__main__":
    ds_path = "data/multiview_data"
    generate_label(ds_path)
    
    
    
