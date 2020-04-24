import os, sys, shutil
import numpy as np
import json
import matplotlib.pyplot as plt


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

def get_K(w, h, fov):
    """get 3x3 intrinsic matrix."""
    K = np.array([
        [fov, 0, w/2],
        [0, -fov, h/2],
        [0, 0, 1]
        ])
    return K

def get_R(roll, pitch, yaw):
    """get 3x3 rotation matrix (camera w.r.t. world)."""
    roll = -roll / 180.0 * np.pi
    pitch = -pitch / 180.0 * np.pi
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
    ext_mat = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0]]) @ np.hstack((R.T, -R.T @ T)) # extrinsic matrix (rows re-aligned)
    P = K @ ext_mat
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

        K = get_K(width, height, width/2) # intrinsics
        R_c2w = get_R(rot[2], rot[0], rot[1])
        R_w2c = R_c2w.T # rotation
        T_c2w = get_T(pos[0], pos[1], pos[2])
        T_w2c = -R_w2c @ T_c2w # translation
        
        camera_retval = retval['cameras'][0, camera_idx]
        camera_retval['K'] = K
        camera_retval['R'] = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]) @ R_w2c # rows re-aligned
        camera_retval['t'] = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]) @ T_w2c # rows re-aligned
        camera_retval['dist'] = np.zeros(5)

    # hardcode bounding boxes
    bbox = np.array([[0, 80, 480, 560]])
    bboxes = np.tile(bbox, (len(retval['camera_names']), 1))

    # fill retval['table']
    for f in os.listdir(data_path):
        f_path = os.path.join(data_path, f)
        if not os.path.isdir(f_path):
            continue

        skeleton_path = os.path.join(f_path, "skeleton.json")
        with open(skeleton_path) as json_file:
            skeleton = json.load(json_file)

        all_joints = {joint["Name"]: joint["KpWorld"] for joint in skeleton}
        num_jnts = len(joints_name)
        x_keypts = np.array([all_joints[jnt]['X'] for jnt in joints_name])
        y_keypts = np.array([all_joints[jnt]['Y'] for jnt in joints_name])
        z_keypts = np.array([all_joints[jnt]['Z'] for jnt in joints_name])
        keypoints = np.hstack((x_keypts.reshape(num_jnts,1),\
                               y_keypts.reshape(num_jnts,1),\
                               z_keypts.reshape(num_jnts,1)))
            
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
    data_path = os.path.join(path, "multiview_data")
    write_path = os.path.join(path, "processed", "S1", "Standing", "imageSequence-undistorted")
    os.makedirs(write_path, exist_ok=True)
    for i in range(1, 5):
        if not os.path.exists(os.path.join(write_path, "%d" % i)):
            os.mkdir(os.path.join(write_path, "%d" % i))

    for f in os.listdir(data_path):
        f_path = os.path.join(data_path, f)
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

    all_joints = {joint["Name"]: joint["KpWorld"] for joint in skeleton}
    num_jnts = len(joints_name)
    X = np.array([all_joints[jnt]['X'] for jnt in joints_name]).reshape(1,num_jnts)
    Y = np.array([all_joints[jnt]['Y'] for jnt in joints_name]).reshape(1,num_jnts)
    Z = np.array([all_joints[jnt]['Z'] for jnt in joints_name]).reshape(1,num_jnts)
    pts_3d = np.vstack((X, Y, Z)) # 3 x n
    pts_3d_homo = np.vstack((pts_3d, np.ones((1,num_jnts)))) # 4 x n    

    with open(camera_file) as camera_json:
        camera = json.load(camera_json)

    proj_mat = get_P(camera['location'][0], camera['location'][1], camera['location'][2],
                     camera['rotation'][2], camera['rotation'][0], camera['rotation'][1],
                     camera['width'], camera['height'], camera['width'] / 2) # 3 x 4
    pts_2d_homo = proj_mat @ pts_3d_homo # 3 x n
    pts_2d = np.zeros((2, pts_2d_homo.shape[1])) # 2 x n
    pts_2d[0, :] = pts_2d_homo[0, :] / pts_2d_homo[2, :]
    pts_2d[1, :] = pts_2d_homo[1, :] / pts_2d_homo[2, :]

    img = plt.imread(img_path)
    plt.imshow(img)
    plt.scatter(pts_2d[0, :], pts_2d[1, :])
    plt.show()

    
if __name__ == "__main__":
    ds_path = "data/multiview_data"
    generate_label(ds_path)
    reorganize_imgs(ds_path)

    plot_joints_2D("data/multiview_data/multiview_data/000000/view_2.png",
                   "data/multiview_data/multiview_data/000000/skeleton.json",
                   "data/multiview_data/multiview_data/camera_2.json")   
    
