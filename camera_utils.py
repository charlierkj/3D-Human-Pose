import numpy as np

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
    
