import numpy as np
import math

PI = math.pi

def skew_symm_mat(w):
    """hat-operation,
    to compute the associated skew-symmetric matrix given 3D vector."""
    w1, w2, w3 = w[0], w[1], w[2]
    w_hat = np.array([
        [0, -w3, w2],
        [w3, 0, -w1],
        [-w2, w1, 0]
        ])
    return w_hat

def compute_rotation(vec_a, vec_b):
    """compute the 3D rotation matrix from vec_a to vec_b."""
    # normalize
    vec_a = vec_a / np.linalg.norm(vec_a)
    vec_b = vec_b / np.linalg.norm(vec_b)

    # rotation represented by w(axis) and theta(angle)
    w = np.cross(vec_a, vec_b)
    sin_theta = np.linalg.norm(w)
    cos_theta = np.dot(vec_a, vec_b)
    w = w / sin_theta
    w_hat = skew_symm_mat(w)
    rot_mat = np.eye(3) + sin_theta * w_hat + \
              (1 - cos_theta) * (w_hat @ w_hat) # Rodrigue's formula
    return rot_mat

def spherical_to_pose(viewpoint):
    """assuming camera aimed to origin, compute 6DOF camera pose.
    viewpoint: 3-tuple, (distance, azimuth, elevation)
    """
    dist, az, el = viewpoint
    phi = el / 180 * PI
    theta = az / 180 * PI
    x = dist * np.cos(theta) * np.cos(phi)
    y = dist * np.sin(theta) * np.cos(phi)
    z = dist * np.sin(phi)
    roll = - el
    pitch = 0
    yaw = - az + PI
    return [x, y, z], [roll, pitch, yaw]

def euler_to_quaternion(euler):
    roll, pitch, yaw = euler
    c1 = np.cos(yaw / 2.0)
    c2 = np.cos(pitch / 2.0)
    c3 = np.cos(roll / 2.0)    
    s1 = np.sin(yaw / 2.0)
    s2 = np.sin(pitch / 2.0)
    s3 = np.sin(roll / 2.0)    
    qw = c1 * c2 * c3 + s1 * s2 * s3
    qx = c1 * c2 * s3 - s1 * s2 * c3
    qy = c1 * s2 * c3 + s1 * c2 * s3
    qz = s1 * c2 * c3 - c1 * s2 * s3
    return [qw, qx, qy, qz]


    
    
