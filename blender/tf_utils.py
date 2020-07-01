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

def viewpoint_to_eulerXYZ(viewpoint, target=(0, 0, 0)):
    """assuming camera aimed to origin, compute 6DOF camera pose.
    viewpoint: 3-tuple, (distance, azimuth, elevation)
    target: 3-tuple, (x, y, z). where should camera be aiming.
    """
    dist, az, el = viewpoint
    phi = el / 180 * PI
    theta = az / 180 * PI
    x = dist * np.cos(theta) * np.cos(phi) + target[0]
    y = dist * np.sin(theta) * np.cos(phi) + target[1]
    z = dist * np.sin(phi) + target[2]
    rx = PI / 2 - phi
    ry = 0
    rz = theta + PI / 2
    return [x, y, z], [rx, ry, rz]

def eulerXYZ_to_quaternion(eulerXYZ):
    rx, ry, rz = eulerXYZ
    q1 = angle_axis_to_quaternion(rx, [1, 0, 0])
    q2 = angle_axis_to_quaternion(ry, [0, 1, 0])
    q3 = angle_axis_to_quaternion(rz, [0, 0, 1])
    q = quaternion_product(quaternion_product(q3, q2), q1)
    return q

def angle_axis_to_quaternion(theta, w):
    """
    theta: scalar (angle in radians)
    w: 3-dim vector (axis)
    """
    w = w / np.linalg.norm(w)
    qw = np.cos(theta / 2)
    qv = np.sin(theta / 2) * np.array(w)
    return [qw, qv[0], qv[1], qv[2]]

def quaternion_product(q, p):
    qw, qv = q[0], np.array(q[1:])
    pw, pv = p[0], np.array(p[1:])
    prodw = qw * pw - np.dot(qv, pv)
    prodv = qw * pv + pw * qv + np.cross(qv, pv)
    return [prodw, prodv[0], prodv[1], prodv[2]]
    


    
    
