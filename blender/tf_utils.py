import numpy as np

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
    
    
