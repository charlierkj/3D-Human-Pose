import numpy as np

class Camera_UE(object):
    
    def __init__(self, x, y, z, roll, pitch, yaw, w, h, fov):
        self.K = self.set_K(w, h, fov)
        self.R = self.set_R(roll, pitch, yaw)
        self.T = self.set_T(x, y, z)
        # self.o = [w/2, h/2] # principle point in pixel
        self.w, self.h = w, h # original image size
        self.w_c, self.h_c = w, h # current image size
        self.ratio_w, self.ratio_h = 1, 1 # ratio after resize

    def set_K(self, w, h, fov):
        """get 3x3 intrinsic matrix."""
        K = np.array([
            [fov, 0, w/2],
            [0, -fov, h/2],
            [0, 0, 1]
            ])
        return K

    def set_R(self, roll, pitch, yaw):
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

    def set_T(self, x, y, z):
        """get 3x1 translation vector (camera w.r.t. world)."""
        T = np.array([
            [x],
            [y],
            [z]
            ])
        return T

    def get_extM(self):
        """get 3x4 extrinsic matrix."""
        ext_mat = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0]]) @ np.hstack((self.R.T, -self.R.T @ self.T)) # extrinsic matrix (rows re-aligned)
        return ext_mat

    def get_P(self):
        """get 3x4 projection matrix."""
        ext_mat = self.get_extM()
        P = self.K @ ext_mat
        return P

    def update_after_crop(self, bbox):
        upper_left_x, upper_left_y, lower_right_x, lower_right_y = bbox
        self.K[0, 0] /= self.ratio_w # reset fx, fy by ratio
        self.K[1, 1] /= self.ratio_h
        new_ox = self.w / 2 - upper_left_x
        new_oy = self.h / 2 - upper_left_y
        self.K[0, 2] = new_ox
        self.K[1, 2] = new_oy
        self.w_c = lower_right_x - upper_left_x
        self.h_c = lower_right_y - upper_left_y

    def update_after_resize(self, image_size):
        new_w, new_h = image_size
        self.ratio_w = new_w / self.w_c
        self.ratio_h = new_h / self.h_c
        fx, fy, ox, oy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        new_fx = fx * self.ratio_w
        new_fy = fy * self.ratio_h
        new_ox = ox * self.ratio_w
        new_oy = oy * self.ratio_h
        self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2] = new_fx, new_fy, new_ox, new_oy
        self.w_c = new_w
        self.h_c = new_h

        
