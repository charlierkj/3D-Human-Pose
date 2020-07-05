import os, json

import numpy as np
import torch
from torch import nn

import visualize
from camera_utils import Camera
from test import evaluate_one_scene


class PoseRefiner(nn.Module):

    def __init__(self, joints_3d_pred, joints_2d_pred, proj_mats, weights):
        """weights: list of num_views components."""
        super(PoseRefiner, self).__init__()

        # convert numpy to tensor
        if not torch.is_tensor(joints_3d_pred):
            joints_3d_pred = torch.from_numpy(joints_3d_pred)
        if not torch.is_tensor(joints_2d_pred):
            joints_2d_pred = torch.from_numpy(joints_2d_pred)
        if not torch.is_tensor(proj_mats):
            proj_mats = torch.from_numpy(proj_mats)
        
        self.joints_3d = nn.Parameter(joints_3d_pred) # parameters to optimize

        self.conn = [list(conn) for conn in visualize.CONNECTIVITY_HUMAN36M]
        self.conn = torch.Tensor(self.conn).type(torch.int64) # dtype: torch.long

        # store static params        
        self.register_buffer('joints_3d_original', joints_3d_pred)
        self.register_buffer('joints_2d_original', joints_2d_pred)
        self.register_buffer('proj_mats', proj_mats) # num_views x 3 x 4
        bone_length_median = torch.median(torch.sqrt(torch.sum((joints_3d_pred[:, self.conn[:, 0], :] \
                      - joints_3d_pred[:, self.conn[:, 1], :])**2, dim=2)), dim=0)[0]
        self.register_buffer('bone_length_median', bone_length_median) # median

        self.register_buffer('wl', torch.Tensor([weights[0]]))
        self.register_buffer('wp', torch.Tensor([weights[1]]))
        self.register_buffer('ws', torch.Tensor([weights[2]]))
        self.register_buffer('wb', torch.Tensor([weights[3]]))

    def __device(self):
        return next(self.parameters()).device

    def forward(self):
        lift_loss = self.lift_loss()
        proj_loss = self.proj_loss()
        smooth_loss = self.smooth_loss()
        bone_loss = self.bone_loss()
        loss = self.wl * lift_loss + self.wp * proj_loss + \
               self.ws * smooth_loss + self.wb * bone_loss
        return loss

    def lift_loss(self):
        # loss = torch.sum((self.joints_3d - self.joints_3d_original)**2)
        u, s, v = torch.svd(self.joints_3d - self.joints_3d_original)
        loss = torch.sum(torch.max(s, dim=1)[0])
        return loss

    def proj_loss(self):
        num_frames, num_views = self.joints_2d_original.shape[0], self.joints_2d_original.shape[1]
        loss = 0
        for frame_idx in range(num_frames):
            for view_idx in range(num_views):
                joints_2d_transpose = visualize.proj_to_2D(\
                    self.proj_mats[view_idx, :, :], self.joints_3d[frame_idx, :, :].T) # 2 x num_jnts
                # loss += torch.sum((joints_2d_transpose - self.joints_2d_original[frame_idx, view_idx, :, :].T)**2)
                u, s, v = torch.svd(joints_2d_transpose.T - self.joints_2d_original[frame_idx, view_idx, :, :])
                loss += torch.max(s)
        return loss

    def smooth_loss(self):
        joints_3d_last = self.joints_3d[:-1, :, :]
        joints_3d_next = self.joints_3d[1:, :, :]
        # loss = torch.sum((joints_3d_next - joints_3d_last)**2)
        u, s, v = torch.svd(joints_3d_next - joints_3d_last)
        loss = torch.sum(torch.max(s, dim=1)[0])
        return loss

    def bone_loss(self):
        bone_length = torch.sqrt(torch.sum((self.joints_3d[:, self.conn[:, 0], :] \
                      - self.joints_3d[:, self.conn[:, 1], :])**2, dim=2))
        # median deviation
        loss = torch.sum((bone_length - self.bone_length_median)**2)
        return loss


def refine_one_scene(joints_3d_pred_path, joints_2d_pred_path, scene_folder, save_folder, \
                     weights=[0.1, 0.0005, 1, 1], iterations=1000, lr=0.01):
    joints_3d_pred = np.load(joints_3d_pred_path)
    joints_2d_pred = np.load(joints_2d_pred_path)

    # load camera projection matrices
    proj_mats = []
    (num_frames, num_views, num_jnts, _)  = joints_2d_pred.shape
    for camera_idx in range(num_views):
        camera_name = "%04d" % camera_idx
        camera_file = os.path.join(scene_folder, 'camera_%s.json' % camera_name)
        with open(camera_file) as camera_json:
            camera = json.load(camera_json)
        cam = Camera(camera['location'][0], camera['location'][1], camera['location'][2],
                     camera['rotation'][2], camera['rotation'][0], camera['rotation'][1],
                     camera['width'], camera['height'], camera['width'] / 2)
        bbox = [80, 0, 560, 480] # hardcoded bbox
        cam.update_after_crop(bbox)
        proj_mat = cam.get_P() # 3 x 4
        proj_mats.append(proj_mat)
    proj_mats = np.stack(proj_mats, axis=0) # num_views x 3 x 4

    # initialize model (refiner)
    model = PoseRefiner(joints_3d_pred, joints_2d_pred, proj_mats, weights)
    if torch.cuda.is_available():
        model.cuda()

    # optimize
    optimizer = torch.optim.Adam([model.joints_3d], lr=lr)
    for i in range(iterations):
        model.train()
        optimizer.zero_grad()
        loss = model()
        loss.backward()
        optimizer.step()
        if i % 50 == 0:
            with torch.no_grad():
                model.eval()
                joints_3d_np = model.joints_3d.cpu().numpy()
                metric = evaluate_one_scene(joints_3d_np, scene_folder, path=False)
                print("Iter: %04d | Loss: %.3f | Error: %.3f" % (i, loss.item(), metric))

    # save results
    os.makedirs(save_folder, exist_ok=True)
    # preds
    joints_3d_refined = model.joints_3d.detach().cpu().numpy()
    joints_2d_refined = np.empty((num_frames, num_views, num_jnts, 2))
    for frame_idx in range(num_frames):
        for view_idx in range(num_views):
            joints_2d_transpose = visualize.proj_to_2D(proj_mats[view_idx, :, :], \
                                                       joints_3d_refined[frame_idx, :, :].T)
            joints_2d_refined[frame_idx, view_idx, :, :] = joints_2d_transpose.T
    np.save(os.path.join(save_folder, 'joints_2d_refined.npy'), joints_2d_refined)
    np.save(os.path.join(save_folder, 'joints_3d_refined.npy'), joints_3d_refined)

    # imgs
    imgs_folder = os.path.join(save_folder, 'imgs')
    visualize.draw_one_scene(os.path.join(save_folder, 'joints_3d_refined.npy'), \
                             scene_folder, imgs_folder, show_img=False)

    # videos
    vid_name = os.path.join(save_folder, 'vid.mp4')
    visualize.make_vid(imgs_folder, vid_name)


if __name__ == "__main__":

    joints_3d_pred_path = 'results/test_01/preds/anim_010/joints_3d.npy'
    joints_2d_pred_path = 'results/test_01/preds/anim_010/joints_2d.npy'
    scene_folder = 'data/test_01/multiview_data/S0/anim_010'
    save_folder = 'results/refined_test_01_S0_anim_010'

    refine_one_scene(joints_3d_pred_path, joints_2d_pred_path, scene_folder, save_folder, \
                     weights=[0.1, 0.0001, 10, 1], iterations=1000, lr=0.1)
        
