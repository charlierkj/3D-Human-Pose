import os
import numpy as np
import torch
from utils.campose import estimate_campose
import datasets.utils as datasets_utils


if __name__ == "__main__":
    data_folder = "data/test_03/multiview_data"
    preds_folder = "results/test_03/preds"

    subj_names = sorted(os.listdir(preds_folder))
    for subj in subj_names:
        subj_folder = os.path.join(preds_folder, subj)
        anim_names = sorted(os.listdir(subj_folder))
        for anim in anim_names:
            anim_folder = os.path.join(subj_folder, anim)
            scene_folder = os.path.join(data_folder, subj, anim)
            joints_2d = np.load(os.path.join(anim_folder, "joints_2d.npy"))
            confidences = np.load(os.path.join(anim_folder, "confidences.npy"))
            keypoints_2d_l = torch.from_numpy(joints_2d[0, 0, ...])
            keypoints_2d_r = torch.from_numpy(joints_2d[0, 1, ...])
            confidences_l = torch.from_numpy(confidences[0, 0, ...])
            confidences_r = torch.from_numpy(confidences[0, 1, ...])

            cam_l = datasets_utils.load_camera(os.path.join(scene_folder, "camera_0000.json"))
            cam_r = datasets_utils.load_camera(os.path.join(scene_folder, "camera_0001.json"))

            K_l = torch.from_numpy(cam_l.K)
            K_r = torch.from_numpy(cam_r.K)

            g_l = torch.inverse(torch.cat((torch.from_numpy(cam_l.get_extM()), torch.Tensor([[0, 0, 0, 1]]).type(torch.float64)), dim=0))
            g_r = torch.inverse(torch.cat((torch.from_numpy(cam_r.get_extM()), torch.Tensor([[0, 0, 0, 1]]).type(torch.float64)), dim=0))

            g_lr = torch.inverse(g_l) @ g_r

            R_gt = g_lr[0:3, 0:3]
            t_gt = g_lr[0:3, 3]

            R, t = estimate_campose(keypoints_2d_l, keypoints_2d_r, \
                     K_l, K_r, \
                     confidences_l, confidences_r)

            #print(t_gt)
            #print(t)
            print(R_gt)
            print(R)
