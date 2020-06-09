import os, json
import numpy as np
import torch
from PIL import Image

from mvn.utils import cfg
from mvn.models_temp.triangulation import AlgebraicTriangulationNet
from datasets.multiview_syndata import MultiView_SynData
import datasets.utils as datasets_utils

import visualize

def evaluate(joints_3d_pred, joints_3d_gt_batch, joints_3d_valid_batch):
    """MPJPE (mean per joint position error) in cm"""
    error_average = torch.sqrt(((joints_3d_pred - joints_3d_gt_batch) ** 2).sum(2)) # batch_size x num_joints
    error_average = joints_3d_valid_batch * error_average.unsqueeze(2) # batch_size x num_joints x 1
    error_average = error_average.mean((1, 2)) # mean error per sample in batch
    return error_average

def multiview_test(model, dataloader, device, save_folder, show_img=False, make_gif=False, make_vid=False):
    batches_per_scene = 15
    model.to(device)
    model.eval()
    metrics = {}
    with torch.no_grad():
        for iter_idx, (images_batch, proj_mats_batch, joints_3d_gt_batch, joints_3d_valid_batch) in enumerate(dataloader):
            if images_batch is None:
                continue

            images_batch = images_batch.to(device)
            proj_mats_batch = proj_mats_batch.to(device)
            joints_3d_gt_batch = joints_3d_gt_batch.to(device)
            joints_3d_valid_batch = joints_3d_valid_batch.to(device)
            
            joints_3d_pred, joints_2d_pred, heatmaps_pred, confidences_pred = model(images_batch, proj_mats_batch)
            
            if iter_idx % batches_per_scene == 0:
                imgs_folder = os.path.join('imgs', 'scene_%03d' % (iter_idx // batches_per_scene + 1))
                imgs_folder = os.path.join(save_folder, imgs_folder)
                if not os.path.exists(imgs_folder):
                    os.makedirs(imgs_folder)

                # reset per scene error and sample counter to zero
                error_per_scene = 0
                sample_counter = 0

            batch_size = images_batch.shape[0]
            for idx_in_batch in range(batch_size):
                vis_img = visualize.visualize_pred(images_batch[idx_in_batch, ::], proj_mats_batch[idx_in_batch, ::], \
                                                   joints_3d_gt_batch[idx_in_batch, ::], joints_3d_pred[idx_in_batch, ::])
                im = Image.fromarray(vis_img)
                if show_img:
                    im.show()
                # img_name = "%02d.png" % (iter_idx % subj_num_batches)
                img_name = "%06d.png" % (iter_idx % batches_per_scene * batch_size + idx_in_batch)
                img_path = os.path.join(imgs_folder, img_name)
                im.save(img_path)

            # evaluate
            error_batch = evaluate(joints_3d_pred, joints_3d_gt_batch, joints_3d_valid_batch)
            error_per_scene += error_batch.sum()
            sample_counter += error_batch.shape[0]

            if iter_idx % batches_per_scene == batches_per_scene - 1:
                error_mean = error_per_scene / sample_counter
                print(error_mean)
                metrics['scene_%03d' % (iter_idx // batches_per_scene + 1)] = float(error_mean.detach().cpu().numpy())
                
    metrics_path = os.path.join(save_folder, 'metrics.json')
    with open(metrics_path, 'w') as metrics_json:
        json.dump(metrics, metrics_json)

    imgs_base_folder = os.path.join(save_folder, 'imgs')
    if make_gif:
        gif_folder = os.path.join(save_folder, 'gifs')
        if not os.path.exists(gif_folder):
            os.mkdir(gif_folder)
        for scene_name in os.listdir(imgs_base_folder):
            imgs_folder = os.path.join(imgs_base_folder, scene_name)
            gif_name = os.path.join(gif_folder, '%s.gif' % scene_name)
            visualize.make_gif(imgs_folder, gif_name)

    if make_vid:
        vid_folder = os.path.join(save_folder, 'vids')
        if not os.path.exists(vid_folder):
            os.mkdir(vid_folder)
        for scene_name in os.listdir(imgs_base_folder):
            imgs_folder = os.path.join(imgs_base_folder, scene_name)
            vid_name = os.path.join(vid_folder, '%s.mp4' % scene_name)
            visualize.make_vid(imgs_folder, vid_name)
    

if __name__ == "__main__":

    device = torch.device(0)
    
    config = cfg.load_config('experiments/syn_data/multiview_data_2_alg.yaml')

    model = AlgebraicTriangulationNet(config, device=device).to(device)

    state_dict = torch.load(config.model.checkpoint)
    for key in list(state_dict.keys()):
        new_key = key.replace("module.", "")
        state_dict[new_key] = state_dict.pop(key)

    model.load_state_dict(state_dict, strict=True)

    print("Loading data..")
    data_path = 'data/test_01/multiview_data'
    dataset = MultiView_SynData(data_path, invalid_joints=(9, 16), bbox=[80, 0, 560, 480], ori_form=1)
    dataloader = datasets_utils.syndata_loader(dataset, batch_size=4)

    save_folder = os.path.join(os.getcwd(), 'results/test_01')
    multiview_test(model, dataloader, device, save_folder, make_vid=True)
