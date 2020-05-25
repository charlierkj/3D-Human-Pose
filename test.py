import os
import numpy as np
import torch
from PIL import Image

from mvn.utils import cfg
from mvn.models_temp.triangulation import AlgebraicTriangulationNet
from datasets.multiview_syndata import MultiView_SynData
import datasets.utils as datasets_utils

import visualize


def multiview_test(model, dataloader, device, show_img=False, make_gif=False, make_vid=False):
    subj_num_batches = 15
    model.to(device)
    model.eval()
    with torch.no_grad():
        for iter_idx, (images_batch, proj_mats_batch, joints_3d_gt_batch) in enumerate(dataloader):
            if images_batch is None:
                continue

            images_batch = images_batch.to(device)
            proj_mats_batch = proj_mats_batch.to(device)
            joints_3d_gt_batch = joints_3d_gt_batch.to(device)
            
            joints_3d_pred, joints_2d_pred, heatmaps_pred, confidences_pred = model(images_batch, proj_mats_batch)
            
            if iter_idx % subj_num_batches == 0:
                imgs_folder = 'S%d' % (iter_idx // subj_num_batches + 1)
                imgs_folder = os.path.join(os.getcwd(), imgs_folder)
                if not os.path.exists(imgs_folder):
                    os.mkdir(imgs_folder)

            vis_img = visualize.visualize_pred(images_batch[0, ::], proj_mats_batch[0, ::], \
                                               joints_3d_gt_batch[0, ::], joints_3d_pred[0, ::])
            im = Image.fromarray(vis_img)
            if show_img:
                im.show()
            img_name = "%02d.png" % (iter_idx % subj_num_batches)
            img_path = os.path.join(imgs_folder, img_name)
            im.save(img_path)

    if make_gif:
        for subj_idx in range(6):
            gif_folder = os.path.join(os.getcwd(), 'S%d' % (subj_idx + 1))
            gif_name = "S%d.gif" % (subj_idx + 1)
            visualize.make_gif(gif_folder, gif_name)

    if make_vid:
        for subj_idx in range(6):
            vid_folder = os.path.join(os.getcwd(), 'S%d' % (subj_idx + 1))
            vid_name = "S%d.mp4" % (subj_idx + 1)
            visualize.make_vid(vid_folder, vid_name)
    

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
    data_path = 'data/multiview_data_2/multiview_data'
    dataset = MultiView_SynData(data_path, bbox=[80, 0, 560, 480])
    dataloader = datasets_utils.syndata_loader(dataset, batch_size=4)

    multiview_test(model, dataloader, device)
