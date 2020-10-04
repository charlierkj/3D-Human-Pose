import os, json
import argparse
import pickle
from collections import defaultdict
import numpy as np

import torch
torch.backends.cudnn.benchmark = True

from PIL import Image
import yaml

from utils import cfg
from utils.eval import *
from models.triangulation import AlgebraicTriangulationNet
from models.loss import KeypointsL2Loss
from models.metric import PCK, PCKh, PCK3D
from datasets.multiview_syndata import MultiView_SynData
from datasets.human36m import Human36MMultiViewDataset
import datasets.utils as datasets_utils

import utils.visualize as visualize

import train
import utils.eval as utils_eval


def test_one_scene(model, dataloader, device, save_folder):

    os.makedirs(save_folder, exist_ok=True)

    # model
    model.to(device)
    model.eval()

    with torch.no_grad():
        
        for iter_idx, (images_batch, proj_mats_batch, joints_3d_gt_batch, joints_3d_valid_batch, joints_2d_gt_batch, info_batch) \
            in enumerate(dataloader):

            if images_batch is None:
                continue

            images_batch = images_batch.to(device)
            proj_mats_batch = proj_mats_batch.to(device)
            joints_3d_gt_batch = joints_3d_gt_batch.to(device)
            joints_3d_valid_batch = joints_3d_valid_batch.to(device)
            joints_2d_gt_batch = joints_2d_gt_batch.to(device)

            batch_size = images_batch.shape[0]
            
            joints_3d_pred, joints_2d_pred, heatmaps_pred, confidences_pred = model(images_batch, proj_mats_batch)

            for batch_i in range(batch_size):
                vis = visualize.visualize_pred_2D(images_batch[batch_i], joints_2d_gt_batch[batch_i], joints_2d_pred[batch_i])
                im = Image.fromarray(vis)
                img_path = os.path.join(save_folder, "%06d.png" % iter_idx * batch_size + batch_i)
                im.save(img_path)

    vid_name = os.path.join(save_folder, 'vid.mp4')
    visualize.make_vid(save_folder, vid_name)

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="experiments/human36m/test/human36m_alg_17jnts.yaml")

    config = cfg.load_config(args.config)

    config_1 = config
    config_1.model.checkpoint = "./logs/exp_17jnts@19.09.2020-04.49.14/checkpoint/0019/weights.pth"
    config_2 = config
    config_2.model.checkpoint = "./logs/ssl_17jnts@26.09.2020-20.50.40/checkpoint/0007/weights.pth"

    device = torch.device(0)
    print(device)

    model_1 = AlgebraicTriangulationNet(config_1, device=device)
    model_1 = torch.nn.DataParallel(model_1, device_ids=[0])

    model_2 = AlgebraicTriangulationNet(config_2, device=device)
    model_2 = torch.nn.DataParallel(model_2, device_ids=[0])

    print("Initializing model weights..")
    model_1 = train.load_pretrained_model(model_1, config_1)
    model_2 = train.load_pretrained_model(model_2, config_2)

    # load data
    print("Loading data..")

    for si in [0, 3000, 6000, 9000, 12000]:
        dataset = Human36MMultiViewDataset(
                    h36m_root=config.dataset.data_root,
                    test=True,
                    image_shape=config.dataset.image_shape,
                    labels_path=config.dataset.labels_path,
                    with_damaged_actions=config.dataset.test.with_damaged_actions,
                    retain_every_n_frames=config.dataset.test.retain_every_n_frames,
                    scale_bbox=config.dataset.test.scale_bbox,
                    kind="human36m",
                    undistort_images=config.dataset.test.undistort_images,
                    ignore_cameras=config.dataset.test.ignore_cameras if hasattr(config.dataset.test, "ignore_cameras") else [],
                    crop=True,
                    start_index=si
                    )
        dataloader = datasets_utils.human36m_loader(dataset, \
                                                    batch_size=config.dataset.test.batch_size, \
                                                    shuffle=config.dataset.test.shuffle, \
                                                    num_workers=config.dataset.test.num_workers)

        save_folder_1 = os.path.join('results/ssl_test/before', '%d' % si)
        save_folder_2 = os.path.join('results/ssl_test/after', '%d' % si)
        test_one_scene(model_1, dataloader, device, save_folder_1)
        test_one_scene(model_2, dataloader, device, save_folder_2)

