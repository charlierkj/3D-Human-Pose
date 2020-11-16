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
import consistency

import train
import utils.eval as utils_eval


def test_one_scene_pseudo(pseudo_labels, dataloader, save_folder, start_index):

    os.makedirs(save_folder, exist_ok=True)

    for iter_idx, (images_batch, proj_mats_batch, joints_3d_gt_batch, joints_3d_valid_batch, joints_2d_gt_batch, indexes) \
            in enumerate(dataloader):
        
        if images_batch is None:
            continue
        
        batch_size = images_batch.shape[0]
        indexes_abs = [idx + start_index for idx in indexes]

        p = 0.2 # percentage
        score_thresh = consistency.get_score_thresh(pseudo_labels, p, separate=True)
        joints_2d_pl_batch, joints_2d_valid_batch = \
                                 consistency.get_pseudo_labels(pseudo_labels, indexes_abs, images_batch.shape[1], score_thresh)
        
        for batch_i in range(batch_size):
            vis = visualize.visualize_pseudo_labels(images_batch[batch_i], joints_2d_pl_batch[batch_i], joints_2d_valid_batch[batch_i])
            im = Image.fromarray(vis)
            print(save_folder)
            print("%06d.png" % (iter_idx * batch_size + batch_i))
            img_path = os.path.join(save_folder, "%06d.png" % (iter_idx * batch_size + batch_i))
            im.save(img_path)
    

def test_one_scene_compare(model_before, model_after, dataloader, device, save_folder):

    os.makedirs(save_folder, exist_ok=True)

    # model
    model_before.to(device)
    model_after.to(device)
    model_before.eval()
    model_after.eval()

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
            
            _, joints_2d_pred_before, _, _ = model_before(images_batch, proj_mats_batch)
            _, joints_2d_pred_after, _, _ = model_after(images_batch, proj_mats_batch)

            for batch_i in range(batch_size):
                vis = visualize.visualize_ssl(images_batch[batch_i], joints_2d_pred_before[batch_i], joints_2d_pred_after[batch_i])
                im = Image.fromarray(vis)
                print(save_folder)
                print("%06d.png" % (iter_idx * batch_size + batch_i))
                img_path = os.path.join(save_folder, "%06d.png" % (iter_idx * batch_size + batch_i))
                im.save(img_path)

    vid_name = os.path.join(save_folder, 'vid.mp4')
    visualize.make_vid(save_folder, vid_name)

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="experiments/human36m/test/human36m_alg_17jnts.yaml")
    parser.add_argument('--mode', type=str, required=True, choices=["train", "test"])
    args = parser.parse_args()

    config = cfg.load_config(args.config)
    config.dataset.train.retain_every_n_frames = 10
    # config_1 = cfg.load_config(args.config)
    # config_2 = cfg.load_config(args.config)

    # config_1.model.checkpoint = "./logs/exp_17jnts@19.09.2020-04.49.14/checkpoint/0019/weights.pth"
    # config_2.model.checkpoint = "./logs/ssl_human36m_17jnts@18.10.2020-02.22.10/checkpoint/0019/weights.pth"
    # print(config_1)
    # print(config_2)

    device = torch.device(0)
    print(device)

    # model_1 = AlgebraicTriangulationNet(config_1, device=device)
    # model_1 = torch.nn.DataParallel(model_1, device_ids=[0])

    # model_2 = AlgebraicTriangulationNet(config_2, device=device)
    # model_2 = torch.nn.DataParallel(model_2, device_ids=[0])

    print("Initializing model weights..")
    # model_1 = train.load_pretrained_model(model_1, config_1)
    # model_2 = train.load_pretrained_model(model_2, config_2)

    # load data
    print("Loading data..")
    pseudo_labels = np.load("pseudo_labels/%s_train_every_10_frames.npy" % config.dataset.type, allow_pickle=True).item() # load pseudo labels
    is_train = True if args.mode == "train" else False

    if is_train:
        si_list = [0, 3000, 6000, 9000, 12000, 20000, 40000, 60000, 80000, 100000]
    else:
        si_list = [0, 400, 800, 1200, 1600, 2000]

    # for si in si_list:
    for si in [0, 2000, 4000, 6000, 8000, 10000]:
        dataset = Human36MMultiViewDataset(
                    h36m_root=config.dataset.data_root,
                    train=is_train,
                    test=not is_train,
                    image_shape=config.dataset.image_shape,
                    labels_path=config.dataset.labels_path,
                    with_damaged_actions=config.dataset.train.with_damaged_actions,
                    retain_every_n_frames=config.dataset.train.retain_every_n_frames,
                    scale_bbox=config.dataset.train.scale_bbox,
                    kind="human36m",
                    undistort_images=config.dataset.train.undistort_images,
                    ignore_cameras=config.dataset.train.ignore_cameras if hasattr(config.dataset.train, "ignore_cameras") else [],
                    crop=True,
                    start_index=si
                    )
        dataloader = datasets_utils.human36m_loader(dataset, \
                                                    batch_size=config.dataset.train.batch_size, \
                                                    shuffle=config.dataset.train.shuffle, \
                                                    num_workers=config.dataset.train.num_workers)
         
        # save_folder = os.path.join('results/ssl_test/compare_2/%s' % args.mode, '%d' % si)
        # test_one_scene_compare(model_1, model_2, dataloader, device, save_folder)
        save_folder = 'results/ssl_test/h36m_pseudo_sep/%d' % si
        test_one_scene_pseudo(pseudo_labels, dataloader, save_folder, start_index=si)

