import os, json
import argparse
import pickle
from collections import defaultdict
import numpy as np
import torch
from PIL import Image

from utils import cfg
from utils.eval import *
from models.triangulation import AlgebraicTriangulationNet
from models.loss import PCK, KeypointsL2Loss
from datasets.multiview_syndata import MultiView_SynData
from datasets.human36m import Human36MMultiViewDataset
import datasets.utils as datasets_utils

import utils.visualize as visualize

import train


def test_one_epoch(model, val_loader, metric, device):
    model.eval()
    with torch.no_grad():
        total_samples = 0
        total_error = 0
        total_detected = 0
        for iter_idx, (images_batch, proj_mats_batch, joints_3d_gt_batch, joints_3d_valid_batch, info_batch) in enumerate(val_loader):
            if images_batch is None:
                continue
                    
            images_batch = images_batch.to(device)
            proj_mats_batch = proj_mats_batch.to(device)
            joints_3d_gt_batch = joints_3d_gt_batch.to(device)
            joints_3d_valid_batch = joints_3d_valid_batch.to(device)
            batch_size = images_batch.shape[0]
            joints_3d_pred, joints_2d_pred, heatmaps_pred, confidences_pred = model(images_batch, proj_mats_batch)

            if isinstance(metric, PCK):
                detected, num_samples = metric(joints_2d_pred, proj_mats_batch, \
                                               joints_3d_gt_batch, joints_3d_valid_batch)
                total_detected += detected
                total_samples += num_samples
                # print(detected, num_samples)
                
            elif isinstance(metric, KeypointsL2Loss):
                error = metric(joints_3d_pred, joints_3d_gt_batch, joints_3d_valid_batch)
                total_error += batch_size * error.item()
                total_samples += batch_size

        pck_acc = total_detected / total_samples # 2D
        mean_error = total_error / total_samples # 3D

    return pck_acc, mean_error


def syndata_test(model, dataloader, device, save_folder, show_img=False, make_gif=False, make_vid=False):
    frames_per_scene = 60
    
    model.to(device)
    model.eval()

    #scene_names = []
    subj_names = []
    anim_names = []
    metrics = {}
    with torch.no_grad():
        for iter_idx, (images_batch, proj_mats_batch, joints_3d_gt_batch, joints_3d_valid_batch, info_batch) in enumerate(dataloader):
            print(iter_idx)
            if iter_idx >= 30:
                break

            if images_batch is None:
                continue

            images_batch = images_batch.to(device)
            proj_mats_batch = proj_mats_batch.to(device)
            joints_3d_gt_batch = joints_3d_gt_batch.to(device)
            joints_3d_valid_batch = joints_3d_valid_batch.to(device)

            batch_size = images_batch.shape[0]
            
            joints_3d_pred, joints_2d_pred, heatmaps_pred, confidences_pred = model(images_batch, proj_mats_batch)

            [subj_name, anim_idx, frame] = info_batch[0]
            if frame == 0:
                anim_name = 'anim_%03d' % anim_idx
                print(subj_name, anim_name)
                if subj_name not in subj_names:
                    subj_names.append(subj_name)
                if anim_name not in anim_names:
                    anim_names.append(anim_name)
                
                joints_3d_pred_np = np.empty([0,] + list(joints_3d_pred.detach().cpu().numpy().shape[1::]))
                joints_2d_pred_np = np.empty([0,] + list(joints_2d_pred.detach().cpu().numpy().shape[1::]))
                heatmaps_pred_np = np.empty([0,] + list(heatmaps_pred.detach().cpu().numpy().shape[1::]))
                confidences_pred_np = np.empty([0,] + list(confidences_pred.detach().cpu().numpy().shape[1::]))

                preds_folder = os.path.join(save_folder, 'preds', subj_name, anim_name)
                os.makedirs(preds_folder, exist_ok=True)

            joints_3d_pred_np = np.concatenate((joints_3d_pred_np, joints_3d_pred.detach().cpu().numpy()), axis=0)
            joints_2d_pred_np = np.concatenate((joints_2d_pred_np, joints_2d_pred.detach().cpu().numpy()), axis=0)
            heatmaps_pred_np = np.concatenate((heatmaps_pred_np, heatmaps_pred.detach().cpu().numpy()), axis=0)
            confidences_pred_np = np.concatenate((confidences_pred_np, confidences_pred.detach().cpu().numpy()), axis=0)

            if (frame + batch_size) == frames_per_scene:
                # save intermediate results
                print('saving intermediate results...')
                np.save(os.path.join(preds_folder, 'joints_3d.npy'), joints_3d_pred_np) # numpy array of size (num_frames, num_joints, 3)
                np.save(os.path.join(preds_folder, 'joints_2d.npy'), joints_2d_pred_np) # numpy array of size (num_frames, num_views, num_joints, 2)
                #np.save(os.path.join(preds_folder, 'heatmaps.npy'), heatmaps_pred_np) # numpy array of size (num_frames, num_views, num_joints, 120, 120)
                np.save(os.path.join(preds_folder, 'confidences.npy'), confidences_pred_np) # numpy array of size (num_frames, num_views, num_joints)

    # save evaluations and visualizations
    for subj_name in subj_names:
        metrics_subj = {}
        for anim_name in anim_names:
            scene_folder = os.path.join(dataloader.dataset.basepath, subj_name, anim_name) # subj currently hardcoded
            if not os.path.exists(scene_folder):
                continue

            # evaluate
            joints_3d_pred_path = os.path.join(save_folder, 'preds', subj_name, anim_name, 'joints_3d.npy')
            joints_2d_pred_path = os.path.join(save_folder, 'preds', subj_name, anim_name, 'joints_2d.npy')
            error_per_scene = evaluate_one_scene(joints_3d_pred_path, scene_folder, invalid_joints=(9, 16))
            metrics_subj[anim_name] = error_per_scene

            # save images
            print('saving result images...')
            imgs_folder = os.path.join(save_folder, 'imgs', subj_name, anim_name)
            visualize.draw_one_scene(joints_3d_pred_path, joints_2d_pred_path, scene_folder, imgs_folder, show_img=show_img)

            # save gifs/videos (optioanl)
            if make_gif:
                print('saving result gifs...')
                gif_name = os.path.join(save_folder, 'gifs', subj_name, '%s.gif' % anim_name)
                visualize.make_gif(imgs_folder, gif_name)

            if make_vid:
                print('saving result videos...')
                vid_name = os.path.join(save_folder, 'vids', subj_name, '%s.mp4' % anim_name)
                visualize.make_vid(imgs_folder, vid_name)

        metrics[subj_name] = metrics_subj
            
    # save evaluation
    print('saving evaluation results...')
    metrics_path = os.path.join(save_folder, 'metrics.json')
    with open(metrics_path, 'w') as metrics_json:
        json.dump(metrics, metrics_json)


def human36m_test(model, dataloader, device, save_folder, show_img=False, make_gif=False, make_vid=False):
    os.makedirs(save_folder, exist_ok=True)
    saveimg_per_iter = 10
    
    model.to(device)
    model.eval()

    preds = defaultdict(list)

    with torch.no_grad():
        for iter_idx, (images_batch, proj_mats_batch, joints_3d_gt_batch, joints_3d_valid_batch, indexes) in enumerate(dataloader):
            print(iter_idx)
            if iter_idx >= 60:
                break

            if images_batch is None:
                continue

            images_batch = images_batch.to(device)
            proj_mats_batch = proj_mats_batch.to(device)
            joints_3d_gt_batch = joints_3d_gt_batch.to(device)
            joints_3d_valid_batch = joints_3d_valid_batch.to(device)

            batch_size = images_batch.shape[0]
            
            joints_3d_pred, joints_2d_pred, heatmaps_pred, confidences_pred = model(images_batch, proj_mats_batch)

            preds["indexes"].append(indexes)
            preds["joints_3d"].append(joints_3d_pred.detach().cpu().numpy())
            preds["joints_2d"].append(joints_2d_pred.detach().cpu().numpy())
            preds["confidences"].append(confidences_pred.detach().cpu().numpy())

            # save images
            imgs_folder = os.path.join(save_folder, "imgs")
            os.makedirs(imgs_folder, exist_ok=True)
            if iter_idx % saveimg_per_iter == 0:
                img = visualize.visualize_pred(images_batch[0], proj_mats_batch[0], joints_3d_gt_batch[0], joints_3d_pred[0], joints_2d_pred[0])
                im = Image.fromarray(img)
                img_path = os.path.join(imgs_folder, "%06d.png" % indexes[0])
                im.save(img_path)

    preds["indexes"] = np.concatenate(preds["indexes"])
    preds["joints_3d"] = np.concatenate(preds["joints_3d"], axis=0)
    preds["joints_2d"] = np.concatenate(preds["joints_2d"], axis=0)
    preds["confidences"] = np.concatenate(preds["confidences"], axis=0)

    print('saving prediction results...')
    preds_path = os.path.join(save_folder, "preds.pkl")
    with open(preds_path, 'wb') as pkl_file:
        pickle.dump(preds, pkl_file)


    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="experiments/syndata/test/syndata_alg_17jnts.yaml")
    args = parser.parse_args()

    config = cfg.load_config(args.config)

    assert config.dataset.type in ("syndata", "human36m")

    device = torch.device(int(config.gpu_id))

    model = AlgebraicTriangulationNet(config, device=device).to(device)

    if config.model.init_weights:
        model = train.load_pretrained_model(model, config)
    
    print("Loading data..")

    if args.data == "syndata":
        dataset = MultiView_SynData(config.dataset.data_root, load_joints=config.model.backbone.num_joints, invalid_joints=(9, 16), \
                                    bbox=config.dataset.bbox, image_shape=config.dataset.image_shape)
        dataloader = datasets_utils.syndata_loader(dataset, \
                                                   batch_size=config.dataset.test.batch_size, \
                                                   shuffle=config.dataset.test.shuffle, \
                                                   num_workers=config.dataset.test.num_workers)

        save_folder = os.path.join(os.getcwd(), 'results/mocap_syndata_%djnts' % config.model.backbone.num_joints)
        #save_folder = os.path.join(os.getcwd(), 'results/mocap_syndata')
        syndata_test(model, dataloader, device, save_folder, make_vid=False)

    elif args.data == "human36m":
        dataset = Human36MMultiViewDataset(
                    h36m_root=config.dataset.data_root,
                    test=True,
                    image_shape=config.dataset.image_shape,
                    labels_path=config.dataset.labels_path,
                    with_damaged_actions=config.dataset.test.with_damaged_actions,
                    retain_every_n_frames_in_test=config.dataset.test.retain_every_n_frames_in_test,
                    scale_bbox=config.dataset.test.scale_bbox,
                    kind="human36m",
                    undistort_images=config.dataset.test.undistort_images,
                    ignore_cameras=config.dataset.test.ignore_cameras if hasattr(config.dataset.val, "ignore_cameras") else [],
                    crop=config.dataset.test.crop if hasattr(config.dataset.val, "crop") else True,
                )
        dataloader = datasets_utils.human36m_loader(dataset, \
                                                    batch_size=config.dataset.test.batch_size, \
                                                    shuffle=config.dataset.test.shuffle, \
                                                    num_workers=config.dataset.test.num_workers)

        save_folder = os.path.join(os.getcwd(), 'results/human36m_%djnts' % config.model.backbone.num_joints)
        human36m_test(model, dataloader, device, save_folder, make_vid=False)

