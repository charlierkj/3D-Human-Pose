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


def test_one_epoch(model, val_loader, metric, device):
    model.eval()
    with torch.no_grad():
        total_samples = 0
        total_error = 0
        total_detected = 0
        total_detected_per_joint = torch.zeros((17, )) # hardcoded
        total_num_per_joint = torch.zeros((17, ))
        for iter_idx, (images_batch, proj_mats_batch, joints_3d_gt_batch, joints_3d_valid_batch, joints_2d_gt_batch, info_batch) \
            in enumerate(val_loader):
            
            if images_batch is None:
                continue
                    
            images_batch = images_batch.to(device)
            if proj_mats_batch is not None:
                proj_mats_batch = proj_mats_batch.to(device)
                joints_3d_gt_batch = joints_3d_gt_batch.to(device)
                joints_3d_valid_batch = joints_3d_valid_batch.to(device)
            joints_2d_gt_batch = joints_2d_gt_batch.to(device)
            batch_size = images_batch.shape[0]
            joints_3d_pred, joints_2d_pred, heatmaps_pred, confidences_pred = model(images_batch, proj_mats_batch)

            detected, error, num_samples, detected_per_joint, num_per_joint\
                      = utils_eval.eval_one_batch(metric, joints_3d_pred, joints_2d_pred, \
                                                  proj_mats_batch, joints_3d_gt_batch, joints_3d_valid_batch, \
                                                  joints_2d_gt_batch)

            total_detected += detected
            total_error += num_samples * error
            total_samples += num_samples

            total_detected_per_joint += detected_per_joint
            total_num_per_joint += num_per_joint

        pck_acc = total_detected / total_samples # 2D
        mean_error = total_error / total_samples # 3D

        pck_acc_per_joint = total_detected_per_joint / total_num_per_joint # 2D per joint

    return pck_acc, mean_error, pck_acc_per_joint


def syndata_test(config, model, dataloader, device, save_folder, \
                 save_img=False, show_img=False, make_gif=False, make_vid=False):
    frames_per_scene = 60

    os.makedirs(save_folder, exist_ok=True)

    # save parameters
    yaml_path = os.path.join(save_folder, "param.yaml")
    param_dict = {}
    param_dict["dataset"] = config.dataset.type
    param_dict["test_batch_size"] = config.dataset.test.batch_size
    param_dict["image_shape"] = config.dataset.image_shape
    param_dict["bbox"] = config.dataset.bbox
    param_dict["num_joints"] = config.model.backbone.num_joints
    if config.model.init_weights:
        param_dict["checkpoint"] = config.model.checkpoint
    else:
        param_dict["checkpoint"] = ""
    param_dict["use_confidences"] = config.model.use_confidences
    param_dict["heatmap_multiplier"] = config.model.heatmap_multiplier
    with open(yaml_path, 'w') as f:
        data = yaml.dump(param_dict, f)

    # model
    model.to(device)
    model.eval()

    # metrics
    metric_pck = PCK()
    metric_pckh = PCKh()
    metric_pck3d = PCK3D()
    metric_error = KeypointsL2Loss()

    #scene_names = []
    subj_names = []
    anim_names = []
    metrics = {}
    with torch.no_grad():
        total_samples = 0
        total_joints_3d = 0
        total_joints_2d = 0
        total_detected_pck = 0
        total_detected_pckh = 0
        total_detected_pck3d = 0
        total_error = 0
        
        for iter_idx, (images_batch, proj_mats_batch, joints_3d_gt_batch, joints_3d_valid_batch, info_batch) in enumerate(dataloader):
            # print(iter_idx)

            if images_batch is None:
                continue

            images_batch = images_batch.to(device)
            proj_mats_batch = proj_mats_batch.to(device)
            joints_3d_gt_batch = joints_3d_gt_batch.to(device)
            joints_3d_valid_batch = joints_3d_valid_batch.to(device)

            batch_size = images_batch.shape[0]
            
            joints_3d_pred, joints_2d_pred, heatmaps_pred, confidences_pred = model(images_batch, proj_mats_batch)

            # evaluate
            detected_pck, num_joints_2d = metric_pck(joints_2d_pred, proj_mats_batch, \
                                                     joints_3d_gt_batch, joints_3d_valid_batch)
            detected_pckh, _ = metric_pckh(joints_2d_pred, proj_mats_batch, \
                                           joints_3d_gt_batch, joints_3d_valid_batch)
            detected_pck3d, num_joints_3d = metric_pck3d(joints_3d_pred, \
                                                         joints_3d_gt_batch, joints_3d_valid_batch)
            error = metric_error(joints_3d_pred, joints_3d_gt_batch, joints_3d_valid_batch).item()
            
            total_samples += batch_size
            total_joints_3d += num_joints_3d
            total_joints_2d += num_joints_2d
            total_detected_pck += detected_pck
            total_detected_pckh += detected_pckh
            total_detected_pck3d += detected_pck3d
            total_error += error * batch_size

            # save predictions
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

    """
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
            # error_per_scene = evaluate_one_scene(joints_3d_pred_path, scene_folder, invalid_joints=(9, 16))
            # metrics_subj[anim_name] = error_per_scene

            # save images
            if save_img:
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
    """
            
    # save evaluations
    pck_acc = float((total_detected_pck / total_joints_2d).detach().cpu())
    pckh_acc = float((total_detected_pckh / total_joints_2d).detach().cpu())
    pck3d_acc = float((total_detected_pck3d / total_joints_3d).detach().cpu())
    mean_error = float(total_error / total_samples)

    print("PCK:", pck_acc)
    print("PCKh:", pckh_acc)
    print("PCK3D:", pck3d_acc)
    print("Error:", mean_error)

    metrics["PCK@%.1f" % metric_pck.thresh] = pck_acc
    metrics["PCKh@%.1f" % metric_pckh.thresh] = pckh_acc
    metrics["PCK3D@%d" % metric_pck3d.thresh] = pck3d_acc
    metrics["error"] = mean_error

    print('saving evaluations...')
    metrics_path = os.path.join(save_folder, 'metrics.json')
    with open(metrics_path, 'w') as metrics_json:
        json.dump(metrics, metrics_json)
        
    # print('saving evaluation results...')
    # metrics_path = os.path.join(save_folder, 'metrics.json')
    # with open(metrics_path, 'w') as metrics_json:
    #     json.dump(metrics, metrics_json)


def real_test(config, model, dataloader, device, save_folder, \
                  save_img=False, show_img=False, make_gif=False, make_vid=False):
    saveimg_per_iter = 10

    os.makedirs(save_folder, exist_ok=True)

    # save parameters
    yaml_path = os.path.join(save_folder, "param.yaml")
    param_dict = {}
    param_dict["dataset"] = config.dataset.type
    param_dict["test_batch_size"] = config.dataset.test.batch_size
    param_dict["image_shape"] = config.dataset.image_shape
    param_dict["scale_bbox"] = config.dataset.test.scale_bbox
    param_dict["num_joints"] = config.model.backbone.num_joints
    if config.model.init_weights:
        param_dict["checkpoint"] = config.model.checkpoint
    else:
        param_dict["checkpoint"] = ""
    param_dict["use_confidences"] = config.model.use_confidences
    param_dict["heatmap_multiplier"] = config.model.heatmap_multiplier
    with open(yaml_path, 'w') as f:
        data = yaml.dump(param_dict, f)

    # model
    model.to(device)
    model.eval()

    # metrics
    metric_pck = PCK()
    metric_pckh = PCKh()
    metric_pck3d = PCK3D()
    metric_error = KeypointsL2Loss()

    preds = defaultdict(list)

    with torch.no_grad():
        total_samples = 0
        total_joints_3d = 0
        total_joints_2d = 0
        total_detected_pck = 0
        total_detected_pckh = 0
        total_detected_pck3d = 0
        total_error = 0
        
        for iter_idx, (images_batch, proj_mats_batch, joints_3d_gt_batch, joints_3d_valid_batch, joints_2d_gt_batch, indexes) in enumerate(dataloader):
            # print(iter_idx)

            if images_batch is None:
                continue

            images_batch = images_batch.to(device)

            if proj_mats_batch is not None:
                proj_mats_batch = proj_mats_batch.to(device)
                joints_3d_gt_batch = joints_3d_gt_batch.to(device)
                joints_3d_valid_batch = joints_3d_valid_batch.to(device)

            batch_size = images_batch.shape[0]

            joints_3d_pred, joints_2d_pred, heatmaps_pred, confidences_pred = model(images_batch, proj_mats_batch)
            print(heatmaps_pred.max())
            preds["indexes"].append(indexes)
            preds["joints_3d"].append(joints_3d_pred.detach().cpu().numpy())
            preds["joints_2d"].append(joints_2d_pred.detach().cpu().numpy())
            preds["confidences"].append(confidences_pred.detach().cpu().numpy())

            # save images
            if save_img:
                imgs_folder = os.path.join(save_folder, "imgs")
                os.makedirs(imgs_folder, exist_ok=True)
                if iter_idx % saveimg_per_iter == 0:
                    # joints plot
                    img_jnt = visualize.visualize_pred(images_batch[0], proj_mats_batch[0], joints_3d_gt_batch[0], joints_3d_pred[0], joints_2d_pred[0])
                    im_jnt = Image.fromarray(img_jnt)
                    img_path = os.path.join(imgs_folder, "joints_%06d.png" % indexes[0])
                    im_jnt.save(img_path)

                    # heatmaps plot
                    vis_joint = (iter_idx // saveimg_per_iter) % 17
                    img_hm = visualize.visualize_heatmap(images_batch[0], joints_2d_gt_batch[0], \
                            heatmaps_pred[0], vis_joint=vis_joint)
                    im_hm = Image.fromarray(img_hm)
                    img_path = os.path.join(imgs_folder, "heatmap_%06d.png" % indexes[0])
                    im_hm.save(img_path)

            # evaluate
            detected_pck, num_joints_2d = metric_pck(joints_2d_pred, proj_mats_batch, \
                                                     joints_3d_gt_batch, joints_3d_valid_batch)
            detected_pckh, _ = metric_pckh(joints_2d_pred, proj_mats_batch, \
                                           joints_3d_gt_batch, joints_3d_valid_batch)
            detected_pck3d, num_joints_3d = metric_pck3d(joints_3d_pred, \
                                                         joints_3d_gt_batch, joints_3d_valid_batch)
            error = metric_error(joints_3d_pred, joints_3d_gt_batch, joints_3d_valid_batch).item()
            
            total_samples += batch_size
            total_joints_3d += num_joints_3d
            total_joints_2d += num_joints_2d
            total_detected_pck += detected_pck
            total_detected_pckh += detected_pckh
            total_detected_pck3d += detected_pck3d
            total_error += error * batch_size

    # save evaluations
    pck_acc = float((total_detected_pck / total_joints_2d).detach().cpu())
    pckh_acc = float((total_detected_pckh / total_joints_2d).detach().cpu())
    pck3d_acc = float((total_detected_pck3d / total_joints_3d).detach().cpu())
    mean_error = float(total_error / total_samples)

    print("PCK:", pck_acc)
    print("PCKh:", pckh_acc)
    print("PCK3D:", pck3d_acc)
    print("Error:", mean_error)

    metrics = {}
    metrics["PCK@%.1f" % metric_pck.thresh] = pck_acc
    metrics["PCKh@%.1f" % metric_pckh.thresh] = pckh_acc
    metrics["PCK3D@%d" % metric_pck3d.thresh] = pck3d_acc
    metrics["error"] = mean_error

    print('saving evaluations...')
    metrics_path = os.path.join(save_folder, 'metrics.json')
    with open(metrics_path, 'w') as metrics_json:
        json.dump(metrics, metrics_json)
    
    # save predictions
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
    parser.add_argument('--save_folder', type=str, default="results/syndata")
    parser.add_argument('--save_img', action='store_true')
    args = parser.parse_args()

    config = cfg.load_config(args.config)

    assert config.dataset.type in ("syndata", "human36m")

    device = torch.device(int(config.gpu_id))
    print(device)

    model = AlgebraicTriangulationNet(config, device=device)

    model = torch.nn.DataParallel(model, device_ids=[int(config.gpu_id)])

    if config.model.init_weights:
        print("Initializing model weights..")
        model = train.load_pretrained_model(model, config)

    # load data
    print("Loading data..")
    save_folder = os.path.join(os.getcwd(), args.save_folder)
    if config.dataset.type == "syndata":
        dataset = MultiView_SynData(config.dataset.data_root, load_joints=config.model.backbone.num_joints, invalid_joints=(), \
                                    bbox=config.dataset.bbox, image_shape=config.dataset.image_shape, \
                                    test=True)
        dataloader = datasets_utils.syndata_loader(dataset, \
                                                   batch_size=config.dataset.test.batch_size, \
                                                   shuffle=config.dataset.test.shuffle, \
                                                   num_workers=config.dataset.test.num_workers)

        # save_folder = os.path.join(os.getcwd(), 'results/mocap_syndata_%djnts' % config.model.backbone.num_joints)
        #save_folder = os.path.join(os.getcwd(), 'results/mocap_syndata')
        syndata_test(config, model, dataloader, device, save_folder, \
                     save_img=args.save_img, make_vid=False)

    elif config.dataset.type == "human36m":
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
                )
        dataloader = datasets_utils.human36m_loader(dataset, \
                                                    batch_size=config.dataset.test.batch_size, \
                                                    shuffle=config.dataset.test.shuffle, \
                                                    num_workers=config.dataset.test.num_workers)

        # save_folder = os.path.join(os.getcwd(), 'results/human36m_%djnts' % config.model.backbone.num_joints)
        human36m_test(config, model, dataloader, device, save_folder, \
                      save_img=args.save_img, make_vid=False)

