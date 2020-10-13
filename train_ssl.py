import os
import numpy as np
import argparse

import torch
torch.backends.cudnn.benchmark = True

from torch import nn
from datetime import datetime
from PIL import Image
import yaml

import matplotlib.pyplot as plt
from itertools import cycle
from tensorboardX import SummaryWriter

from utils import cfg
from models.triangulation import AlgebraicTriangulationNet
from models.loss import HeatmapMSELoss, KeypointsMSELoss, KeypointsMSESmoothLoss, KeypointsMAELoss, KeypointsL2Loss
from models.metric import PCK, PCKh, PCK3D
from datasets.multiview_syndata import MultiView_SynData
from datasets.human36m import Human36MMultiViewDataset
from datasets.mpii import Mpii
import datasets.utils as datasets_utils
import utils.visualize as visualize
import train
import test
import utils.eval as utils_eval

import consistency


def train_one_epoch_ssl(config, model, syn_train_loader, real_train_loader, \
                        criterion, metric, opt, e, device, \
                        checkpoint_dir, writer=None, \
                        gamma=10, \
                        log_every_iters=1, vis_every_iters=1):
    model.train()
    batch_size = syn_train_loader.batch_size
    iters_per_epoch = round(min(syn_train_loader.dataset.__len__() / syn_train_loader.batch_size, \
                                real_train_loader.dataset.__len__() / real_train_loader.batch_size))
    print("Estimated iterations per epoch is %d." % iters_per_epoch)
    
    total_train_loss_syn = 0
    total_detected_syn = 0
    total_error_syn = 0
    total_samples_syn = 0 # num_joints or num_frames

    total_train_loss_real = 0
    total_detected_real = 0
    total_error_real = 0
    total_samples_real = 0 # num_joints or num_frames

    # jointly training
    joint_loader = zip(syn_train_loader, real_train_loader)

    for iter_idx, ((syn_images_batch, syn_proj_mats_batch, syn_joints_3d_gt_batch, syn_joints_3d_valid_batch, syn_joints_2d_gt_batch, syn_info_batch), \
                   (real_images_batch, real_proj_mats_batch, real_joints_3d_gt_batch, real_joints_3d_valid_batch, real_joints_2d_gt_batch, real_indexes)) \
                   in enumerate(joint_loader):

        opt.zero_grad()

        # train on syndata
        if syn_images_batch is None:
            continue

        syn_images_batch = syn_images_batch.to(device)
        syn_proj_mats_batch = syn_proj_mats_batch.to(device)
        syn_joints_3d_gt_batch = syn_joints_3d_gt_batch.to(device)
        syn_joints_3d_valid_batch = syn_joints_3d_valid_batch.to(device)
        syn_joints_2d_gt_batch = syn_joints_2d_gt_batch.to(device)

        syn_joints_3d_pred, syn_joints_2d_pred, syn_heatmaps_pred, syn_confidences_pred = model(syn_images_batch, syn_proj_mats_batch)

        if isinstance(criterion, HeatmapMSELoss):
            syn_loss = criterion(syn_heatmaps_pred, syn_joints_2d_gt_batch)
        else:
            syn_loss = criterion(syn_joints_3d_pred, syn_joints_3d_gt_batch, syn_joints_3d_valid_batch)

        # train on h36m
        if real_images_batch is None:
            continue

        real_images_batch = real_images_batch.to(device)
        if real_proj_mats_batch is not None:
            real_proj_mats_batch = real_proj_mats_batch.to(device)
            real_joints_3d_gt_batch = real_joints_3d_gt_batch.to(device)
            real_joints_3d_valid_batch = real_joints_3d_valid_batch.to(device)
        real_joints_2d_gt_batch = real_joints_2d_gt_batch.to(device)

        real_joints_3d_pred, real_joints_2d_pred, real_heatmaps_pred, real_confidences_pred = model(real_images_batch, real_proj_mats_batch)

        pseudo_labels = np.load("pseudo_labels/%s_train.npy" % config.dataset.type, allow_pickle=True).item() # load pseudo labels
        p = 0.2 * (e // 10 + 1) # percentage
        score_thresh = consistency.get_score_thresh(pseudo_labels, p, separate=True)
        real_joints_2d_pl_batch, real_joints_2d_valid_batch = \
                                 consistency.get_pseudo_labels(pseudo_labels, real_indexes, real_images_batch.shape[1], score_thresh)
        real_joints_2d_pl_batch = real_joints_2d_pl_batch.to(device)
        real_joints_2d_valid_batch = real_joints_2d_valid_batch.to(device)

        """
        # debug
        if iter_idx == 0:
            for batch_idx in range(h36m_joints_2d_gt_batch.shape[0]):
                for view_idx in range(h36m_joints_2d_gt_batch.shape[1]):
                    for j in range(h36m_joints_2d_gt_batch.shape[2]):
                        plt.imshow(h36m_images_batch[batch_idx, view_idx, 0, :, :].detach().cpu().numpy())
                        plt.scatter(h36m_joints_2d_gt_batch[batch_idx, view_idx, j, 0].detach().cpu().numpy(), \
                                h36m_joints_2d_gt_batch[batch_idx, view_idx, j, 1].detach().cpu().numpy(), \
                                s=10, color="red")
                        plt.xlabel("%s" \
                                % h36m_joints_2d_valid_batch[batch_idx, view_idx, j, 0].detach().cpu().numpy())
                        plt.savefig("hm/%d_%d_%d.png" % (batch_idx, view_idx, j))
                        plt.close()
        """

        if isinstance(criterion, HeatmapMSELoss):
            real_loss = criterion(real_heatmaps_pred, real_joints_2d_pl_batch, real_joints_2d_valid_batch)
        else:
            raise ValueError("Please use 2D Heatmap Loss for training on real dataset!")

        # optimize
        loss = syn_loss + gamma * real_loss
        loss.backward()
        opt.step()

        # evaluate on syndata
        syn_detected, syn_error, syn_num_samples = utils_eval.eval_one_batch(metric, syn_joints_3d_pred, syn_joints_2d_pred, \
                                                                             syn_proj_mats_batch, syn_joints_3d_gt_batch, syn_joints_3d_valid_batch, \
                                                                             syn_joints_2d_gt_batch)

        total_train_loss_syn += syn_num_samples * syn_loss.item()
        total_detected_syn += syn_detected
        total_error_syn += syn_num_samples * syn_error
        total_samples_syn += syn_num_samples

        # evaluate on h36m
        real_detected, real_error, real_num_samples = utils_eval.eval_one_batch(metric, real_joints_3d_pred, real_joints_2d_pred, \
                                                                                real_proj_mats_batch, real_joints_3d_gt_batch, real_joints_3d_valid_batch, \
                                                                                real_joints_2d_gt_batch)

        total_train_loss_real += real_num_samples * real_loss.item()
        total_detected_real += real_detected
        total_error_real += real_num_samples * real_error
        total_samples_real += real_num_samples

        # logger
        if iter_idx % log_every_iters == log_every_iters - 1:
            logging_iter = iter_idx + 1 - log_every_iters
            mean_loss_logging_syn = total_train_loss_syn / total_samples_syn
            pck_acc_logging_syn = total_detected_syn / total_samples_syn
            mean_error_logging_syn = total_error_syn / total_samples_syn
            mean_loss_logging_real = total_train_loss_real / total_samples_real
            pck_acc_logging_real = total_detected_real / total_samples_real
            mean_error_logging_real = total_error_real / total_samples_real
            print("epoch: %d, iter: %d" % (e, logging_iter))
            print("        (Syndata) train loss: %f, train acc: %.3f, train error: %.3f" \
                  % (mean_loss_logging_syn, pck_acc_logging_syn, mean_error_logging_syn))
            print("        (Real) train loss: %f, train acc: %.3f, train error: %.3f" \
                  % (mean_loss_logging_real, pck_acc_logging_real, mean_error_logging_real))

            if writer is not None:
                writer.add_scalar("train_loss/syndata/iter", mean_loss_logging_syn, e * iters_per_epoch + logging_iter)
                writer.add_scalar("train_pck/syndata/iter", pck_acc_logging_syn, e * iters_per_epoch + logging_iter)
                writer.add_scalar("train_error/syndata/iter", mean_error_logging_syn, e * iters_per_epoch + logging_iter)
                writer.add_scalar("train_loss/real/iter", mean_loss_logging_real, e * iters_per_epoch + logging_iter)
                writer.add_scalar("train_pck/real/iter", pck_acc_logging_real, e * iters_per_epoch + logging_iter)
                writer.add_scalar("train_error/real/iter", mean_error_logging_real, e * iters_per_epoch + logging_iter)

        # save images
        if iter_idx % vis_every_iters == 0:
            vis_iter = iter_idx
            # visualize first sample in batch
            if writer is not None:
                # joints_vis_syn = visualize.visualize_pred(syn_images_batch[0], syn_proj_mats_batch[0], syn_joints_3d_gt_batch[0], \
                #                                           syn_joints_3d_pred[0], syn_joints_2d_pred[0])
                joints_vis_syn = visualize.visualize_pred_2D(syn_images_batch[0], syn_joints_2d_gt_batch[0], syn_joints_2d_pred[0])
                writer.add_image("joints/syndata/iter", joints_vis_syn.transpose(2, 0, 1), global_step=e*iters_per_epoch+vis_iter)
                # joints_vis_h36m = visualize.visualize_pred(h36m_images_batch[0], h36m_proj_mats_batch[0], h36m_joints_3d_gt_batch[0], \
                #                                            h36m_joints_3d_pred[0], h36m_joints_2d_pred[0])
                joints_vis_real = visualize.visualize_pred_2D(real_images_batch[0], real_joints_2d_gt_batch[0], real_joints_2d_pred[0])
                writer.add_image("joints/real/iter", joints_vis_real.transpose(2, 0, 1), global_step=e*iters_per_epoch+vis_iter)
        
                vis_joint = (iter_idx // vis_every_iters) % 16
                heatmap_vis_syn = visualize.visualize_heatmap(syn_images_batch[0], syn_joints_2d_gt_batch[0], \
                                                              syn_heatmaps_pred[0], vis_joint=vis_joint)
                writer.add_image("heatmap/syndata/joint_%d/iter" % vis_joint, heatmap_vis_syn.transpose(2, 0, 1), global_step=e*iters_per_epoch+vis_iter)
                heatmap_vis_real = visualize.visualize_heatmap(real_images_batch[0], real_joints_2d_gt_batch[0], \
                                                               real_heatmaps_pred[0], vis_joint=vis_joint)
                writer.add_image("heatmap/real/joint_%d/iter" % vis_joint, heatmap_vis_real.transpose(2, 0, 1), global_step=e*iters_per_epoch+vis_iter)

    # save logging per epoch to tensorboard
    mean_loss_syn = total_train_loss_syn / total_samples_syn
    pck_acc_syn = total_detected_syn / total_samples_syn
    mean_error_syn = total_error_syn / total_samples_syn
    mean_loss_real = total_train_loss_real / total_samples_real
    pck_acc_real = total_detected_real / total_samples_real
    mean_error_real = total_error_real / total_samples_real
    if writer is not None:
        writer.add_scalar("train_loss/syndata/epoch",  mean_loss_syn, e)
        writer.add_scalar("train_pck/syndata/epoch", pck_acc_syn, e)
        writer.add_scalar("train_error/syndata/epoch", mean_error_syn, e)
        writer.add_scalar("train_loss/real/epoch",  mean_loss_real, e)
        writer.add_scalar("train_pck/real/epoch", pck_acc_real, e)
        writer.add_scalar("train_error/real/epoch", mean_error_real, e)
        
    return mean_loss_syn, pck_acc_syn, mean_error_syn, \
           mean_loss_real, pck_acc_real, mean_error_real
    

def ssl_train(config, model, syn_train_loader, real_train_loader, val_loader,\
              criterion, opt, epochs, device, \
              gamma=10, \
              log_every_iters=20, vis_every_iters=500, \
              resume=False, logdir="./logs", exp_log="ssl_mpii_17jnts@02.10.2020-20.40.35"):
    # configure dir and writer for saving weights and intermediate evaluation
    if not resume:
        experiment_name = "{}_{}_{}jnts@{}".format("ssl", config.dataset.type, "%d" % config.model.backbone.num_joints, datetime.now().strftime("%d.%m.%Y-%H.%M.%S"))
        start_epoch = 0
    else:
        experiment_name = exp_log
        start_epoch = int(sorted(os.listdir(os.path.join(logdir, experiment_name, "checkpoint")))[-1]) + 1
        
    experiment_dir = os.path.join(logdir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    checkpoint_dir = os.path.join(experiment_dir, "checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(experiment_dir, "tensorboard"))
    yaml_path = os.path.join(experiment_dir, "param.yaml")

    param_dict = {}
    param_dict["dataset"] = config.dataset.type
    param_dict["train_batch_size"] = config.dataset.train.batch_size
    param_dict["test_batch_size"] = config.dataset.test.batch_size
    param_dict["syndata_bbox"] = config.dataset.bbox
    
    if config.dataset.type == "human36m":
        param_dict["h36m_scale_bbox"] = config.dataset.train.scale_bbox
        param_dict["h36m_train_retain_every_n_frames"] = config.dataset.train.retain_every_n_frames
        param_dict["h36m_test_retain_every_n_frames"] = config.dataset.test.retain_every_n_frames
        
    param_dict["image_shape"] = config.dataset.image_shape
    param_dict["num_joints"] = config.model.backbone.num_joints
    param_dict["backbone_num_layers"] = config.model.backbone.num_layers
    param_dict["criterion"] = config.opt.criterion
    param_dict["opt_lr"] = config.opt.lr
    param_dict["epochs"] = config.opt.n_epochs
    param_dict["gamma"] = gamma
    with open(yaml_path, 'w') as f:
        data = yaml.dump(param_dict, f)
    
    model.to(device)

    if isinstance(criterion, HeatmapMSELoss):
        metric = PCK()
    else:
        metric = KeypointsL2Loss()

    for e in range(start_epoch, epochs):
        # generate pseudo labels
        if e % 10 == 0:
            consistency.generate_pseudo_labels(config, model, real_train_loader, device, \
                    num_tfs=5)
        
        # train for one epoch
        train_loss_syn, train_acc_syn, train_error_syn, \
        train_loss_real, train_acc_real, train_error_real \
        = train_one_epoch_ssl(config, model, syn_train_loader, real_train_loader, \
                              criterion, metric, opt, e, device, \
                              checkpoint_dir, writer, gamma, log_every_iters, vis_every_iters)

        # evaluate
        test_acc, test_error = test.test_one_epoch(model, val_loader, metric, device)
        
        writer.add_scalar("test_pck/epoch", test_acc, e)
        writer.add_scalar("test_error/epoch", test_error, e)
        
        print('Epoch: %03d | Train Loss: %f | Train Acc: %.3f | Train Error: %.3f | Test Acc: %.3f | Test Error: %.3f' \
              % (e, train_loss_real, train_acc_real, train_error_real, test_acc, test_error))

        # save weights
        with torch.no_grad():
            checkpoint_dir_e = os.path.join(checkpoint_dir, "%04d" % e)
            os.makedirs(checkpoint_dir_e, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(checkpoint_dir_e, "weights.pth"))

    # save weights
    # torch.save(model.state_dict(), os.path.join(checkpoint_dir, "weights.pth"))
    print("Training Done.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="experiments/human36m/train/human36m_alg_17jnts.yaml")
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--logdir', type=str, default="./logs")
    parser.add_argument('--gamma', type=float, default=10.0)
    args = parser.parse_args()

    config = cfg.load_config(args.config)

    assert config.dataset.type in ("syndata", "human36m", "mpii")

    device = torch.device(int(config.gpu_id))
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = AlgebraicTriangulationNet(config, device=device)

    model = torch.nn.DataParallel(model, device_ids=[int(config.gpu_id)])

    if config.model.init_weights:
        print("Initializing model weights..")
        model = train.load_pretrained_model(model, config)

    # load data
    print("Loading data..")

    # synthetic training data
    syndata_root = "../mocap_syndata/multiview_data"
    syn_train_set = MultiView_SynData(syndata_root, load_joints=config.model.backbone.num_joints, invalid_joints=(), \
                                  bbox=config.dataset.bbox, image_shape=config.dataset.image_shape, \
                                  train=True, with_aug=config.dataset.train.with_aug)
    syn_train_loader = datasets_utils.syndata_loader(syn_train_set, \
                                                 batch_size=config.dataset.train.batch_size, \
                                                 shuffle=config.dataset.train.shuffle, \
                                                 num_workers=config.dataset.train.num_workers)

    if config.dataset.type == "human36m":
        # real training data
        real_train_set = dataset = Human36MMultiViewDataset(
            h36m_root=config.dataset.data_root,
            train=True,
            image_shape=config.dataset.image_shape,
            labels_path=config.dataset.labels_path,
            with_damaged_actions=config.dataset.train.with_damaged_actions,
            retain_every_n_frames=config.dataset.train.retain_every_n_frames,
            scale_bbox=config.dataset.train.scale_bbox,
            kind="human36m",
            undistort_images=config.dataset.train.undistort_images,
            ignore_cameras=config.dataset.train.ignore_cameras if hasattr(config.dataset.train, "ignore_cameras") else [],
            crop=True,
        )
        real_train_loader = datasets_utils.human36m_loader(real_train_set, \
                                                           batch_size=config.dataset.train.batch_size, \
                                                           shuffle=config.dataset.train.shuffle, \
                                                           num_workers=config.dataset.train.num_workers)

        # validating data
        val_set = Human36MMultiViewDataset(
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
        val_loader = datasets_utils.human36m_loader(val_set, \
                                                    batch_size=config.dataset.test.batch_size, \
                                                    shuffle=config.dataset.test.shuffle, \
                                                    num_workers=config.dataset.test.num_workers)

    elif config.dataset.type == "mpii":
        # real training data
        real_train_set = dataset = Mpii(
            image_path=config.dataset.data_root,
            anno_path=config.dataset.labels_path,
            inp_res=config.dataset.image_shape[0],
            out_res=config.dataset.image_shape[0]//4,
            is_train=True
        )
        real_train_loader = datasets_utils.mpii_loader(real_train_set, \
                                                       batch_size=config.dataset.train.batch_size*4, \
                                                       shuffle=config.dataset.train.shuffle, \
                                                       num_workers=config.dataset.train.num_workers)

        # validating data
        val_set = Mpii(
            image_path=config.dataset.data_root,
            anno_path=config.dataset.labels_path,
            inp_res=config.dataset.image_shape[0],
            out_res=config.dataset.image_shape[0]//4,
            is_train=False
        )
        val_loader = datasets_utils.mpii_loader(val_set, \
                                                batch_size=config.dataset.test.batch_size*4, \
                                                shuffle=config.dataset.test.shuffle, \
                                                num_workers=config.dataset.test.num_workers)
        
    # configure loss
    if config.opt.criterion == "MSESmooth":
        criterion = KeypointsMSESmoothLoss(config.opt.mse_smooth_threshold)
    elif config.opt.criterion == "MSE":
        criterion = KeypointsMSELoss()
    elif config.opt.criterion == "MAE":
        criterion = KeypointsMAELoss()
    elif config.opt.criterion == "Heatmap":
        criterion = HeatmapMSELoss(config)

    # configure optimizer
    opt = torch.optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), lr=config.opt.lr)
    # opt = torch.optim.RMSprop(model.parameters(), lr=config.opt.lr)

    ssl_train(config, model, syn_train_loader, real_train_loader, val_loader, \
              criterion, opt, config.opt.n_epochs, device, \
              gamma=args.gamma, \
              resume=args.resume, logdir=args.logdir)
