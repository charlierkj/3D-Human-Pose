import os
import numpy as np
import argparse

import torch
torch.backends.cudnn.benchmark = True

from torch import nn
from datetime import datetime
from PIL import Image
import yaml

from tensorboardX import SummaryWriter

from utils import cfg
from models.triangulation import AlgebraicTriangulationNet
from models.loss import HeatmapMSELoss, KeypointsMSELoss, KeypointsMSESmoothLoss, KeypointsMAELoss, KeypointsL2Loss
from models.metric import PCK, PCKh, PCK3D
from datasets.multiview_syndata import MultiView_SynData
import datasets.utils as datasets_utils
import utils.visualize as visualize
import test
import utils.eval as utils_eval


def train_one_epoch_ssl(model, syn_train_loader, h36m_train_loader, criterion, metric, opt, e, device, \
                        gamma=10, \
                        checkpoint_dir, writer=None, \
                        log_every_iters=1, vis_every_iters=1):
    model.train()
    batch_size = syn_train_loader.batch_size
    iters_per_epoch = round(max(syn_train_loader.dataset.__len__(), h36m_train_loader.dataset.__len__()) / batch_size)
    print("Estimated iterations per epoch is %d." % iters_per_epoch)
    
    total_train_loss_syn = 0
    total_detected_syn = 0
    total_error_syn = 0
    total_samples_syn = 0 # num_joints or num_frames

    total_train_loss_h36m = 0
    total_detected_h36m = 0
    total_error_h36m = 0
    total_samples_h36m = 0 # num_joints or num_frames

    # jointly training
    joint_loader = zip(cycle(syn_train_loader), h36m_train_loader)

    for iter_idx, ((syn_images_batch, syn_proj_mats_batch, syn_joints_3d_gt_batch, syn_joints_3d_valid_batch, syn_info_batch), \
                   (h36m_images_batch, h36m_proj_mats_batch, h36m_joints_3d_gt_batch, h36m_joints_3d_valid_batch, h36m_indexes))in enumerate(joint_loader):

        opt.zero_grad()

        # train on syndata
        if syn_images_batch is None:
            continue

        syn_images_batch = syn_images_batch.to(device)
        syn_proj_mats_batch = syn_proj_mats_batch.to(device)
        syn_joints_3d_gt_batch = syn_joints_3d_gt_batch.to(device)
        syn_joints_3d_valid_batch = syn_joints_3d_valid_batch.to(device)

        syn_joints_3d_pred, syn_joints_2d_pred, syn_heatmaps_pred, syn_confidences_pred = model(syn_images_batch, syn_proj_mats_batch)

        if isinstance(criterion, HeatmapMSELoss):
            syn_loss = criterion(syn_heatmaps_pred, syn_proj_mats_batch, syn_joints_3d_gt_batch, syn_joints_3d_valid_batch)
        else:
            syn_loss = criterion(syn_joints_3d_pred, syn_joints_3d_gt_batch, syn_joints_3d_valid_batch)

        # train on h36m
        if h36m_images_batch is None:
            continue

        h36m_images_batch = h36m_images_batch.to(device)
        h36m_proj_mats_batch = h36m_proj_mats_batch.to(device)
        h36m_joints_3d_gt_batch = h36m_joints_3d_gt_batch.to(device)
        h36m_joints_3d_valid_batch = h36m_joints_3d_valid_batch.to(device)

        h36m_joints_3d_pred, h36m_joints_2d_pred, h36m_heatmaps_pred, h36m_confidences_pred = model(h36m_images_batch, h36m_proj_mats_batch)

        pseudo_labels = np.load("pseudo_labels/human36m_train.npy", allow_pickle=True).item() # load pseudo labels
        p = 0.2 * (e // 2 + 1) # percentage
        score_thresh = consistency.get_score_thresh(pseudo_labels, percentage=p)
        h36m_joints_2d_gt_batch, h36m_joints_2d_valid_batch = consistency.get_pseudo_labels(pseudo_labels, h36m_indexes, score_thresh)
        h36m_joints_2d_gt_batch = h36m_joints_2d_gt_batch.to(device)
        h36m_joints_2d_valid_batch = h36m_joints_2d_valid_batch.to(device)

        if isinstance(criterion, HeatmapMSELoss):
            h36m_loss = criterion(h36m_heatmaps_pred, h36m_proj_mats_batch, h36m_joints_3d_gt_batch, h36m_joints_3d_valid_batch)
        else:
            h36m_loss = criterion(h36m_joints_3d_pred, h36m_joints_3d_gt_batch, h36m_joints_3d_valid_batch)

        # optimize
        loss = syn_loss + gamma * h36m_loss
        loss.backward()
        opt.step()

        # evaluate on syndata
        syn_detected, syn_error, syn_samples = utils_eval.eval_one_batch(metric, syn_joints_3d_pred, syn_joints_2d_pred, \
                                                                         syn_proj_mats_batch, syn_joints_3d_gt_batch, syn_joints_3d_valid_batch)

        total_train_loss_syn += syn_num_samples * syn_loss.item()
        total_detected_syn += syn_detected
        total_error_syn += syn_num_samples * syn_error
        total_samples_syn += syn_num_samples

        # evaluate on h36m
        h36m_detected, h36m_error, h36m_samples = utils_eval.eval_one_batch(metric, h36m_joints_3d_pred, h36m_joints_2d_pred, \
                                                                            h36m_proj_mats_batch, h36m_joints_3d_gt_batch, h36m_joints_3d_valid_batch)

        total_train_loss_h36m += h36m_num_samples * h36m_loss.item()
        total_detected_h36m += h36m_detected
        total_error_h36m += h36m_num_samples * h36m_error
        total_samples_h36m += h36m_num_samples

        # logger
        if iter_idx % log_every_iters == log_every_iters - 1:
            logging_iter = iter_idx + 1 - log_every_iters
            mean_loss_logging_syn = total_train_loss_syn / total_samples_syn
            pck_acc_logging_syn = total_detected_syn / total_samples_syn
            mean_error_logging_syn = total_error_syn / total_samples_syn
            mean_loss_logging_h36m = total_train_loss_h36m / total_samples_h36m
            pck_acc_logging_h36m = total_detected_h36m / total_samples_h36m
            mean_error_logging_h36m = total_error_h36m / total_samples_h36m
            print("epoch: %d, iter: %d" % (e, logging_iter))
            print("        (Syndata) train loss: %f, train acc: %.3f, train error: %.3f" \
                  % (mean_loss_logging_syn, pck_acc_logging_syn, mean_error_logging_syn))
            print("        (Human36m) train loss: %f, train acc: %.3f, train error: %.3f" \
                  % (mean_loss_logging_h36m, pck_acc_logging_h36m, mean_error_logging_h36m))

            if writer is not None:
                writer.add_scalar("train_loss/syndata/iter", mean_loss_logging_syn, e * iters_per_epoch + logging_iter)
                writer.add_scalar("train_pck/syndata/iter", pck_acc_logging_syn, e * iters_per_epoch + logging_iter)
                writer.add_scalar("train_error/syndata/iter", mean_error_logging_syn, e * iters_per_epoch + logging_iter)
                writer.add_scalar("train_loss/h36m/iter", mean_loss_logging_h36m, e * iters_per_epoch + logging_iter)
                writer.add_scalar("train_pck/h36m/iter", pck_acc_logging_h36m, e * iters_per_epoch + logging_iter)
                writer.add_scalar("train_error/h36m/iter", mean_error_logging_h36m, e * iters_per_epoch + logging_iter)

        # save images
        if iter_idx % vis_every_iters == 0:
            vis_iter = iter_idx
            # visualize first sample in batch
            if writer is not None:
                joints_vis_syn = visualize.visualize_pred(syn_images_batch[0], syn_proj_mats_batch[0], syn_joints_3d_gt_batch[0], \
                                                          syn_joints_3d_pred[0], syn_joints_2d_pred[0])
                writer.add_image("joints/syndata/iter", joints_vis_syn.transpose(2, 0, 1), global_step=e*iters_per_epoch+vis_iter)
                joints_vis_h36m = visualize.visualize_pred(h36m_images_batch[0], h36m_proj_mats_batch[0], h36m_joints_3d_gt_batch[0], \
                                                           h36m_joints_3d_pred[0], h36m_joints_2d_pred[0])
                writer.add_image("joints/h36m/iter", joints_vis_h36m.transpose(2, 0, 1), global_step=e*iters_per_epoch+vis_iter)
        
                vis_joint = (iter_idx // vis_every_iters) % 17
                heatmap_vis_syn = visualize.visualize_heatmap(syn_images_batch[0], syn_proj_mats_batch[0], syn_joints_3d_gt_batch[0], \
                                                              syn_heatmaps_pred[0], syn_vis_joint=vis_joint)
                writer.add_image("heatmap/syndata/joint_%d/iter" % vis_joint, heatmap_vis_syn.transpose(2, 0, 1), global_step=e*iters_per_epoch+vis_iter)
                heatmap_vis_h36m = visualize.visualize_heatmap(h36m_images_batch[0], h36m_proj_mats_batch[0], h36m_joints_3d_gt_batch[0], \
                                                               h36m_heatmaps_pred[0], h36m_vis_joint=vis_joint)
                writer.add_image("heatmap/h36m/joint_%d/iter" % vis_joint, heatmap_vis_h36m.transpose(2, 0, 1), global_step=e*iters_per_epoch+vis_iter)

    # save logging per epoch to tensorboard
    mean_loss_syn = total_train_loss_syn / total_samples_syn
    pck_acc_syn = total_detected_syn / total_samples_syn
    mean_error_syn = total_error_syn / total_samples_syn
    mean_loss_h36m = total_train_loss_h36m / total_samples_h36m
    pck_acc_h36m = total_detected_h36m / total_samples_h36m
    mean_error_h36m = total_error_h36m / total_samples_h36m
    if writer is not None:
        writer.add_scalar("train_loss/syndata/epoch",  mean_loss_syn, e)
        writer.add_scalar("train_pck/syndata/epoch", pck_acc_syn, e)
        writer.add_scalar("train_error/syndata/epoch", mean_error_syn, e)
        writer.add_scalar("train_loss/h36m/epoch",  mean_loss_h36m, e)
        writer.add_scalar("train_pck/h36m/epoch", pck_acc_h36m, e)
        writer.add_scalar("train_error/h36m/epoch", mean_error_h36m, e)
        
    return mean_loss_syn, pck_acc_syn, mean_error_syn, \
           mean_loss_h36m, pck_acc_h36m, mean_error_h36m
    

def ssl_train(config, model, syn_train_loader, h36m_train_loader, val_loader, criterion, opt, epochs, device, \
              gamma=10, \
              log_every_iters=20, vis_every_iters=500, \
              resume=False, logdir="./logs", exp_log="exp_27jnts@15.08.2020-05.15.16"):
    # configure dir and writer for saving weights and intermediate evaluation
    if not resume:
        experiment_name = "{}_{}jnts@{}".format("ssl", "%d" % config.model.backbone.num_joints, datetime.now().strftime("%d.%m.%Y-%H.%M.%S"))
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
    param_dict["dataset"] = 'joint'
    param_dict["train_batch_size"] = config.dataset.train.batch_size
    param_dict["test_batch_size"] = config.dataset.test.batch_size
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
        if e % 2 == 0:
            generate_pseudo_labels(config, model, h36m_train_loader, device, \
                                   num_tfs=5)
        
        # train for one epoch
        train_loss_syn, train_acc_syn, train_error_syn, \
        train_loss_h36m, train_acc_h36m, train_error_h36m \
        = train_one_epoch_ssl(model, syn_train_loader, h36m_train_loader, criterion, metric, opt, e, device, gamma\
                              checkpoint_dir, writer, log_every_iters, vis_every_iters)

        # evaluate
        test_acc, test_error = test.test_one_epoch(model, val_loader, metric, device)
        
        writer.add_scalar("test_pck/epoch", test_acc, e)
        writer.add_scalar("test_error/epoch", test_error, e)
        
        print('Epoch: %03d | Train Loss: %f | Train Acc: %.3f | Train Error: %.3f | Test Acc: %.3f | Test Error: %.3f' \
              % (e, train_loss_h36m, train_acc_h36m, train_error_h36m, test_acc, test_error))

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
    parser.add_argument('--config', type=str, default="experiments/syndata/test/syndata_alg_17jnts.yaml")
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--logdir', type=str, default="./logs")
    parser.add_argument('--gamma', type=float, default=10.0)
    args = parser.parse_args()

    config = cfg.load_config(args.config)

    device = torch.device(int(config.gpu_id))
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = AlgebraicTriangulationNet(config, device=device)

    model = torch.nn.DataParallel(model, device_ids=[int(config.gpu_id)])

    if config.model.init_weights:
        print("Initializing model weights..")
        model = load_pretrained_model(model, config)

    # load data
    print("Loading data..")

    # training data
    syn_train_set = MultiView_SynData(config.dataset.data_root, load_joints=config.model.backbone.num_joints, invalid_joints=(), \
                                  bbox=config.dataset.bbox, image_shape=config.dataset.image_shape, \
                                  train=True)
    syn_train_loader = datasets_utils.syndata_loader(train_set, \
                                                 batch_size=config.dataset.train.batch_size, \
                                                 shuffle=config.dataset.train.shuffle, \
                                                 num_workers=config.dataset.train.num_workers)

    h36m_train_set = dataset = Human36MMultiViewDataset(
        h36m_root=config.dataset.data_root,
        train=True,
        image_shape=config.dataset.image_shape,
        labels_path=config.dataset.labels_path,
        with_damaged_actions=config.dataset.train.with_damaged_actions,
        retain_every_n_frames_in_test=config.dataset.test.retain_every_n_frames_in_test,
        scale_bbox=config.dataset.train.scale_bbox,
        kind="human36m",
        undistort_images=config.dataset.train.undistort_images,
        ignore_cameras=config.dataset.train.ignore_cameras if hasattr(config.dataset.train, "ignore_cameras") else [],
        crop=True,
    )
    h36m_train_loader = datasets_utils.human36m_loader(dataset, \
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
        retain_every_n_frames_in_test=config.dataset.test.retain_every_n_frames_in_test,
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

    ssl_train(config, model, syn_train_loader, h36m_train_loader, val_loader, criterion, opt, config.opt.n_epochs, device, \
              gamma=args.gamma, \
              resume=args.resume, logdir=args.logdir)
