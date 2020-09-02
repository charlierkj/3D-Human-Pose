import os
import numpy as np
import argparse
import torch
from torch import nn
from datetime import datetime
from PIL import Image
import yaml

from tensorboardX import SummaryWriter

from utils import cfg
from models.triangulation import AlgebraicTriangulationNet
from models.loss import HeatmapMSELoss, PCK, KeypointsMSELoss, KeypointsMSESmoothLoss, KeypointsMAELoss, KeypointsL2Loss
from datasets.multiview_syndata import MultiView_SynData
import datasets.utils as datasets_utils
import utils.visualize as visualize
import test


def load_pretrained_model(model, config, init_joints=17):
    device = next(model.parameters()).device
    model_state_dict = model.state_dict()

    pretrained_state_dict = torch.load(config.model.checkpoint)
    for key in list(pretrained_state_dict.keys()):
        new_key = key.replace("module.", "")
        pretrained_state_dict[new_key] = pretrained_state_dict.pop(key)

    new_pretrained_state_dict = {}
    for k, v in pretrained_state_dict.items():
        if k in model_state_dict and v.shape == model_state_dict[k].shape:
            new_pretrained_state_dict[k] = v
        elif k == "backbone.alg_confidences.head.4.weight":
            print("Reiniting alg_confidences layer filters:", k)
            o = torch.zeros_like(model_state_dict[k][:, :])
            nn.init.xavier_uniform_(o)
            n_filters = min(o.shape[0], v.shape[0], init_joints)
            o[:n_filters, :] = v[:n_filters, :]
            new_pretrained_state_dict[k] = o
        elif k == "backbone.alg_confidences.head.4.bias":
            print("Reiniting alg_confidences layer biases:", k)
            o = torch.zeros_like(model_state_dict[k][:])
            nn.init.zeros_(o)
            n_filters = min(o.shape[0], v.shape[0], init_joints)
            o[:n_filters] = v[:n_filters]
            new_pretrained_state_dict[k] = o
        elif k == "backbone.final_layer.weight":  # TODO
            print("Reiniting final layer filters:", k)
            o = torch.zeros_like(model_state_dict[k][:, :, :, :])
            nn.init.xavier_uniform_(o)
            n_filters = min(o.shape[0], v.shape[0], init_joints)
            o[:n_filters, :, :, :] = v[:n_filters, :, :, :]
            new_pretrained_state_dict[k] = o
        elif k == "backbone.final_layer.bias":
            print("Reiniting final layer biases:", k)
            o = torch.zeros_like(model_state_dict[k][:])
            nn.init.zeros_(o)
            n_filters = min(o.shape[0], v.shape[0], init_joints)
            o[:n_filters] = v[:n_filters]
            new_pretrained_state_dict[k] = o

    not_inited_params = set(pretrained_state_dict.keys()) - set(new_pretrained_state_dict.keys())
    if len(not_inited_params) > 0:
        print("Parameters [{}] were not inited".format(not_inited_params))

    model.load_state_dict(new_pretrained_state_dict, strict=True)
    print("Successfully loaded pretrained weights for whole model")
    return model


def train_one_epoch(model, train_loader, criterion, metric, opt, e, device, \
                    checkpoint_dir, writer=None, log_every_iters=1):
    model.train()
    total_samples = train_loader.dataset.__len__()
    batch_size = train_loader.batch_size
    iters_per_epoch = round(total_samples / batch_size)
    print("Estimated iterations per epoch is %d." % iters_per_epoch)
    total_train_loss = 0

    sum_loss_per_log = 0
    sum_samples_per_log = 0

    for iter_idx, (images_batch, proj_mats_batch, joints_3d_gt_batch, joints_3d_valid_batch, info_batch) in enumerate(train_loader):
        if images_batch is None:
            continue

        images_batch = images_batch.to(device)
        proj_mats_batch = proj_mats_batch.to(device)
        joints_3d_gt_batch = joints_3d_gt_batch.to(device)
        joints_3d_valid_batch = joints_3d_valid_batch.to(device)

        batch_size = images_batch.shape[0]

        joints_3d_pred, joints_2d_pred, heatmaps_pred, confidences_pred = model(images_batch, proj_mats_batch)

        # use predictions of invalid joints as groundtruth
        #joints_clone = ~(torch.squeeze(joints_3d_valid_batch, 2).type(torch.bool))
        #joints_3d_gt_batch[joints_clone] = joints_3d_pred[joints_clone].detach().clone()
        #joints_all_valid = torch.ones_like(joints_3d_valid_batch)

        # calculate loss
        if isinstance(criterion, HeatmapMSELoss):
            loss = criterion(heatmaps_pred, proj_mats_batch, joints_3d_gt_batch, joints_3d_valid_batch)
        else:
            loss = criterion(joints_3d_pred, joints_3d_gt_batch, joints_3d_valid_batch)
            
        if iter_idx % log_every_iters == log_every_iters - 1:
            logging_iter = iter_idx + 1 - log_every_iters
            mean_loss_per_log = sum_loss_per_log / sum_samples_per_log
            print("epoch: %d, iter: %d, train loss: %.3f" % (e, logging_iter, mean_loss_per_log))

            if writer is not None:
                writer.add_scalar("train loss (iter)", mean_loss_per_log, e * iters_per_epoch + logging_iter)

            sum_loss_per_log = 0
            sum_samples_per_log = 0
        
        total_train_loss += batch_size * loss.item()
        sum_loss_per_log += batch_size * loss.item()
        sum_samples_per_log += batch_size

        # optimize
        opt.zero_grad()
        loss.backward()
        opt.step()

    # save training loss per epoch to tensorboard
    mean_loss = total_train_loss / total_samples
    if writer is not None:
        writer.add_scalar("train loss (epoch)",  mean_loss, e)

    # evaluate using training set
    with torch.no_grad():
        model.eval()
        pck_acc, mean_error = test.test_one_epoch(model, train_loader, metric, device)
        if writer is not None:
            writer.add_scalar("train pck (epoch)", pck_acc, e)
            writer.add_scalar("train error (epoch)", mean_error, e)
        
    return mean_loss, pck_acc, mean_error
    

def multiview_train(config, model, train_loader, val_loader, criterion, opt, epochs, device, \
                    log_every_iters=100, \
                    resume=False, logdir="./logs", exp_log="exp_27jnts@15.08.2020-05.15.16"):
    # configure dir and writer for saving weights and intermediate evaluation
    if not resume:
        experiment_name = "{}_{}jnts@{}".format("exp", "%d" % config.model.backbone.num_joints, datetime.now().strftime("%d.%m.%Y-%H.%M.%S"))
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
    param_dict["image_shape"] = config.dataset.image_shape
    param_dict["num_joints"] = config.model.backbone.num_joints
    param_dict["backbone_num_layers"] = config.model.backbone.num_layers
    param_dict["criterion"] = config.opt.criterion
    param_dict["opt_lr"] = config.opt.lr
    param_dict["epochs"] = config.opt.n_epochs
    with open(yaml_path, 'w') as f:
        data = yaml.dump(param_dict, f)
    
    model.to(device)

    if isinstance(criterion, HeatmapMSELoss):
        metric = PCK()
    else:
        metric = KeypointsL2Loss()

    for e in range(start_epoch, epochs):
        # train for one epoch
        train_loss, train_acc, train_error = train_one_epoch(model, train_loader, criterion, metric, opt, e, device, \
                                                                      checkpoint_dir, writer, log_every_iters)

        # evaluate
        test_acc, test_error = test.test_one_epoch(model, val_loader, metric, device)
        
        writer.add_scalar("test pck (epoch)", test_acc, e)
        writer.add_scalar("test error (epoch)", test_error, e)
        
        print('Epoch: %03d | Train Loss: %.3f | Train Acc: %.3f | Train Error: %.3f | Test Acc: %.3f | Test Error: %.3f' \
              % (e, train_loss, train_acc, train_error, test_acc, test_error))

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
    train_set = MultiView_SynData(config.dataset.data_root, load_joints=config.model.backbone.num_joints, invalid_joints=(), \
                                  bbox=config.dataset.bbox, image_shape=config.dataset.image_shape, \
                                  train=True)
    train_loader = datasets_utils.syndata_loader(train_set, \
                                                 batch_size=config.dataset.train.batch_size, \
                                                 shuffle=config.dataset.train.shuffle, \
                                                 num_workers=config.dataset.train.num_workers)

    val_set = MultiView_SynData(config.dataset.data_root, load_joints=config.model.backbone.num_joints, invalid_joints=(), \
                                bbox=config.dataset.bbox, image_shape=config.dataset.image_shape, \
                                test=True)
    val_loader = datasets_utils.syndata_loader(val_set, \
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
        criterion = HeatmapMSELoss(config.dataset.image_shape)

    # configure optimizer
    opt = torch.optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), lr=config.opt.lr)

    multiview_train(config, model, train_loader, val_loader, criterion, opt, config.opt.n_epochs, device, \
                    resume=args.resume, logdir=args.logdir)
