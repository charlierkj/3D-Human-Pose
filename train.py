import os
import numpy as np
import argparse
import torch
from torch import nn
from datetime import datetime
from PIL import Image

from tensorboardX import SummaryWriter

from utils import cfg
from models.triangulation import AlgebraicTriangulationNet
from models.loss import KeypointsMSELoss, KeypointsMSESmoothLoss
from datasets.multiview_syndata import MultiView_SynData
import datasets.utils as datasets_utils

import visualize


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


def multiview_train(model, dataloader, criterion, opt, epochs, device, \
        continue_train=False, exp_log="exp_27jnts@15.08.2020-05.15.16"):
    # configure dir and writer for saving weights and intermediate evaluation
    if not continue_train:
        experiment_name = "{}_{}jnts@{}".format("exp", "%d" % dataloader.dataset.num_jnts, datetime.now().strftime("%d.%m.%Y-%H.%M.%S"))
        start_epoch = 0
    else:
        experiment_name = exp_log
        start_epoch = int(sorted(os.listdir(os.path.join("./logs", experiment_name, "checkpoint")))[-1]) + 1
        
    experiment_dir = os.path.join("./logs", experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    checkpoint_dir = os.path.join(experiment_dir, "checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(experiment_dir, "tensorboard"))
    
    model.to(device)
    model.train()

    for e in range(start_epoch, epochs):
        total_loss = 0
        total_samples = 0
        for iter_idx, (images_batch, proj_mats_batch, joints_3d_gt_batch, joints_3d_valid_batch, info_batch) in enumerate(dataloader):
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
            joints_all_valid = torch.ones_like(joints_3d_valid_batch)

            # calculate loss
            loss = criterion(joints_3d_pred, joints_3d_gt_batch, joints_all_valid)
            print("epoch: %d, iter: %d, loss: %.3f" % (e, iter_idx, loss.item()))
            total_loss += batch_size * loss.item()
            total_samples += batch_size

            # optimize
            opt.zero_grad()
            loss.backward()
            opt.step()

        # save loss per epoch to tensorboard
        total_loss /= total_samples
        writer.add_scalar("training loss", total_loss, e)

        # evaluate
        if e % 1 == 0:
            with torch.no_grad():
                model.eval()
                total_error = 0
                for iter_idx, (images_batch, proj_mats_batch, joints_3d_gt_batch, joints_3d_valid_batch, info_batch) in enumerate(dataloader):
                    if images_batch is None:
                        continue
                    
                    images_batch = images_batch.to(device)
                    proj_mats_batch = proj_mats_batch.to(device)
                    joints_3d_gt_batch = joints_3d_gt_batch.to(device)
                    joints_3d_valid_batch = joints_3d_valid_batch.to(device)
                    batch_size = images_batch.shape[0]
                    joints_3d_pred, joints_2d_pred, heatmaps_pred, confidences_pred = model(images_batch, proj_mats_batch)
                    metric = KeypointsMSELoss()
                    error = metric(joints_3d_pred, joints_3d_gt_batch, joints_3d_valid_batch)
                    total_error += batch_size * error.item()

                total_error /= total_samples
                writer.add_scalar("training error", total_error, e)
                print('Epoch: %03d | Train Loss: %.3f | Train Error: %.2f' % (e, total_loss, total_error))

                # save weights
                checkpoint_dir_e = os.path.join(checkpoint_dir, "%04d" % e)
                os.makedirs(checkpoint_dir_e, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(checkpoint_dir_e, "weights.pth"))

    # save weights
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "weights.pth"))
    print("Training Done.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=2)
    parser.add_argument('--num_jnts', type=int, default=23)
    args = parser.parse_args()

    device = torch.device(args.gpu_id)
    print(device)
    
    config = cfg.load_config('experiments/syndata/train/syndata_alg_%djnts.yaml' % args.num_jnts)

    model = AlgebraicTriangulationNet(config, device=device).to(device)

    model = load_pretrained_model(model, config)

    print("Loading data..")
    data_path = '../mocap_syndata/multiview_data'
    dataset = MultiView_SynData(data_path, load_joints=args.num_jnts, invalid_joints=(9, 16), bbox=[80, 0, 560, 480])
    dataloader = datasets_utils.syndata_loader(dataset, batch_size=2, shuffle=True)

    # configure loss
    if config.opt.criterion == "MSESmooth":
        criterion = KeypointsMSESmoothLoss(config.opt.mse_smooth_threshold)
    else:
        criterion = KeypointsMSELoss()

    # configure optimizer
    opt = torch.optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), lr=0.0001)

    multiview_train(model, dataloader, criterion, opt, 3, device, continue_train=True)
