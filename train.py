import os
import numpy as np
import torch
from torch import nn
from datetime import datetime
from PIL import Image

#from tensorboard import SummaryWriter

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


def multiview_train(model, dataloader, criterion, opt, epochs, device):
    # configure dir and writer for saving weights and intermediate evaluation
    experiment_name = "{}@{}".format("exp", datetime.now().strftime("%d.%m.%Y-%H.%M.%S"))
    experiment_dir = os.path.join("./logs", experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    checkpoint_dir = os.path.join(experiment_dir, "checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)
    #writer = SummaryWriter(os.path.join(experiment_dir, "tensorboard"))
    
    model.to(device)
    model.train()

    for e in range(epochs):
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

            # calculate loss
            loss = criterion(joints_3d_pred, joints_3d_gt_batch, joints_3d_valid_batch)
            print(loss.item())
            total_loss += batch_size * loss.item()
            total_samples += batch_size

            # optimize
            opt.zero_grad()
            loss.backward()
            opt.step()

        # save loss per epoch to tensorboard
        total_loss /= total_samples
        #writer.add_scalar("training loss", total_loss, e)

        # evaluate
        if e % 10 == 0:
            with torch.no_grad():
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
                    error = KeypointsMSELoss(joints_3d_pred, joints_3d_gt_batch, joints_3d_valid_batch)
                    total_error += batch_size * error.item()

                total_error /= total_samples
                #writer.add_scalar("training error", total_error, e)
                print('Epoch: %03d | Train Loss: %.3f | Train Accuracy: %.2f%%' % (e, total_loss, test_error))

    # save weights
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "weights.pth"))
    print("Training Done.")


if __name__ == "__main__":

    device = torch.device(0)
    
    config = cfg.load_config('experiments/syn_data/multiview_data_2_alg.yaml')

    model = AlgebraicTriangulationNet(config, device=device).to(device)

    model = load_pretrained_model(model, config)

    print("Loading data..")
    data_path = 'data/test_03/multiview_data'
    dataset = MultiView_SynData(data_path, load_joints=23, bbox=[80, 0, 560, 480])
    dataloader = datasets_utils.syndata_loader(dataset, batch_size=1)

    # configure loss
    if config.opt.criterion == "MSESmooth":
        criterion = KeypointsMSESmoothLoss(config.opt.mse_smooth_threshold)
    else:
        criterion = KeypointsMSELoss()

    # configure optimizer
    opt = torch.optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), lr=0.00001)

    multiview_train(model, dataloader, criterion, opt, 10, device)
