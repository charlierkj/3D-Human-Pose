import os, sys
import argparse
import numpy as np
import torch
import torchgeometry as tgm
import cv2
import matplotlib.pyplot as plt

from utils import op, cfg
from models.triangulation import AlgebraicTriangulationNet
from datasets.human36m import Human36MMultiViewDataset
import datasets.utils as datasets_utils
import consistency
import train


def get_features(config, model, dataloader, device, label_path, write_path):

    if os.path.exists(output_path):
        print("File %s already exists" % output_path)
        return

    num_joints = config.model.backbone.num_joints

    # fill in parameters
    retval = {
        'dataset': config.dataset.type,
        'split': "train",
        'image_shape': config.dataset.image_shape,
        'num_joints': config.model.backbone.num_joints,
        'scale_bbox': config.dataset.train.scale_bbox,
        'retain_every_n_frames': config.dataset.train.retain_every_n_frames,
        'pseudo_label_path': label_path
        }
    feats_dtype = np.dtype([
        ('data_idx', np.int32),
        ('view_idx', np.int8),
        ('feats', np.float32, (num_joints, 256))
        ])
    retval['features'] = []

    # re-configure data loader
    data_loader.shuffle=False

    # load pseudo labels
    pseudo_labels = np.load(label_path, allow_pickle=True).item()

    print("Generating feature vectors...")
    print("Estimated number of iterations is: %d" % round(dataloader.dataset.__len__() / dataloader.batch_size))
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for iter_idx, (images_batch, _, _, _, _, indexes) in enumerate(realloader):
            if images_batch is None:
                continue
                    
            images_batch = images_batch.to(device)

            batch_size = images_batch.shape[0]
            num_views = images_batch.shape[1]
            image_shape = images_batch.shape[3:]
            print(image_shape)
            assert batch_size == len(indexes)
            
            _, _, _, _, features = model(images_batch, None)
            feature_shape = features.shape[3:]
            ratio_h = feature_shape[0] / image_shape[0]
            ratio_w = feature_shape[1] / image_shape[1]
            joints_2d_batch, _ = consistency.get_pseudo_labels(pseudo_labels, indexes, num_views, 0)
            print(joints_2d_batch.shape)
            feat_xs = (joints_2d_batch[:, :, :, 0] * ratio_w).astype(np.int32)
            feat_ys = (joints_2d_batch[:, :, :, 1] * ratio_h).astype(np.int32)

            # fill in pseudo labels
            for batch_idx, data_idx in enumerate(indexes):
                for view_idx in range(num_views):
                    feat_x = feat_xs[batch_idx, view_idx, :]
                    feat_y = feat_ys[batch_idx, view_idx, :]
                    feats_segment = np.empty(1, dtype=feats_dtype)
                    feats_segment['data_idx'] = data_idx
                    feats_segment['view_idx'] = view_idx
                    feats_segment['feats'] = features[batch_idx, view_idx, :, feat_y, feat_x].cpu().numpy()
                    print(feats_segment['feats'].shape)

                    retval['features'].append(feats_segment)

    retval['features'] = np.concatenate(retval['features'])
    assert retval['features'].ndim == 1
    print("Total number of images in the dataset: ", len(retval['features']))

    # save pseudo labels
    save_folder = "feats"
    os.makedirs(save_folder, exist_ok=True)

    print("Saving features to %s" % os.path.join(save_folder, write_path))
    np.save(os.path.join(save_folder, write_path), retval)
    print("Done.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="experiments/human36m/train/human36m_alg_17jnts.yaml")
    parser.add_argument('--label_path', type=str, default="pseudo_labels/human36m_train_every_10_frames.npy")
    parser.add_argument('--write_path', type=str, default="/ft_human36m_train_every_10_frames.npy")
    args = parser.parse_args()

    config = cfg.load_config(args.config)

    assert config.dataset.type == "human36m"

    device = torch.device(int(config.gpu_id))
    print(device)

    model = AlgebraicTriangulationNet(config, device=device)

    model = torch.nn.DataParallel(model, device_ids=[int(config.gpu_id)])

    if config.model.init_weights:
        print("Initializing model weights..")
        model = train.load_pretrained_model(model, config)

    # load data
    print("Loading data..")
    dataset = Human36MMultiViewDataset(
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
    dataloader = datasets_utils.human36m_loader(dataset, \
                                                batch_size=config.dataset.train.batch_size, \
                                                shuffle=config.dataset.train.shuffle, \
                                                num_workers=config.dataset.train.num_workers)

    get_features(config, model, dataloader, device, args.label_path, args.write_path)

