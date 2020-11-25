import os, sys
import time
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

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def generate_features(config, model, dataloader, device, label_path, write_path):

    if os.path.exists(write_path):
        print("File %s already exists" % write_path)
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
    dataloader.shuffle=False

    # load pseudo labels
    pseudo_labels = np.load(label_path, allow_pickle=True).item()

    print("Generating feature vectors...")
    print("Estimated number of iterations is: %d" % round(dataloader.dataset.__len__() / dataloader.batch_size))
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for iter_idx, (images_batch, _, _, _, _, indexes) in enumerate(dataloader):
            if images_batch is None:
                continue
                    
            images_batch = images_batch.to(device)

            batch_size = images_batch.shape[0]
            num_views = images_batch.shape[1]
            image_shape = images_batch.shape[3:]
            assert batch_size == len(indexes)
            
            _, _, _, _, features = model(images_batch, None)
            feature_shape = features.shape[3:]
            ratio_h = feature_shape[0] / image_shape[0]
            ratio_w = feature_shape[1] / image_shape[1]
            joints_2d_batch, _ = consistency.get_pseudo_labels(pseudo_labels, indexes, num_views, 0)
            joints_2d_batch = joints_2d_batch.cpu().numpy()
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
                    feats_segment['feats'] = np.transpose(features[batch_idx, view_idx, :, feat_y, feat_x].cpu().numpy())

                    retval['features'].append(feats_segment)

    retval['features'] = np.concatenate(retval['features'])
    assert retval['features'].ndim == 1
    print("Total number of images in the dataset: ", len(retval['features']))

    # save pseudo labels
    save_folder = "feats"
    os.makedirs(save_folder, exist_ok=True)

    print("Saving features to %s" % write_path)
    np.save(write_path, retval)
    print("Done.")


def get_features(feats_npy, indexes, num_views, joints_2d_valid_batch):
    data_indexes = feats_npy['features']['data_idx']
    features_all = feats_npy['features']['feats']

    batch_size = len(indexes)
    num_joints = features_all.shape[1]
    feature_dims = features_all.shape[2]

    features_out = []
    for jnt_idx in range(num_joints):
        features_out.append(np.empty((0, feature_dims)))

    for i, data_idx in enumerate(indexes):
        features_seg = features_all[data_indexes==data_idx][0:num_views]
        joints_2d_valid = joints_2d_valid_batch[i, 0:num_views, :, :].squeeze(-1)
        for jnt_idx in range(num_joints):
            features_jnt = features_seg[:, jnt_idx, :]
            features_append = features_jnt[joints_2d_valid[:, jnt_idx]]
            features_out[jnt_idx] = np.vstack((features_out[jnt_idx], features_append))
    return features_out


def plot_tSNE(dataloader, device, label_path, feat_path):
    # re-configure data loader
    dataloader.shuffle=False

    # load pseudo labels and their features
    pseudo_labels = np.load(label_path, allow_pickle=True).item()
    feats_npy = np.load(feat_path, allow_pickle=True).item()
    num_joints = pseudo_labels['num_joints']

    # filtered features and detections in pseudo labels
    features_pl = []
    detections_pl = []
    for jnt_idx in range(num_joints):
        features_pl.append(np.empty((0, 256)))
        detections_pl.append(np.empty(0,).astype(np.bool))

    error_thresh = 20
    p = 0.2
    score_thresh = consistency.get_score_thresh(pseudo_labels, p, separate=True)
    for iter_idx, (images_batch, _, _, _, joints_2d_gt_batch, indexes) in enumerate(dataloader):
        if images_batch is None:
            continue

        joints_2d_pseudo, joints_2d_valid_batch = \
                consistency.get_pseudo_labels(pseudo_labels, indexes, images_batch.shape[1], score_thresh)

        features_batch = get_features(feats_npy, indexes, images_batch.shape[1], joints_2d_valid_batch)
        detections = (torch.norm(joints_2d_pseudo - joints_2d_gt_batch, dim=-1, keepdim=True) < error_thresh)

        for jnt_idx in range(num_joints):
            detections_jnt = detections[:, :, jnt_idx, :][joints_2d_valid_batch[:, :, jnt_idx, :]]
            features_pl[jnt_idx] = np.vstack((features_pl[jnt_idx], features_batch[jnt_idx]))
            assert len(features_batch[jnt_idx]) == len(detections_jnt)
            detections_pl[jnt_idx] = np.concatenate((detections_pl[jnt_idx], detections_jnt))
    
    # t-SNE visualization for each joint pseudo label
    for jnt_idx in range(num_joints):
        print("Joint %d:" % jnt_idx)
        time_start = time.time()
        tsne = TSNE(n_components=2)
        tsne_results = tsne.fit_transform(features_pl[jnt_idx])
        print("t-SNE done. Time elapsed: %f secs" % (time.time() - time_start))
        # plot
        pos = tsne_results[detections_pl[jnt_idx], :]
        neg = tsne_results[~detections_pl[jnt_idx], :]
        plt.title("t-SNE visualization")
        plt.scatter(x=pos[:, 0], y=pos[:, 1], alpha=0.3, label="inlier")
        plt.scatter(x=neg[:, 0], y=neg[:, 1], alpha=0.3, label="outlier")
        plt.legend()
        plt.savefig("figs/tsne_pseudo_labels_joint_%d.png" % jnt_idx)
        plt.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="experiments/human36m/train/human36m_alg_17jnts.yaml")
    parser.add_argument('--label_path', type=str, default="pseudo_labels/human36m_train_every_10_frames.npy")
    parser.add_argument('--write_path', type=str, default="feats/ft_human36m_train_every_10_frames.npy")
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

    # generate_features(config, model, dataloader, device, args.label_path, args.write_path)
    plot_tSNE(dataloader, device, args.label_path, args.write_path)

