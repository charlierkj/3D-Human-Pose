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
import train


def get_score_thresh(pseudo_labels, percentage, separate=False):
    scores = pseudo_labels['labels']['scores']
    if separate: # separately get pseudo labels for each joint
        scores = np.sort(scores, axis=0)
        num_scores = len(scores)
        idx = int(num_scores * (1 - percentage))
        score_thresh = scores[idx, :].reshape(1, -1)
    else:
        scores = scores.flatten()
        scores = np.sort(scores)[::-1]
        num_scores = len(scores)
        idx = int(num_scores * percentage)
        score_thresh = scores[idx]
    # print(score_thresh)
    return score_thresh


def get_pseudo_labels(pseudo_labels, indexes, num_views, score_thresh):
    data_indexes = pseudo_labels['labels']['data_idx']
    joints_2d_all = pseudo_labels['labels']['joints_2d']
    scores_all = pseudo_labels['labels']['scores']

    batch_size = len(indexes)
    num_joints = joints_2d_all.shape[1]

    joints_2d_gt_batch = np.zeros(shape=(batch_size, num_views, num_joints, 2), dtype=np.float32)
    joints_2d_gt_valid_batch = np.zeros(shape=(batch_size, num_views, num_joints, 1), dtype=np.bool)

    for i, data_idx in enumerate(indexes):
        joints_2d_gt = joints_2d_all[data_indexes==data_idx][0:num_views]
        scores = scores_all[data_indexes==data_idx][0:num_views]
        joints_2d_gt_valid = np.expand_dims(scores >= score_thresh, axis=-1)
        joints_2d_gt_batch[i, 0:joints_2d_gt.shape[0], :, :] = joints_2d_gt
        joints_2d_gt_valid_batch[i, 0:joints_2d_gt_valid.shape[0], :, :] = joints_2d_gt_valid
    joints_2d_gt_batch = torch.from_numpy(joints_2d_gt_batch) # batch_size x num_views x num_joints x 2
    joints_2d_gt_valid_batch = torch.from_numpy(joints_2d_gt_valid_batch) # batch_size x num_views x num_joints x 1
    return joints_2d_gt_batch, joints_2d_gt_valid_batch


def consistency_ensemble(model, images_batch, proj_mats_batch, num_tfs=4):
    sf = 0.25 # scale factor
    rf = 30 # angle factor
    
    batch_size = images_batch.shape[0]
    num_views = images_batch.shape[1]
    image_shape = images_batch.shape[3:] # [h, w]

    for i in range(num_tfs):
        if i==0:
            # original images
            _, joints_2d_org, heatmaps_pred, _, _ = model(images_batch, proj_mats_batch)
            num_joints = heatmaps_pred.shape[2]
            heatmap_shape = heatmaps_pred.shape[3:] # [h, w]
            heatmaps_pred = heatmaps_pred.view(batch_size * num_views, num_joints, *heatmap_shape)
        else:
            # warping images
            images_expanded = images_batch.view(-1, 3, *image_shape)
            
            image_center = torch.ones(batch_size * num_views, 2) # rotation center
            image_center[..., 0] = image_shape[1] / 2 # x
            image_center[..., 1] = image_shape[0] / 2 # y
            scale = (1 + sf * torch.randn(batch_size * num_views)).clamp(1 - sf, 1 + sf) # scale
            angle = (rf * torch.randn(batch_size * num_views)).clamp(-2 * rf, 2 * rf) # rotation angle
            # scale = torch.ones(batch_size * num_views)
            # angle = [0, -60, -30, 30, 60][i] * torch.ones(batch_size * num_views)
            # scale = [1, 0.75, 0.88, 1.12, 1.25][i] * torch.ones(batch_size * num_views)
            # angle = torch.zeros(batch_size * num_views)

            M = tgm.get_rotation_matrix2d(image_center, angle, scale) # transform
            M = M.to(images_expanded.device)
            images_warped = tgm.warp_affine(images_expanded, M, \
                                            dsize=(image_shape[0], image_shape[1]))
            images_warped = images_warped.view(batch_size, num_views, 3, *image_shape)

            # prediction of warped images
            _, _, heatmaps_warped, _, _ = model(images_warped, proj_mats_batch)

            # warping back heatmaps
            heatmap_center = torch.ones(batch_size * num_views, 2)
            heatmap_center[..., 0] = heatmap_shape[1] / 2 # x
            heatmap_center[..., 1] = heatmap_shape[0] / 2 # y
            scale_back = (1 / scale)
            angle_back = (- angle)

            M_back = tgm.get_rotation_matrix2d(heatmap_center, angle_back, scale_back)
            M_back = M_back.to(heatmaps_warped.device)
            heatmaps_back = tgm.warp_affine(heatmaps_warped.view(batch_size * num_views, num_joints, *heatmap_shape), M_back, \
                                            dsize=(heatmap_shape[0], heatmap_shape[1]))
            heatmaps_back /= heatmaps_back.sum(dim=(2,3), keepdim=True)
            heatmaps_pred += heatmaps_back

    # caculate coordinates and scores
    heatmaps_pred /= num_tfs # average heatmaps
    # joints_2d_pred, heatmaps_pred = op.integrate_tensor_2d(heatmaps_pred, False)
    # scores_pred = heatmaps_pred[torch.arange(batch_size * num_views).repeat_interleave(num_joints), \
    #                             torch.arange(num_joints).repeat(batch_size * num_views), \
    #                             joints_2d_pred.view(-1, 2).type(torch.int64)[:, 1], \
    #                             joints_2d_pred.view(-1, 2).type(torch.int64)[:, 0]]
    # # reshape back
    # joints_2d_pred = joints_2d_pred.view(batch_size, num_views, num_joints, 2)
    # scores_pred = scores_pred.view(batch_size, num_views, num_joints)
    scores_pred, max_idx = torch.max(heatmaps_pred.view(batch_size, num_views, num_joints, -1), dim=-1)
    joints_2d_pred = torch.zeros_like(joints_2d_org)
    joints_2d_pred[:, :, :, 0] = max_idx % heatmap_shape[1] # x
    joints_2d_pred[:, :, :, 1] = max_idx // heatmap_shape[1] # y
    # upscale coordinates
    joints_2d_transformed = torch.zeros_like(joints_2d_pred)
    joints_2d_transformed[:, :, :, 0] = joints_2d_pred[:, :, :, 0] * (image_shape[1] / heatmap_shape[1])
    joints_2d_transformed[:, :, :, 1] = joints_2d_pred[:, :, :, 1] * (image_shape[0] / heatmap_shape[0])
    joints_2d_pred = joints_2d_transformed
    # convert back to numpy
    joints_2d_pred = joints_2d_pred.detach().cpu().numpy() # batch_size x num_views x num_joints x 2
    scores_pred = scores_pred.detach().cpu().numpy() # batch_size x num_views x num_joints

    """
    # debug
    for batch_idx in range(batch_size):
        for view_idx in range(num_views):
            for j in range(num_joints):
                plt.imshow(images_batch[batch_idx, view_idx, 0, :, :].detach().cpu().numpy())
                hm = heatmaps_pred.view(batch_size, num_views, num_joints, *heatmap_shape)[batch_idx, view_idx, j, :, :]
                hm_resized = cv2.resize(hm.detach().cpu().numpy(), (image_shape[1], image_shape[0]))
                plt.imshow(hm_resized, alpha=0.5)
                plt.scatter(joints_2d_pred[batch_idx, view_idx, j, 0], joints_2d_pred[batch_idx, view_idx, j, 1], \
                        s=2, color="red")
                plt.xlabel("score=%f" % scores_pred[batch_idx, view_idx, j])
                plt.savefig("hm/%d_%d_%d.png"%(batch_idx, view_idx, j))
                plt.close()
    """
    return joints_2d_pred, scores_pred


def generate_pseudo_labels(config, model, real_loader, device, \
                           num_tfs=4, test=False):

    if os.path.exists("pseudo_labels/%s_train.npy" % config.dataset.type):
        return

    num_joints = config.model.backbone.num_joints

    # fill in parameters
    retval = {
        'dataset': config.dataset.type,
        'split': "test" if test else "train",
        'image_shape': config.dataset.image_shape,
        'num_joints': config.model.backbone.num_joints,
        'scale_bbox': config.dataset.test.scale_bbox if test else config.dataset.train.scale_bbox,
        'retain_every_n_frames': config.dataset.test.retain_every_n_frames if test else config.dataset.train.retain_every_n_frames,
        'num_tfs': num_tfs
        }
    labels_dtype = np.dtype([
        ('data_idx', np.int32),
        ('view_idx', np.int8),
        ('joints_2d', np.float32, (num_joints, 2)),
        ('scores', np.float32, (num_joints, ))
        ])
    retval['labels'] = []

    # re-configure data loader
    real_loader.shuffle=False

    print("Generating pseudo labels...")
    print("Estimated number of iterations is: %d" % round(real_loader.dataset.__len__() / real_loader.batch_size))
    model.eval()
    with torch.no_grad():
        for iter_idx, (images_batch, _, _, _, _, indexes) in enumerate(real_loader):
            # if iter_idx > 1:
            #     break
            # print(iter_idx)
            if images_batch is None:
                continue
                    
            images_batch = images_batch.to(device)

            batch_size = images_batch.shape[0]
            num_views = images_batch.shape[1]
            assert batch_size == len(indexes)
            
            joints_2d_pred, scores_pred = consistency_ensemble(model, images_batch, None, num_tfs=num_tfs)

            # fill in pseudo labels
            for batch_idx, data_idx in enumerate(indexes):
                for view_idx in range(num_views):
                    labels_segment = np.empty(1, dtype=labels_dtype)
                    labels_segment['data_idx'] = data_idx
                    labels_segment['view_idx'] = view_idx
                    labels_segment['joints_2d'] = joints_2d_pred[batch_idx, view_idx, ...]
                    labels_segment['scores'] = scores_pred[batch_idx, view_idx, ...]

                    retval['labels'].append(labels_segment)

    retval['labels'] = np.concatenate(retval['labels'])
    assert retval['labels'].ndim == 1
    print("Total number of images in the dataset: ", len(retval['labels']))

    # save pseudo labels
    save_folder = "pseudo_labels"
    os.makedirs(save_folder, exist_ok=True)
    if test:
        write_path = os.path.join(save_folder, "%s_test.npy" % config.dataset.type)
    else:
        write_path = os.path.join(save_folder, "%s_train.npy" % config.dataset.type)

    print("Saving pseudo labels to %s" % write_path)
    np.save(write_path, retval)
    print("Done.")

    # configure back data loader
    real_loader.shuffle = config.dataset.train.shuffle



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="experiments/human36m/test/human36m_alg_17jnts.yaml")
    parser.add_argument('--num_tfs', type=int, default=4)
    parser.add_argument('--test', action='store_true')
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
    if args.test:
        print("Will generate pseudo labels for test set.")
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
    else:
        print("Will generate pseudo labels for train set.")
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

    generate_pseudo_labels(config, model, dataloader, device, num_tfs=args.num_tfs, test=args.test)

