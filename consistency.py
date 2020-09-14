import os, sys
import argparse
import numpy as np
import torch
import torchgeometry as tgm
import matplotlib.pyplot as plt

from utils import op, cfg
from models.triangulation import AlgebraicTriangulationNet
from datasets.human36m import Human36MMultiViewDataset
import datasets.utils as datasets_utils
import train


def consistency_ensemble(model, images_batch, proj_mats_batch, num_tfs=4):
    sf = 0.25 # scale factor
    rf = 30 # angle factor
    
    batch_size = images_batch.shape[0]
    num_views = images_batch.shape[1]
    image_shape = images_batch.shape[3:] # [h, w]

    for i in range(num_tfs):
        if i==0:
            # original images
            _, joints_2d_org, heatmaps_pred, _ = model(images_batch, proj_mats_batch)
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

            M = tgm.get_rotation_matrix2d(image_center, angle, scale) # transform
            M = M.to(images_expanded.device)
            images_warped = tgm.warp_affine(images_expanded, M, \
                                            dsize=(image_shape[0], image_shape[1]))
            images_warped = images_warped.view(batch_size, num_views, 3, *image_shape)

            # prediction of warped images
            _, _, heatmaps_warped, _ = model(images_warped, proj_mats_batch)

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
    
    return joints_2d_pred, scores_pred


def generate_pseudo_labels(config, model, h36m_loader, device, \
                           num_tfs=4, test=False):
    num_joints = config.model.backbone.num_joints

    # fill in parameters
    retval = {
        'dataset': config.dataset.type,
        'image_shape': config.dataset.image_shape,
        'num_joints': config.model.backbone.num_joints,
        'scale_bbox': config.dataset.train.scale_bbox
        }
    labels_dtype = np.dtype([
        ('data_idx', np.int32),
        ('view_idx', np.int8),
        ('joints_2d', np.float32, (num_joints, 2)),
        ('scores', np.float32, (num_joints, ))
        ])
    retval['labels'] = []

    print("Generating pseudo labels...")
    model.eval()
    with torch.no_grad():
        for iter_idx, (images_batch, proj_mats_batch, joints_3d_gt_batch, joints_3d_valid_batch, indexes) in enumerate(h36m_loader):
            # if iter_idx > 5:
            #     break

            if images_batch is None:
                continue
                    
            images_batch = images_batch.to(device)
            proj_mats_batch = proj_mats_batch.to(device)
            joints_3d_gt_batch = joints_3d_gt_batch.to(device)
            joints_3d_valid_batch = joints_3d_valid_batch.to(device)

            batch_size = images_batch.shape[0]
            num_views = images_batch.shape[1]
            assert batch_size == len(indexes)
            
            joints_2d_pred, scores_pred = consistency_ensemble(model, images_batch, proj_mats_batch, num_tfs=num_tfs)

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
        write_path = os.path.join(save_folder, "human36m_test.npy")
    else:
        write_path = os.path.join(save_folder, "human36m_train.npy")

    print("Saving pseudo labels to %s" % write_path)
    np.save(write_path, retval)
    print("Done.")



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
                    retain_every_n_frames_in_test=config.dataset.test.retain_every_n_frames_in_test,
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
                    retain_every_n_frames_in_test=config.dataset.test.retain_every_n_frames_in_test,
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
