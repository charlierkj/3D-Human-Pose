import numpy as np
import torch
import torch.utils.data as td

def collate_fn(batch):
    # mainly used to convert numpy to tensor and concatenate,
    # with making sure the data type of tensors is float
    samples = list(d for d in batch if d is not None)
    if len(samples) == 0:
        print("current batch is empty")
        return None

    images_batch = torch.stack([sample['images'] for sample in samples], dim=0) # batch_size x num_views x 3 x h x w
    proj_mats_batch = torch.stack([torch.stack([torch.from_numpy(cam.get_P()) for cam in sample['cameras']], dim=0)\
                                   for sample in samples], dim=0) # batch_size x num_views x 3 x 4
    joints_3d_gt_batch = torch.stack([sample['joints_3d_gt'] for sample in samples], dim=0) # batch_size x num_joints x 3

    images_batch, proj_mats_batch, joints_3d_gt_batch \
                  = images_batch.type(torch.float32), proj_mats_batch.type(torch.float32), joints_3d_gt_batch.type(torch.float32)
    return images_batch, proj_mats_batch, joints_3d_gt_batch


def syndata_loader(dataset, batch_size=1, shuffle=False):
    dataloader = td.DataLoader(dataset,
                               batch_size=batch_size,
                               shuffle=shuffle,
                               collate_fn = collate_fn,
                               pin_memory=True)
    return dataloader
    
