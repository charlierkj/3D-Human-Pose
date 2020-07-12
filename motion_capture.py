import os
import argparse
import numpy as np
import torch

from mvn.utils import cfg
from mvn.models_temp.triangulation import AlgebraicTriangulationNet
from datasets.multiview_syndata import MultiView_SynData
import datasets.utils as datasets_utils

from test import multiview_test
from blender.run_blender import *


if __name__ == '__main__':

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/test_03_temp/multiview_data')
    parser.add_argument('--output_folder', type=str, default='results/test_03_temp')
    args = parser.parse_args()

    # prepare model and data
    device = torch.device(0)
    config = cfg.load_config('experiments/syn_data/multiview_data_2_alg.yaml')
    model = AlgebraicTriangulationNet(config, device=device).to(device)

    state_dict = torch.load(config.model.checkpoint)
    for key in list(state_dict.keys()):
        new_key = key.replace("module.", "")
        state_dict[new_key] = state_dict.pop(key)
    model.load_state_dict(state_dict, strict=True)

    print("Loading data ...")
    dataset = MultiView_SynData(args.data_path, invalid_joints=(9, 16), bbox=[80, 0, 560, 480], ori_form=1)
    dataloader = datasets_utils.syndata_loader(dataset, batch_size=4)

    # 3D human pose estimation
    print("Estimating human pose ...")
    output_folder = os.path.abspath(args.output_folder)
    multiview_test(model, dataloader, device, output_folder, make_vid=True)

    # convert to .bvh file
    print("Converting to .bvh file ...")
    npy_path = os.path.join(output_folder, 'preds', 'S0', 'anim_000', 'joints_3d.npy')
    bvh_file = os.path.join(output_folder, 'S0_anim_000.bvh')
    blender_convert(npy_path, bvh_file)

    # render
    print("Rendering in Blender ...")
    save_folder = os.path.join(output_folder, 'rendered')
    blender_render(bvh_file, save_folder)
