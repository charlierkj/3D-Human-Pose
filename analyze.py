import os, json
import numpy as np
import matplotlib.pyplot as plt

from test import evaluate_one_scene, evaluate_one_batch


def plot_error_per_subject(dataset_folder, result_folder, write_path):
    preds_folder = os.path.join(result_folder, 'preds')
    subj_names = os.listdir(preds_folder)
    errors = []
    for subj_name in subj_names:
        error_subj = 0
        subj_folder = os.path.join(preds_folder, subj_name)
        anim_names = os.listdir(subj_folder)
        num_anims = len(anim_names)
        for anim_name in anim_names:
            anim_folder = os.path.join(subj_folder, anim_name)
            joints_3d_pred_path = os.path.join(anim_folder, 'joints_3d.npy')
            scene_folder = os.path.join(dataset_folder, subj_name, anim_name)
            error_scene = evaluate_one_scene(joints_3d_pred_path, scene_folder)
            error_subj += error_scene
        error_subj /= num_anims
        errors.append(error_subj)
    # plot
    plt.bar(range(len(subj_names)), errors, tick_labels=subj_names)
    plt.savefig(write_path)


if __name__ == "__main__":

    dataset_folder = '../mocap_syndata/multiview_data'
    result_folder = 'results/mocap_syndata'

    plot_error_per_subject(dataset_folder, result_folder, write_path='figs/temp.png')
