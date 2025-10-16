'''
Reuse Evaluate Pose Script to evaluate the poses from Excel.
Report the metrics for each Excel.
'''

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from options import MonodepthOptions
import copy
from evaluate_pose import evaluate, create_pose_model

def evaluate_with_model(opt, pose_model):
    """Evaluate pose with a pre-loaded model to avoid reloading"""
    # This is a simplified version that reuses the pre-loaded model
    # We'll call the original evaluate function but modify it to skip model loading
    
    # Create a copy of opt to avoid modifying the original
    import copy
    opt_copy = copy.deepcopy(opt)
    
    # Set a flag to indicate we have a pre-loaded model
    opt_copy.pre_loaded_model = pose_model
    
    # Call the original evaluate function
    return evaluate(opt_copy)


def compute_average_metrics(metrics_all_seqs):
    # compute the average metrics per track_length for all sequences
    # metrics in order of ate, re, rpe_trans, rpe_rot
    average_metrics = {}
    for seq_name, metrics_dict in metrics_all_seqs.items():
        for track_length, metrics in metrics_dict.items():
            # Initialize track_length if it doesn't exist
            if track_length not in average_metrics:
                average_metrics[track_length] = {"ate_mean": [], "re_mean": [], "rpe_trans_mean": [], "rpe_rot_mean": [],
                                                "ate_std": [], "re_std": [], "rpe_trans_std": [], "rpe_rot_std": []}
            
            average_metrics[track_length]["ate_mean"].append(np.mean(metrics[0]))
            average_metrics[track_length]["ate_std"].append(np.std(metrics[0]))
            average_metrics[track_length]["re_mean"].append(np.mean(metrics[1]))
            average_metrics[track_length]["re_std"].append(np.std(metrics[1]))
            average_metrics[track_length]["rpe_trans_mean"].append(np.mean(metrics[2]))
            average_metrics[track_length]["rpe_trans_std"].append(np.std(metrics[2]))
            average_metrics[track_length]["rpe_rot_mean"].append(np.mean(metrics[3]))
            average_metrics[track_length]["rpe_rot_std"].append(np.std(metrics[3]))


    for track_length, metrics in average_metrics.items():
        print("-"*20)
        print(f"track_length: {track_length}, sanity_total sequences for mean: {len(metrics['ate_mean'])}")
        # print('ate_mean: {}'.format(metrics["ate_mean"]))
        # print('ate_std: {}'.format(metrics["ate_std"]))
        # print('re_mean: {}'.format(metrics["re_mean"]))
        # print('re_std: {}'.format(metrics["re_std"]))
        # print('rpe_trans_mean: {}'.format(metrics["rpe_trans_mean"]))
        # print('rpe_trans_std: {}'.format(metrics["rpe_trans_std"]))
        # print('rpe_rot_mean: {}'.format(metrics["rpe_rot_mean"]))
        # print('rpe_rot_std: {}'.format(metrics["rpe_rot_std"]))
        # compute mean of mean/std over seqs

        float_digits = 4
        print('ate_mean_mean: {:.{}f}'.format(np.mean(metrics["ate_mean"]), float_digits))
        print('re_mean_mean: {:.{}f}'.format(np.mean(metrics["re_mean"]), float_digits))
        print('rpe_trans_mean_mean: {:.{}f}'.format(np.mean(metrics["rpe_trans_mean"]), float_digits))
        print('rpe_rot_mean_mean: {:.{}f}'.format(np.mean(metrics["rpe_rot_mean"]), float_digits))
        print('ate_std_mean: {:.{}f}'.format(np.mean(metrics["ate_std"]), float_digits))
        print('re_std_mean: {:.{}f}'.format(np.mean(metrics["re_std"]), float_digits))
        print('rpe_trans_std_mean: {:.{}f}'.format(np.mean(metrics["rpe_trans_std"]), float_digits))
        print('rpe_rot_std_mean: {:.{}f}'.format(np.mean(metrics["rpe_rot_std"]), float_digits))

    return average_metrics

    

def main(args):
    ''' 
    udpate save_poses_root based on the video 
    update eval_split_appendix based on the video, start_idx, end_idx_cutoff
    '''

    excel_root = args.excel_root
    excel_name = args.excel_name
    excel_path = os.path.join(excel_root, excel_name)
    assert os.path.exists(excel_path), f"Excel file not found at {excel_path}"

    # construct save_poses_root based on the excel_root and excel_name
    assert os.path.exists(os.path.join(args.excel_root, args.excel_name)), f"Excel file not found at {os.path.join(args.excel_root, args.excel_name)}"
    if args.save_poses_root is None:
        save_poses_root = os.path.join(os.path.dirname(__file__), "splits", args.dataset, excel_name.replace(".xlsx", ""))
    else:
        save_poses_root = os.path.join(args.save_poses_root, excel_name.replace(".xlsx", ""))
    os.makedirs(save_poses_root, exist_ok=True)
    print(f"save_poses_root: {save_poses_root}")

    # Load model once before processing all sequences
    print("Loading model once for all evaluations...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_root = args.load_weights_folder
    pose_model = create_pose_model(args, device, model_root)
    pose_model.cuda()
    pose_model.eval()
    print("Model loaded successfully!")

    df = pd.read_excel(excel_path)
    metrics_all_seqs = {}
    total_frames = 0
    total_sequences = 0
    for index, row in df.iterrows():
        # print(row["video"])
        # print(row["start_idx"])
        # print(row["end_idx_cutoff"])
        print("--------------------------------")
        # construct eval_split_appendix based on the video
        eval_split_appendix = ''.join([str(row["video"]), "_", str(row["start_idx"]), "_", str(row["end_idx_cutoff"])])

        new_args = copy.deepcopy(args)
        new_args.save_poses_root = save_poses_root
        new_args.eval_split_appendix = eval_split_appendix
        # evaluate the pose with pre-loaded model
        metrics_per_seq = evaluate_with_model(new_args, pose_model)
        assert eval_split_appendix not in metrics_all_seqs, f"eval_split_appendix {eval_split_appendix} already exists in metrics_all_seqs? The excel contain repeated sequences?"
        metrics_all_seqs[eval_split_appendix] = metrics_per_seq
        total_sequences += 1
        total_frames += row["end_idx_cutoff"] - row["start_idx"] + 1 # we keep the end frames
        # print('per seq frames: {}'.format(metrics_per_seq[track_length_to_count_frames]["ate"].shape[0]))
    
    # save the metrics_all_seqs to a npz file
    npz_path = os.path.join(save_poses_root, f"{excel_name.replace('.xlsx', '')}_metrics_all_seqs.npz")
    np.savez_compressed(npz_path, metrics_all_seqs=metrics_all_seqs)
    print(f"saved metrics_all_seqs to {npz_path}")

    # report the per track_length metrics
    print("="*60)
    print("Average Metrics per track_length  over {} sequences, total {} frames".format(len(metrics_all_seqs), total_frames))
    print("="*60)
    average_metrics = compute_average_metrics(metrics_all_seqs)
 
    return metrics_all_seqs
    
if __name__ == "__main__":

    options = MonodepthOptions()
    args = options.parse()


    # evaluate(options.parse())

    
    metrics_all_seqs = main(args)