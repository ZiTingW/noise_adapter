# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 14:38:33 2025

@author: wenzt
"""

import os
import argparse
import numpy as np


from dataset_model import DiverseSinglePhotonDataset
from extract_noise_embedding import create_tof_matrix_numpy, extract_mean_photon_sbr




parser = argparse.ArgumentParser(description="Single-Photon feature extraction (configurable).")

parser.add_argument("--root", type=str, default="C:\\document\\data\\spad_real_diverse\\test", help="path of dataset")
parser.add_argument("--dataset", type=str, default="DiverseSinglePhotonDataset",
                    choices=['DiverseSinglePhotonDataset'],
                    help="Dataset")
parser.add_argument("--indices_npy", type=str, default=None, help="subset index .npy path")
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--num_workers", type=int, default=2)

parser.add_argument("--save_dir", type=str, default='./feas/')
parser.add_argument("--save_prefix", type=str, default="spadrealdiverse_clipvitb32")

args = parser.parse_args()
os.makedirs(args.save_dir, exist_ok=True)


indices = None
if args.indices_npy is not None and os.path.isfile(args.indices_npy):
    indices = np.load(args.indices_npy)
    indices = [int(i) for i in indices.tolist()]

dataset = DiverseSinglePhotonDataset(root=args.root, indices=indices, transform=None, rawdata = True)


noise = []
labels = []
for isamp in range(len(dataset)):
    depth, spad_coords, spad_data, binDuration, label, fnames = dataset[isamp]
    
    tof = create_tof_matrix_numpy(spad_coords, spad_data)
    mean_photon_per_pixel, mean_sbr, msppp = extract_mean_photon_sbr(tof, depth[0,:,:].numpy(), binDuration, window = 10)
    noise += [ [mean_photon_per_pixel, mean_sbr, msppp] ]
    labels += [label.numpy()]
    

noise = np.array(noise)
labels = np.array(labels)

noise_path = os.path.join(args.save_dir, f"{args.save_prefix}_noise_win10.npy")
label_path = os.path.join(args.save_dir, f"{args.save_prefix}_labels_check.npy")

np.save(noise_path, noise)
np.save(label_path, labels)

print(f"Saved features to: {noise_path}")
print(f"Saved labels   to: {label_path}")

