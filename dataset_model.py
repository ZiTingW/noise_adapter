# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 13:38:54 2023

@author: wenzt
"""

import numpy as np
import os
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from PIL import Image
from torchvision.datasets.folder import ImageFolder, default_loader
import torch.nn.functional as F 
import torchvision
import torchvision.transforms as transforms

import glob
import random


class MLP(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim=[2048], out_dim=2048):
        super().__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim[0]),
            torch.nn.BatchNorm1d(hidden_dim[0]),
            torch.nn.ReLU(inplace=True)
        )
        
        if len(hidden_dim) == 1:
            self.layer2 = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim[0], out_dim)
                # torch.nn.BatchNorm1d(out_dim)
            )
            self.num_layers = 1
        else:
            self.layer2 = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim[0], hidden_dim[1]),
                torch.nn.BatchNorm1d(hidden_dim[1]),
                torch.nn.ReLU(inplace=True)
            )
            self.layer3 = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim[1], out_dim)
                # torch.nn.BatchNorm1d(out_dim)
            )
            self.num_layers = 2
            
        self.emb = None
        
    # def set_layers(self, num_layers):
    #     self.num_layers = num_layers

    def forward(self, x):
        
        x = self.layer1(x)
        if self.num_layers == 1:
            self.emb = x.clone()
            x = self.layer2(x)
        else:
            x = self.layer2(x)
            self.emb = x.clone()
            x = self.layer3(x)
        return x     

        

class DiverseSinglePhotonDataset(Dataset):


    def __init__(
        self,
        root: str,
        transform=None,
        indices=None,
        frame_filter=None,
        strict_keys: bool = True,
        rawdata: bool = False
    ):
        super().__init__()
        self.root = root
        self.transform = transform
        self.strict_keys = strict_keys
        self.rawdata = rawdata

        #
        if frame_filter is None:
            self.frame_filter = None
        elif isinstance(frame_filter, (list, tuple, set)):
            self.frame_filter = set(int(x) for x in frame_filter)
        else:
            self.frame_filter = {int(frame_filter)}

        # collect all data (fpath, label, frame_count)
        all_samples = []  # list of (fpath, label:int, frame:int)
        for class_name in sorted(os.listdir(root), key=lambda x: (len(x), x)):
            class_dir = os.path.join(root, class_name)
            if not os.path.isdir(class_dir):
                continue
            try:
                label = int(class_name)
            except ValueError:
                continue

            for sub_name in sorted(os.listdir(class_dir), key=lambda x: (len(x), x)):
                sub_dir = os.path.join(class_dir, sub_name)
                if not os.path.isdir(sub_dir):
                    continue
                try:
                    frame_count = int(sub_name)
                except ValueError:
                    continue

                if self.frame_filter is not None and frame_count not in self.frame_filter:
                    continue

                for fpath in sorted(glob.glob(os.path.join(sub_dir, "*.npz"))):
                    all_samples.append((fpath, label, frame_count))

        if not self.strict_keys:
            filtered = []
            for (fpath, label, frame_count) in all_samples:
                try:
                    with np.load(fpath) as data:
                        if ("depth" in data) or ("depth_ssp" in data):
                            filtered.append((fpath, label, frame_count))
                except Exception:
                    pass
            all_samples = filtered

        if indices is None:
            self.samples = all_samples
            self.indices = list(range(len(all_samples)))
        else:
            self.samples = [all_samples[i] for i in indices]
            self.indices = list(indices)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        fpath, label, frame_count = self.samples[index]

        with np.load(fpath) as data:
            # depth: 'depth' -> 'depth_ssp'
            if "depth" in data:
                depth = data["depth"].astype(np.float32)
            elif "depth_ssp" in data:
                depth = data["depth_ssp"].astype(np.float32)
            else:
                if self.strict_keys:
                    raise KeyError(f"Missing 'depth' (or 'depth_ssp') in {fpath}")
                
                raise KeyError(f"Missing 'depth' keys in {fpath} (strict_keys=False fallback)")

            _spad_coords = data["spad_coords"] if "spad_coords" in data else None
            _spad_data   = data["spad_data"]   if "spad_data"   in data else None
            _binDuration = data["binDuration"] if "binDuration" in data else None

        depth = torch.from_numpy(depth).unsqueeze(0)  # [1, H, W]
        if self.transform is not None:
            depth = self.transform(depth)

        filename_str = os.path.basename(fpath)
        label_tensor = torch.tensor(label, dtype=torch.long)
        index_tensor = torch.tensor(self.indices[index], dtype=torch.long)

        if self.rawdata:
            return depth, _spad_coords, _spad_data, _binDuration, label_tensor, filename_str
        else:
            return depth, label_tensor, index_tensor, filename_str

    def stats_by_frame(self):
        """返回每个帧数子文件夹包含的样本数统计 dict，如 {6: 1000, 10: 1000, ...}"""
        counts = {}
        for _, _, frame_count in self.samples:
            counts[frame_count] = counts.get(frame_count, 0) + 1
        return counts

