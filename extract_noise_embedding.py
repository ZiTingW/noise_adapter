# -*- coding: utf-8 -*-
"""
Created on Fri May  9 21:29:21 2025

@author: wenzt
"""

import numpy as np
import os
from scipy.sparse import csc_matrix


def restore_spd_matrix(path):

    dic_csc = np.load(path, allow_pickle=True)
    
    name = 'spad'
    depth = dic_csc['depth']
    data = dic_csc[f"{name}_data"]
    indices = dic_csc[f"{name}_coords"]
    shape = dic_csc[f"{name}_shape"]
    binDuration = dic_csc['binDuration']
    
    return depth, data, indices, shape, binDuration
    
def create_tof_matrix_numpy(spad_coords, spad_data):

    rows = spad_coords[0].astype(int)
    cols = spad_coords[1].astype(int)
    times = spad_coords[2]

    max_row, max_col = rows.max() + 1, cols.max() + 1

    tof_matrix = np.empty((max_row, max_col), dtype=object)

    for i in range(max_row):
        for j in range(max_col):
            tof_matrix[i, j] = []

    for idx in range(len(rows)):
        row, col, time = rows[idx], cols[idx], times[idx]
        photon_count = int(spad_data[idx])

        tof_matrix[row, col].extend([time] * photon_count)

    for i in range(max_row):
        for j in range(max_col):
            if tof_matrix[i, j]:  
                tof_matrix[i, j] = np.array(tof_matrix[i, j])
            else:
                tof_matrix[i, j] = np.array([])

    return tof_matrix

def extract_mean_photon_sbr(tof, depth_ssp, binDuration, window = 5):
    
    c = 3e8  # speed of light in m/s

    H, W = depth_ssp.shape
    total_photon = 0
    total_signal = 0
    total_background = 0
    valid_pixel_count = 0

    for i in range(H):
        for j in range(W):
            hist = tof[i, j]
            if len(hist) == 0:
                continue  # skip if no photon detected

            dist = depth_ssp[i, j]
            if dist <= 0:
                continue  # invalid depth

            binid = int(round((2 * dist) / (binDuration * c)))
            signal_bins = set(range(binid - window, binid + window + 1))

            signal_count = sum(1 for b in hist[:] if b in signal_bins)
            background_count = len(hist) - signal_count

            total_photon += len(hist)
            total_signal += signal_count
            total_background += background_count
            valid_pixel_count += 1

    mean_photon_per_pixel = total_photon / (H * W)
    mean_sbr = (total_signal / (total_background + 1e-6)) if total_background > 0 else 0.0

    return mean_photon_per_pixel, mean_sbr, total_signal/64/64

