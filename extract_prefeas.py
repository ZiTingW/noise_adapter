# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 14:38:33 2025

@author: wenzt
"""

import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T

from dataset_model import DiverseSinglePhotonDataset

class NormalizeByImageMax:
    def __call__(self, tensor):
        # tensor: [1, H, W]
        max_val = tensor.max()
        return tensor / max_val if max_val > 1e-6 else tensor

class AddGaussianNoise:
    def __init__(self, mean=0., std=1.):
        self.mean, self.std = mean, std
    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std + self.mean
    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


def load_backbone(args):

    device = args.device

    if args.model == "clip_vitb32":
        import clip
        model, preprocess = clip.load(
            "ViT-B/32",
            device=device,
            download_root=args.clip_download_root
        )

        model.eval()
        return model

    elif args.model in {"dinov2_vits14", "dinov2_vitb14"}:
        name = "dinov2_vits14" if args.model == "dinov2_vits14" else "dinov2_vitb14"
        model = torch.hub.load('facebookresearch/dinov2', name)  # type: ignore
        model = model.to(device).eval()

        return model

    elif args.model == "dinov3_local":

        assert os.path.isdir(args.dinov3_repo), "dinov3_repo 不存在"
        assert os.path.isfile(args.dinov3_weights), "dinov3_weights 不存在"
        model = torch.hub.load(args.dinov3_repo, 'dinov3_vits16', source='local', weights=args.dinov3_weights)  # type: ignore
        model = model.to(device).eval()

        return model

    else:
        raise ValueError(f"unkonwn model: {args.model}")


@torch.no_grad()
def extract_feats_once(dataloader, model, device, use_clip, repeat_gray_to_rgb):

    all_feats = []
    all_labels = []
    all_fnames = []

    for inputs, labels, sample_id, fnames in dataloader:
        print(len(all_feats))

        if inputs.ndim == 5:
            B, N, C, H, W = inputs.shape
            inputs = inputs.view(B * N, C, H, W)
            labels = labels.view(-1)

        inputs = inputs.to(device=device, dtype=torch.float32)

        if repeat_gray_to_rgb and inputs.shape[1] == 1:
            inputs = inputs.repeat(1, 3, 1, 1)

        if use_clip:
            # OpenAI CLIP：encode_image
            feats = model.encode_image(inputs)
        else:
            # DINO 
            feats = model(inputs)

        feats = feats.detach().cpu().numpy()
        all_feats.append(feats)
        all_labels.append(labels.detach().cpu().numpy())
        all_fnames.extend(list(fnames))

    all_feats = np.concatenate(all_feats, axis=0) if len(all_feats) else np.zeros((0, 1))
    all_labels = np.concatenate(all_labels, axis=0) if len(all_labels) else np.zeros((0,), dtype=np.int64)
    return all_feats, all_labels, np.array(all_fnames)

def main():
    parser = argparse.ArgumentParser(description="Single-Photon feature extraction (configurable).")

    parser.add_argument("--root", type=str, required=True, help="root of dataset")
    parser.add_argument("--dataset", type=str, default="DiverseSinglePhotonDataset",
                        choices=['DiverseSinglePhotonDataset'],
                        help=" Dataset name")
    parser.add_argument("--indices_npy", type=str, default=None, help="path of subset index")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=2)

    # preprocessing
    parser.add_argument("--resize", type=int, default=224)
    parser.add_argument("--hflip_p", type=float, default=0)
    parser.add_argument("--vflip_p", type=float, default=0)
    parser.add_argument("--use_noise", default=False, type=str)
    parser.add_argument("--noise_std", type=float, default=0.1)
    parser.add_argument("--normalize_by_max", default=True, type=str)
    parser.add_argument("--repeat_gray_to_rgb", default=True, type=str)

    # model/device
    parser.add_argument("--model", type=str, default="clip_vitb32",
                        choices=["clip_vitb32", "dinov2_vits14", "dinov2_vitb14", "dinov3_local"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--clip_download_root", type=str, default='./pretrain_encoder/',
                        help="root of local clip ckpt")
    parser.add_argument("--dinov3_repo", type=str, default='./dinov3_repo/', help="local DINOv3 repo ")
    parser.add_argument("--dinov3_weights", type=str, default='./pretrain_encoder/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth', help="local DINOv3 ckpt")

    #
    parser.add_argument("--augment_runs", type=int, default=1,
                        help="multi-augmentation for features")

    # save
    parser.add_argument("--save_dir", type=str, default='./pretrain_encoder/feas/')
    parser.add_argument("--save_prefix", type=str, default="spadrealdiverse_clipvitb32")

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # 
    transforms_list = [
        T.RandomHorizontalFlip(p=args.hflip_p),
        T.RandomVerticalFlip(p=args.vflip_p),
        T.Resize((args.resize, args.resize)),
    ]
    if args.use_noise:
        transforms_list.append(AddGaussianNoise(std=args.noise_std))
    if args.normalize_by_max:
        transforms_list.append(NormalizeByImageMax())
    transform = T.Compose(transforms_list)

    # dataset
    indices = None
    if args.indices_npy is not None and os.path.isfile(args.indices_npy):
        indices = np.load(args.indices_npy)
        indices = [int(i) for i in indices.tolist()]

    dataset = DiverseSinglePhotonDataset(root=args.root, indices=indices, transform=transform)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        # pin_memory=True,
        shuffle=False,
        drop_last=False
    )

    model = load_backbone(args)
    use_clip = ( "clip" in args.model)

    all_runs_feats = []
    all_runs_labels = None
    all_runs_fnames = None

    for run in range(args.augment_runs):
        feats, labels, fnames = extract_feats_once(
            dataloader, model, args.device, use_clip=use_clip, repeat_gray_to_rgb=args.repeat_gray_to_rgb
        )
        all_runs_feats.append(feats)
        if all_runs_labels is None:
            all_runs_labels = labels
            all_runs_fnames = fnames

        print(f"[Run {run+1}/{args.augment_runs}] feats shape: {feats.shape}, labels: {labels.shape}")

    if len(all_runs_feats) == 1:
        final_feats = all_runs_feats[0]
    else:
        final_feats = np.vstack(all_runs_feats)

    feat_path = os.path.join(args.save_dir, f"{args.save_prefix}_features.npy")
    label_path = os.path.join(args.save_dir, f"{args.save_prefix}_labels.npy")
    fname_path = os.path.join(args.save_dir, f"{args.save_prefix}_filenames.npy")

    np.save(feat_path, final_feats)
    np.save(label_path, all_runs_labels)
    np.save(fname_path, np.array(all_runs_fnames, dtype='U'))

    print(f"Saved features to: {feat_path}")
    print(f"Saved labels   to: {label_path}")
    print(f"Saved filenames to: {fname_path}")

if __name__ == "__main__":
    main()
