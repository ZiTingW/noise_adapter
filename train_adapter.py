# -*- coding: utf-8 -*-
"""
Created on Tue May  6 22:53:27 2025

@author: wenzt
"""

import os
import argparse
from importlib import reload 
import logging
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import cls_acc, search_hp

# -- Dataset splitting indices --
def split_indices(labels, train_ratio, val_ratio, seed):
    np.random.seed(seed)
    classes = np.unique(labels)
    train_idx, val_idx, test_idx = [], [], []
    for c in classes:
        idx = np.where(labels == c)[0]
        np.random.shuffle(idx)
        n_train = int(len(idx) * train_ratio)
        n_val = int(len(idx) * val_ratio)
        train_idx += idx[:n_train].tolist()
        val_idx += idx[n_train:n_train + n_val].tolist()
        test_idx += idx[n_train + n_val:].tolist()
    return np.array(train_idx), np.array(val_idx), np.array(test_idx)

# -- Sample N-shot support set --
def sample_support(labels, shots, seed):
    np.random.seed(seed)
    support_idx = []
    classes = np.unique(labels)
    for c in classes:
        idx = np.where(labels == c)[0]
        np.random.shuffle(idx)
        support_idx += idx[:shots].tolist()
    return np.array(support_idx)


# noise adapter
class NoiseMLPAdapter(nn.Module):
    def __init__(self, feat_dim, noise_dim, hidden_dim=256, num_classes= 11):
        super().__init__()
        self.noise_mlp = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.ReLU(),
            # nn.LeakyReLU(),
            nn.Linear(hidden_dim, feat_dim),
            nn.Sigmoid()  # gating factor
            # nn.Tanh()
            # nn.LeakyReLU()
        )
        # self.classifier = nn.Linear(feat_dim, num_classes)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, feat, noise):
        gate = self.noise_mlp(noise)
        modulated_feat = feat * gate + feat
        return self.classifier(modulated_feat)
    

def run_noise_adapter_train(support_feats, support_noise, support_labels,
                            val_feats, val_noise, val_labels,
                            test_feats, test_noise, test_labels,
                            adapter_module, epochs=10, batch_size = 32, lr=1e-3, save_dir='./results'):

    logging.info("\n-------- Training noise-aware adapter on the support set. --------")
    device = support_feats.device
    adapter_module = adapter_module.to(device)

    # optimizer and scheduler
    train_loader = DataLoader(
        TensorDataset(torch.cat([support_feats, support_noise], dim=1), support_labels),
        batch_size=batch_size, shuffle=True
    )
    
    
    optimizer = torch.optim.Adam(adapter_module.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader))

    best_val_acc = 0.0
    for epoch in range(epochs):
        adapter_module.train()
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            x_feats = x[:,:-2]
            x_noise = x[:,-2:]
            logits = adapter_module(x_feats, x_noise)
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
    
        adapter_module.eval()
        with torch.no_grad():
            val_logits = adapter_module(val_feats, val_noise)
            val_pred = val_logits.argmax(dim=-1)
            val_acc = (val_pred == val_labels).float().mean().item() * 100
    
        logging.info(f"Epoch {epoch+1}: Val Acc = {val_acc:.2f}%")
    
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'adapter': adapter_module.state_dict(),
                'val_acc': val_acc
            }, os.path.join(save_dir, 'best_adapter.pth'))

    logging.info("\n-------- Evaluating on the test set. --------")
    checkpoint = torch.load(os.path.join(save_dir, 'best_adapter.pth'))
    adapter_module.load_state_dict(checkpoint['adapter'])
    adapter_module.eval()
    with torch.no_grad():
        test_logits = adapter_module(test_feats, test_noise)
        test_pred = test_logits.argmax(dim=-1)
        test_acc = (test_pred == test_labels).float().mean().item() * 100

    logging.info(f"**** Noise-Aware Adapter Test Accuracy: {test_acc:.2f}% ****")
    np.save(os.path.join(save_dir,'acc.npy'), np.array([test_acc]))
    np.save(os.path.join(save_dir, 'test_pred.npy'), test_pred.cpu().numpy() )

        
def simulate_realistic_augmented_feat(h, g, g2, alpha=0.1, gamma=1):#0.7
    
    with torch.no_grad():
        weight = ((1 - g/1).clamp(min=0.0)) ** gamma
        noise_std = alpha * weight
    
        noise = torch.randn_like(h) * noise_std
        h_noisy = h + noise
        h_aug = (1 + g2) * h_noisy
    return h_aug
    
        
def run_noise_adapter_feasaug_trainv2(support_feats, support_labels,
                            val_feats, val_labels,
                            test_feats, test_labels, sup_gate, sup_feats0, best_val_acc_before, best_test_acc_before,
                            classifier, epochs=10, batch_size = 32, lr=1e-3, save_dir='./results'):
    
    idx_gate_sample = [i for i in range(sup_gate.size(0))]
    
    logging.info("\n-------- Training noise-aware adapter on the support set. --------")
    device = support_feats.device
    classifier = classifier.to(device)

    # optimizer and scheduler
    train_loader = DataLoader(
        TensorDataset(torch.cat([sup_feats0, support_feats], dim=1), support_labels),
        batch_size=batch_size, shuffle=True
    )
    
    
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader))
    
    totloss = []  
    best_val_acc = 0.0
    
    gmean = sup_gate.mean(dim=0)
    for epoch in range(epochs):
        classifier.train()
        
        for x_all, y in train_loader:
            x_all, y = x_all.to(device), y.to(device)
            x = x_all[:,512:]
            x0 = x_all[:,:512]
            
            if len(totloss) > 10:# (epochs/2):#epoch > 30:
                                
                gate_idx = np.random.choice(idx_gate_sample, size=x.size(0)*2, replace=True)
                noisy_x = simulate_realistic_augmented_feat(x, gmean, sup_gate[gate_idx[x.size(0):],:], alpha=0.025)#0.025
                
                combined_x = torch.cat([x, noisy_x], dim=0)
                combined_y = torch.cat([y, y], dim=0)  # 使用相同的标签
                
                logits = classifier(combined_x)
                loss = F.cross_entropy(logits, combined_y)
            else:
                logits = classifier(x)
                loss = F.cross_entropy(logits, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            totloss.append(loss.item())
            
            
        classifier.eval()
        with torch.no_grad():
            val_logits = classifier(val_feats)
            val_pred = val_logits.argmax(dim=-1)
            val_acc = (val_pred == val_labels).float().mean().item() * 100

        logging.info(f"Epoch {epoch+1}: Val Acc = {val_acc:.2f}%")
    
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'classifier': classifier.state_dict(),
                'val_acc': val_acc
            }, os.path.join(save_dir, 'best_cls.pth'))


    logging.info("\n-------- Evaluating on the test set. --------")
    checkpoint = torch.load(os.path.join(save_dir, 'best_cls.pth'))
    classifier.load_state_dict(checkpoint['classifier'])
    classifier.eval()
    
    with torch.no_grad():
        test_logits = classifier(test_feats)
        test_pred = test_logits.argmax(dim=-1)
        test_acc = (test_pred == test_labels).float().mean().item() * 100

    logging.info(f"**** Noise-Aware Adapter Test Accuracy: {test_acc:.2f}% ****")
    
    if best_val_acc >= best_val_acc_before:#True:#
        np.save(os.path.join(save_dir,'acc.npy'), np.array([test_acc]))
        np.save(os.path.join(save_dir, 'test_pred.npy'), test_pred.cpu().numpy() )

# --linear/MLP probing
def train_cls(support_feats, support_labels,
                            val_feats, val_labels,
                            test_feats, test_labels,
                            model, epochs=10, batch_size = 32, lr=1e-3, save_dir='./results'):
    
    device = support_feats.device
    model = model.to(device)
    
    train_loader = DataLoader(
        TensorDataset(support_feats, support_labels),
        batch_size=batch_size, shuffle=True
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader)) 

    best_val_acc = 0.0
    for epoch in range(epochs):
        model.train()
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = F.cross_entropy(logits, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(val_feats)
            val_pred = val_logits.argmax(dim=-1)
            val_acc = (val_pred == val_labels).float().mean().item() * 100

        logging.info(f"Epoch {epoch+1}: Val Acc = {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'adapter': model.state_dict(),
                'val_acc': val_acc
            }, os.path.join(save_dir, 'best_adapter.pth'))

    logging.info("\n-------- Evaluating on the test set. --------")
    checkpoint = torch.load(os.path.join(save_dir, 'best_adapter.pth'))
    model.load_state_dict(checkpoint['adapter'])
    model.eval()
    with torch.no_grad():
        test_logits = model(test_feats)
        test_pred = test_logits.argmax(dim=-1)
        test_acc = (test_pred == test_labels).float().mean().item() * 100

    logging.info(f"**** Linear/MLP probing Test Accuracy: {test_acc:.2f}% ****")
    np.save(os.path.join(save_dir,'acc.npy'), np.array([test_acc]))
    np.save(os.path.join(save_dir, 'test_pred.npy'), test_pred.cpu().numpy() )
    
    return model

# -- CLIP-Adapter --
class CLIPAdapter(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        adapter_hidden_dim: int = 256,   
        alpha: float = 0.5,            
        beta: float = 0.5,        
        clip_scale: int = 100,
        learnable_alpha_beta: bool = False
    ):
        super().__init__()
        self.feat_dim = feat_dim
        
        self.visual_adapter = nn.Sequential(
            nn.Linear(feat_dim, adapter_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(adapter_hidden_dim, feat_dim),
            nn.ReLU(inplace=True)
        )
        
        self.text_adapter = nn.Sequential(
            nn.Linear(feat_dim, adapter_hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(adapter_hidden_dim, feat_dim, bias=False),
            nn.ReLU(inplace=True)
        )
        
        self.clip_scale = clip_scale
        if learnable_alpha_beta:
            self.alpha = nn.Parameter(torch.tensor(alpha))
            self.beta = nn.Parameter(torch.tensor(beta))
        else:
            self.alpha = alpha
            self.beta = beta

    def forward(self, image_features, text_features):

        adapted_vis = self.visual_adapter(image_features)
        adapted_txt = self.text_adapter(text_features)
        
        image_features = self.alpha * adapted_vis + (1 - self.alpha) * image_features
        text_features = self.beta * adapted_txt + (1 - self.beta) * text_features
        
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        logits = self.clip_scale * (image_features @ text_features.t())
        
        return logits

def search_hp_clip(adapter, val_feats, val_labels, text_features,
              clip_scale, alpha_max, beta_max, alpha_steps, beta_steps, device):

    adapter.eval()
    best_acc = 0.0
    best_alpha, best_beta = 0.5, 0.5
    
    alpha_list = torch.linspace(0, alpha_max, steps=alpha_steps)
    beta_list = torch.linspace(0, beta_max, steps=beta_steps)
    
    val_feats = val_feats.to(device)
    val_labels = val_labels.to(device)
    
    with torch.no_grad():
 
        adapted_txt = adapter.text_adapter(text_features)
        adapted_vis = adapter.visual_adapter(val_feats)
        
        for alpha in alpha_list:
 
            mixed_vis = alpha * adapted_vis + (1 - alpha) * val_feats
            mixed_vis = F.normalize(mixed_vis, dim=-1)
            
            for beta in beta_list:
 
                mixed_txt = beta * adapted_txt + (1 - beta) * text_features
                mixed_txt = F.normalize(mixed_txt, dim=-1)
                
                logits = clip_scale * mixed_vis @ mixed_txt.t()
                acc = (logits.argmax(1) == val_labels).float().mean().item()
                
                if acc > best_acc:
                    best_acc = acc
                    best_alpha = alpha.item()
                    best_beta = beta.item()
    
    return best_alpha, best_beta

def run_clip_adapter(support_feats, support_labels, val_feats, val_labels,
                    test_feats, test_labels, text_features, adapter,
                    logit_scale, alpha_max, beta_max, alpha_steps, beta_steps,
                    lr, epochs, batch_size, save_dir, device, allfeas = None):


    train_loader = DataLoader(
        TensorDataset(support_feats, support_labels),
        batch_size=batch_size, shuffle=True
    )
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader))

    best_val, best_ep = 0.0, 0

    for e in range(1, epochs+1):
        adapter.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            logits = adapter(x, text_features)
            loss = F.cross_entropy(logits, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        
        adapter.eval()
        with torch.no_grad():

            logits_val = adapter(val_feats, text_features)
            acc_val = (logits_val.argmax(1) == val_labels.to(device)).float().mean().item()
        
        tloss = loss.item()
        logging.info(f"CLIP-Adapter Ep{e}/{epochs} Val Acc:{acc_val:.4f} Loss:{tloss:.4f}")
        
        if acc_val > best_val:
            best_val, best_ep = acc_val, e
            torch.save(adapter.state_dict(), os.path.join(save_dir, 'best_adapter.pth'))
    
    adapter.load_state_dict(torch.load(os.path.join(save_dir, 'best_adapter.pth')))
    
    best_alpha, best_beta = search_hp_clip(
        adapter, val_feats, val_labels, text_features,
        logit_scale, alpha_max, beta_max, alpha_steps, beta_steps, device
    )
    logging.info(f"CLIP-Adapter best alpha={best_alpha:.3f}, beta={best_beta:.3f}")
    
    with torch.no_grad():
        adapter.alpha = best_alpha
        adapter.beta = best_beta
        
        logits_test = adapter(test_feats, text_features)
        acc_test = (logits_test.argmax(1) == test_labels.to(device)).float().mean().item()
    
    logging.info(f"CLIP-Adapter Test Acc:{acc_test:.4f}")
    
    np.savez(os.path.join(save_dir, 'clip_adapter_hp.npz'), alpha=best_alpha, beta=best_beta)
    np.save(os.path.join(save_dir, 'acc.npy'), np.array([acc_test]))
    np.save(os.path.join(save_dir, 'test_pred.npy'), logits_test.argmax(1).cpu().numpy() )
    
    return acc_test


# Meta-Adapter
class MetaAdapter(nn.Module):
    def __init__(self, dim=1024, num_heads=1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.alpha_proj = nn.Linear(dim, 1, bias=True)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.alpha_proj.weight)
        nn.init.constant_(self.alpha_proj.bias, 1)

    def forward(self, query, key, value):
        B, K, C = key.shape
        res = query

        query = query.reshape(B, 1, C)
        key = torch.cat([query, key], dim=1)
        value = torch.cat([query, value], dim=1)
        query = self.q_proj(query).reshape(B, self.num_heads, C)
        key = self.k_proj(key)

        query = query.reshape(B, self.num_heads, 1, -1).permute(0, 2, 1, 3)
        key = key.reshape(B, K + 1, 1, -1).permute(0, 2, 1, 3)
        value = value.reshape(B, K + 1, 1, -1).permute(0, 2, 1, 3)

        attn_weight = (query @ key.transpose(-1, -2) / torch.sqrt(torch.tensor(self.dim, dtype=torch.float))).softmax(-1)
        attn = attn_weight @ value

        alpha = torch.nn.functional.sigmoid(self.alpha_proj(res).reshape(B, -1, 1, 1))
        attn = (alpha * attn).squeeze()

        attn = res + attn
        attn = F.normalize(attn, p=2, dim=-1)
        return attn


def run_meta_adapter(support_feats, support_labels, test_features, test_labels, clip_weights, adapter,
                     lr, epochs, batch_size, save_dir):
    # Zero-shot CLIP
    clip_logits = 100. * test_features @ clip_weights
    acc = cls_acc(clip_logits, test_labels)
    logging.info("**** Zero-shot CLIP's test accuracy on novel classes: {:.2f}. ****".format(acc))

    
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr, eps=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs * len(DataLoader(TensorDataset(support_feats.T, support_labels), batch_size=batch_size)))
    loader = DataLoader(TensorDataset(support_feats.T, support_labels), batch_size=batch_size, shuffle=True)

    best_acc, best_epoch = 0.0, 0

    query = clip_weights.T
    key = support_feats.T.reshape(query.shape[0], -1, query.shape[1])

    for train_idx in range(1, epochs+1):
        # Train
        adapter.train()
        correct_samples, all_samples = 0, 0
        loss_list = []
        logging.info('Train Epoch: {:} / {:}'.format(train_idx, epochs))

        for i, (images, target) in enumerate(loader):
            image_features, target = images.cuda(), target.cuda()
            weights = adapter(query, key, key)
            tip_logits = 100. * image_features @ weights.T

            loss = F.cross_entropy(tip_logits, target)

            acc = cls_acc(tip_logits, target)
            correct_samples += acc / 100 * len(tip_logits)
            all_samples += len(tip_logits)
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()


        current_lr = scheduler.get_last_lr()[0]
        logging.info('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples,
                                                                       correct_samples, all_samples,
                                                                       sum(loss_list) / len(loss_list)))

        # Eval
        adapter.eval()

        query_test = clip_weights.T
        key_test = support_feats.T.reshape(query_test.shape[0], -1, query_test.shape[1])
        weights = adapter(query_test, key_test, key_test)
        tip_logits = 100. * test_features @ weights.T
        acc = cls_acc(tip_logits, test_labels)

        if acc > best_acc:
            best_acc = acc
            torch.save(adapter.state_dict(), save_dir + "/best_meta_" + str( len(support_feats)  / (int(test_labels.max()) + 1) ) + "shots.pth")
            torch.save(key, save_dir + "/keys" + str( len(support_feats)  / (int(test_labels.max()) + 1) ) + "shots.pt")

    logging.info("**** Meta-Adapter's best accuracy: {:.2f}. ****".format(best_acc))
    np.save(os.path.join(save_dir,'acc.npy'), np.array([best_acc]))
    np.save(os.path.join(save_dir, 'test_pred.npy'), tip_logits.argmax(1).cpu().numpy() )


# -- Tip-Adapter --
def run_tip_adapter(cache_k, support_labels, val_feats, val_labels,
                    test_feats, test_labels, clip_weights,
                    init_beta, init_alpha, beta_steps, alpha_steps, save_dir):
    logging.info("\n-------- Searching hyperparameters on the val set. --------")
    # Zero-shot CLIP
    clip_logits = 100. * val_feats[:,:len(clip_weights)] @ clip_weights
    acc0 = cls_acc(clip_logits, val_labels)
    logging.info(f"\n**** Zero-shot CLIP's val accuracy: {acc0:.2f}. ****\n")

    # Tip-Adapter warm-up
    beta, alpha = init_beta, init_alpha
    cache_v = F.one_hot(support_labels, num_classes=int(support_labels.max())+1).float().to(cache_k.device)
    affinity = val_feats @ cache_k.T
    cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_v
    
    tip_logits = clip_logits + cache_logits * alpha
    acc1 = cls_acc(tip_logits, val_labels)
    logging.info(f"**** Tip-Adapter's val accuracy: {acc1:.2f}. ****\n")

    # Search Hyperparameters
    best_beta, best_alpha = search_hp(
        {'search_hp':True,'search_scale':[init_beta,init_alpha],'search_step':[beta_steps,alpha_steps],'clip_scale':100.0},
        cache_k, cache_v, val_feats, val_labels, clip_weights
    )

    logging.info("\n-------- Evaluating on the test set. --------")
    # Zero-shot CLIP
    clip_logits = 100. * test_feats[:,:len(clip_weights)] @ clip_weights
    acc2 = cls_acc(clip_logits, test_labels)
    logging.info(f"\n**** Zero-shot CLIP's test accuracy: {acc2:.2f}. ****\n")

    # Tip-Adapter
    affinity = test_feats @ cache_k.T
    cache_logits = ((-1) * (best_beta - best_beta * affinity)).exp() @ cache_v
    tip_logits = clip_logits + cache_logits * best_alpha
    acc3 = cls_acc(tip_logits, test_labels)
    logging.info(f"**** Tip-Adapter's test accuracy: {acc3:.2f}. ****\n")
    np.savez(os.path.join(save_dir,'tip_hp.npz'), beta=best_beta, alpha=best_alpha)
    np.save(os.path.join(save_dir, 'acc.npy'), np.array([acc3]))
    np.save(os.path.join(save_dir, 'test_pred.npy'), tip_logits.argmax(1).cpu().numpy() )

# -- Tip-Adapter-F fine-tune + search/eval --
def run_tip_adapter_f(cache_k, support_labels, val_feats, val_labels,
                      test_feats, test_labels, clip_weights,
                      clip_scale, beta_max, alpha_max, beta_steps, alpha_steps,
                      lr, epochs, batch_size, save_dir, device):
    
    cache_v = F.one_hot(support_labels, num_classes=int(support_labels.max())+1).float().to(device)
    num_k, dim = cache_k.shape
    adapter = nn.Linear(dim, num_k, bias=False).to(device)
    adapter.weight.data = cache_k.clone()
    # adapter.weight = nn.Parameter(cache_k)
    
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs * len(DataLoader(TensorDataset(cache_k, support_labels), batch_size=batch_size)))
    loader = DataLoader(TensorDataset(cache_k, support_labels), batch_size=batch_size, shuffle=True)
    best_val, best_ep = 0.0, 0
    for e in range(1, epochs+1):
        adapter.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            aff = adapter(x)
            c_log = ((-beta_max + beta_max * aff).exp() @ cache_v)
            logits = clip_scale * (x[:,:len(clip_weights)] @ clip_weights) + alpha_max * c_log
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        adapter.eval()
        with torch.no_grad():
            aff_v = adapter(val_feats)
            c_v = ((-beta_max + beta_max * aff_v).exp() @ cache_v)
            log_v = clip_scale * (val_feats[:,:len(clip_weights)] @ clip_weights) + alpha_max * c_v
            acc_v = (log_v.argmax(1) == val_labels).float().mean().item()
        logging.info(f"Tip-Adapter-F Ep{e}/{epochs} Val Acc:{acc_v:.4f}")
        if acc_v > best_val:
            best_val, best_ep = acc_v, e
            torch.save(adapter.weight, os.path.join(save_dir, f'bestF_ep{e}.pth'))
    adapter.weight.data = torch.load(os.path.join(save_dir, f'bestF_ep{best_ep}.pth'))
    best_b, best_a = search_hp(
        {'search_hp':True,'search_scale':[beta_max,alpha_max],'search_step':[beta_steps,alpha_steps],'clip_scale':clip_scale},
        cache_k, cache_v, val_feats, val_labels, clip_weights, adapter
    )
    logging.info(f"Tip-Adapter-F best beta={best_b:.3f}, alpha={best_a:.3f}")
    with torch.no_grad():
        aff_t = adapter(test_feats)
        c_t = ((-best_b + best_b * aff_t).exp() @ cache_v)
        pred = clip_scale * (test_feats[:,:len(clip_weights)] @ clip_weights) + best_a * c_t
        acc_t = (pred.argmax(1) == test_labels).float().mean().item()
    logging.info(f"Tip-Adapter-F Test Acc:{acc_t:.4f}")
    np.savez(os.path.join(save_dir,'tipf_hp.npz'), beta=best_b, alpha=best_a)
    np.save(os.path.join(save_dir, 'acc.npy'), np.array([acc_t]))
    np.save(os.path.join(save_dir, 'test_pred.npy'), pred.argmax(1).cpu().numpy() )


def main(args):
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    # torch.backends.cudnn.deterministic = True
    
    exp = f"{args.method}_alpha0020_shots{args.shots}_seed{args.seed}"
    save_dir = os.path.join(args.output_dir, exp)
    os.makedirs(save_dir, exist_ok=True)
    reload(logging)
    
    feats = np.load(args.image_feat_path)
    
    if args.use_noise:
        noise = np.load(args.noise_path)
        feats = np.hstack((feats, noise))
    
    labels = np.load(args.label_path)
    tr, vl, te = split_indices(labels, args.train_ratio, args.val_ratio, args.seed)
    np.save(os.path.join(save_dir, 'tr_idx.npy'), tr)
    np.save(os.path.join(save_dir, 'vl_idx.npy'), vl)
    np.save(os.path.join(save_dir, 'te_idx.npy'), te)
    
    all_feats = torch.from_numpy(feats).float().to(device)
    all_labels = torch.from_numpy(labels).long().to(device)
    all_feats /= all_feats.norm(dim=-1, keepdim=True)
    
    sup_idx = sample_support(all_labels[tr].cpu().numpy(), args.shots, args.seed)
    np.save(os.path.join(save_dir, 'sup_idx.npy'), tr[sup_idx])
    
    sup_feats = all_feats[tr][sup_idx]
    sup_labels = all_labels[tr][sup_idx]
    val_feats, val_labels = all_feats[vl], all_labels[vl]
    test_feats, test_labels = all_feats[te], all_labels[te]
    clip_w = torch.from_numpy(np.load(args.text_feat_path)).float().to(device)
    clip_w /= clip_w.norm(dim=0, keepdim=True)
    
    if args.method == 'noise_adapter_feasaug':
        noise = np.load(args.noise_path)
        noise = torch.from_numpy(noise)
        mean = noise.mean(dim=0, keepdim=True)
        std = noise.std(dim=0, keepdim=True) + 1e-6
        noise = (noise - mean) / std
        support_noise, val_noise, test_noise = noise[tr][sup_idx].float().to(device), noise[vl].float().to(device), noise[te].float().to(device)
    
    
    if args.method == 'clip_adapter':
        adapter = CLIPAdapter(sup_feats.size(1), args.hidden_dim, args.alpha,  args.beta, args.clip_scale).to(device)
        run_clip_adapter(sup_feats, sup_labels, val_feats, val_labels,
                            test_feats, test_labels, clip_w.T, adapter,
                            args.clip_scale, args.alpha_max, args.beta_max, args.alpha_steps, args.beta_steps,
                            args.lr, args.epochs, args.batch_size, save_dir, device, allfeas = all_feats)
        
    elif args.method == 'tip_adapter':
        run_tip_adapter(sup_feats, sup_labels,
                        val_feats, val_labels,
                        test_feats, test_labels,
                        clip_w,
                        init_beta=args.beta_max, init_alpha=args.alpha_max,
                        beta_steps=args.beta_steps, alpha_steps=args.alpha_steps,
                        save_dir=save_dir)
        
    elif args.method == 'tip_adapter_f':
        run_tip_adapter_f(sup_feats, sup_labels,
                          val_feats, val_labels,
                          test_feats, test_labels,
                          clip_w,
                          clip_scale=args.clip_scale,
                          beta_max=args.beta_max, alpha_max=args.alpha_max,
                          beta_steps=args.beta_steps, alpha_steps=args.alpha_steps,
                          lr=args.lr, epochs=args.epochs, batch_size=args.batch_size,
                          save_dir=save_dir, device=device)

    elif args.method == 'probing':
        from dataset_model import MLP
        # model = MLP(sup_feats.size(1), [args.hidden_dim], int(labels.max()) + 1 ).cuda()
        model = torch.nn.Linear(sup_feats.size(1), int(labels.max()) + 1 ).cuda()
        train_cls(sup_feats, sup_labels,
                                    val_feats, val_labels,
                                    test_feats, test_labels,
                                    model, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, save_dir=save_dir)
        # t=0
        
    elif args.method == 'meta_adapter':
        adapter = MetaAdapter(dim=sup_feats.size(1)).cuda()
        run_meta_adapter(sup_feats.T, sup_labels, test_feats, test_labels, clip_w, adapter,
                         epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, save_dir=save_dir)

        
    elif args.method == 'noise_adapter_feasaug':
        adapter = NoiseMLPAdapter(sup_feats.size(1), 2, hidden_dim=args.hidden_dim, num_classes= int(labels.max()) + 1)  
        run_noise_adapter_train(sup_feats, support_noise, sup_labels,
                                    val_feats, val_noise, val_labels,
                                    test_feats, test_noise, test_labels,
                                    adapter, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, save_dir=save_dir)
        
        checkpoint = torch.load(os.path.join(save_dir, 'best_adapter.pth'))
        best_val_acc_before = checkpoint['val_acc']
        best_test_acc_before = np.load( os.path.join(save_dir, 'acc.npy') )
        adapter.load_state_dict(checkpoint['adapter'])
        adapter.eval()

        gate = adapter.noise_mlp(support_noise)
        g1 = gate.detach().cpu().numpy()

        g1mean = torch.from_numpy(g1.mean(axis=0))
        sensitivity = torch.ones(g1mean.shape) *1
        
        ### feature channle via linear classifier
        sensitivity = sensitivity.cuda() 
        
        with torch.no_grad():
            gate = adapter.noise_mlp(noise.float().to(device))
            modulated_feat = all_feats * gate + all_feats

        sup_feats0 = sup_feats.clone()
        sup_feats = modulated_feat[tr][sup_idx]
        sup_gate = gate[tr][sup_idx]
        val_feats = modulated_feat[vl]
        test_feats = modulated_feat[te]
        
        classifier = nn.Sequential(
            nn.Linear(sup_feats.size(1), args.hidden_dim),
            nn.BatchNorm1d(args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, int(labels.max()) + 1)
        )

        run_noise_adapter_feasaug_trainv2(sup_feats, sup_labels,
                                    val_feats, val_labels,
                                    test_feats, test_labels, gate, sup_feats0, best_val_acc_before, best_test_acc_before,
                                    classifier, epochs=args.epochs, batch_size = args.batch_size, lr=args.lr, save_dir=save_dir)
        

        
        



p = argparse.ArgumentParser()


###spi-real dataset
p.add_argument('--image_feat_path', default='./totfeas_clipvitb32_spirealrawreorg.npy', type=str)
p.add_argument('--noise_path', default='./noise_spirealrawreorg.npy', type=str)
p.add_argument('--use_noise', default=False, type=bool)
p.add_argument('--label_path', default='./totlabels_clipvitb32_spirealrawreorg.npy', type=str)
p.add_argument('--text_feat_path',default='./textfeas_clipvitb32_val.npy', type=str)

p.add_argument('--output_dir', default='./experiments')
p.add_argument('--method', default='noise_adapter_feasaug', type=str, help = ['clip_adapter','tip_adapter','tip_adapter_f'])  
p.add_argument('--train_ratio', type=float, default=0.6)
p.add_argument('--val_ratio', type=float, default=0.1)
p.add_argument('--shots', default=1, type=int)
p.add_argument('--seed', type=int, default=1)
p.add_argument('--device', default='cuda')
# model args
p.add_argument('--hidden_dim', type=int, default=512)
p.add_argument('--alpha', type=float, default=0.5)
p.add_argument('--beta', type=float, default=0)
p.add_argument('--clip_scale', type=float, default=100.0)
p.add_argument('--lr', type=float, default=0.001)
p.add_argument('--epochs', type=int, default=100)
p.add_argument('--batch_size', type=int, default=64)
# search args
p.add_argument('--beta_max', type=float, default=7)
p.add_argument('--alpha_max', type=float, default=3)
p.add_argument('--beta_steps', type=int, default=100)
p.add_argument('--alpha_steps', type=int, default=20)
args = p.parse_args()

methods = ['noise_adapter_feasaug']
# methods = [ 'probing', 'clip_adapter', 'tip_adapter', 'tip_adapter_f', 'meta_adapter', 'noise_adapter_feasaug']
shots = [1,2]
num_exp = 1
for im in methods:
    for ishot in shots:
        args.method = im
        args.shots = ishot
        print(im, ishot)
        for iexp in range(num_exp):
            args.seed = iexp + 1
            main(args)