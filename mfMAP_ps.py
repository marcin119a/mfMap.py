#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mfMAP_ps.py  –  Multi-omics Fusion MAP for Primary Site prediction

Architecture (identical to mfMAP):
  Encoder  : DNA view + RNA view  →  μ, log_var
  Latent z : z ~ N(μ, σ)
  Decoder  : z  →  recon_DNA, recon_RNA
  Classifier (NEW): μ  →  primary_site label

Difference vs. mfMAP:
  - The classifier predicts PRIMARY SITE (tissue / organ of origin)
    instead of cancer-type-specific molecular subtypes.
  - Primary site labels are read from `primary_site` column of the
    label file (or any column name given via --ps_col flag).
  - Label mapping is built dynamically from the data (no hard-coded
    organ-specific dictionaries).
  - Works across cancer types in a single run.

Required label file columns:
  barcode   – sample ID
  type      – 'tumor' or 'cell'
  primary_site (or value of --ps_col)  –  e.g. 'breast', 'colon', …
                                          'NOLBL' for unlabeled samples
"""

import torch
import numpy as np
import math
import pandas as pd
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from earlystoping import Earlystopping
from sklearn import metrics
import tensorflow.compat.v1 as tf
import os
from pathlib import Path
from lr_scheduler import ReduceLROnPlateau

PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
import re

tf.app.flags.DEFINE_string('f', '', 'kernel')
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_bool('parallel', True, 'Parallelism of model')
flags.DEFINE_bool('output_loss_record', True, 'Output loss record')
flags.DEFINE_bool('early_stopping', False, 'Early stopping')
flags.DEFINE_integer('random_seed', 42, 'Random seed for torch.')
flags.DEFINE_integer('batch_size', 32, 'Batch size.')
flags.DEFINE_integer('latent_space_dim', 2, 'Latent space dimensionality.')
flags.DEFINE_float('learning_rate', 0.001152512, 'Initial learning rate.')
flags.DEFINE_integer('p1_epoch_num', 2, 'Unsupervised pre-training epochs.')
flags.DEFINE_integer('p2_epoch_num', 3000, 'Supervised training epochs.')
flags.DEFINE_integer('level_2_dim_dna', 1024, 'DNA encoder hidden dim 1.')
flags.DEFINE_integer('level_3_dim_dna', 512,  'DNA encoder hidden dim 2.')
flags.DEFINE_integer('level_2_dim_rna', 1024, 'RNA encoder hidden dim 1.')
flags.DEFINE_integer('level_3_dim_rna', 512,  'RNA encoder hidden dim 2.')
flags.DEFINE_integer('level_4_dim', 256, 'Shared encoder hidden dim.')
flags.DEFINE_integer('classifier_1_dim', 128, 'Classifier hidden dim 1.')
flags.DEFINE_integer('classifier_2_dim', 64,  'Classifier hidden dim 2.')
flags.DEFINE_string('input_path', 'data_fake', 'Root data directory.')
flags.DEFINE_string('organ', 'BRCA', 'Organ / dataset sub-folder.')
flags.DEFINE_string('input1_fn', 'features_mut_cnv_comb.txt', 'DNA view file.')
flags.DEFINE_string('input2_fn', 'features_exp.txt',           'RNA view file.')
flags.DEFINE_string('label_fn',  'dataset_labels.txt',         'Label file.')
# Column in the label file that contains primary site labels
flags.DEFINE_string('ps_col', 'primary_site', 'Primary site column in label file.')
flags.DEFINE_string('nolbl_str', 'NOLBL', 'String used for unlabeled samples.')
flags.DEFINE_integer('patience', 50, 'Early stopping patience.')
flags.DEFINE_float('beta', 0.9, 'Beta for hard-bootstrap loss.')
flags.DEFINE_string('separate_testing', 'yes', 'Use separate validation set.')
flags.DEFINE_string('use_cell', 'yes', 'Include cell line data.')
FLAGS.input_path = os.path.join(FLAGS.input_path, FLAGS.organ)
torch.manual_seed(FLAGS.random_seed)
torch.cuda.manual_seed_all(FLAGS.random_seed)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def prepare_input(data_location, label_fn, input1_fn, input2_fn):
    """Load DNA view, RNA view and primary-site labels.

    Label file must contain columns:
        barcode, type  (tumor | cell),  <ps_col>  (e.g. 'primary_site')

    Returns the same tuple structure as mfMAP.prepare_input_tum_cell so that
    the training loop can be kept identical.
    """
    dna_df = pd.read_table(os.path.join(data_location, input1_fn), index_col=0)
    rna_df = pd.read_table(os.path.join(data_location, input2_fn), index_col=0)
    label_df = pd.read_table(os.path.join(data_location, label_fn), index_col=None)

    if 'barcode' in label_df.columns:
        label_df.index = label_df['barcode']

    dna_df = dna_df.clip(1e-3, 0.999)
    # RNA preprocessing: log1p transform + per-gene min-max scaling to (0,1)
    # so BCE reconstruction loss remains numerically valid.
    rna_df = np.log1p(rna_df)
    rna_min = rna_df.min(axis=1)
    rna_max = rna_df.max(axis=1)
    rna_range = (rna_max - rna_min).replace(0, 1.0)
    rna_df = rna_df.sub(rna_min, axis=0).div(rna_range, axis=0)
    rna_df = rna_df.clip(1e-3, 0.99)

    ps_col = FLAGS.ps_col
    nolbl  = FLAGS.nolbl_str

    if ps_col not in label_df.columns:
        raise KeyError(f"Missing label column '{ps_col}' in {label_fn}")
    # Normalize primary-site labels to strings and replace missing values.
    label_df[ps_col] = label_df[ps_col].fillna(nolbl).astype(str).str.strip()
    label_df.loc[label_df[ps_col].eq('') | label_df[ps_col].str.lower().eq('nan'), ps_col] = nolbl

    # ---- Build primary-site → integer mapping dynamically ----------------
    unique_sites = sorted(label_df[ps_col].unique().tolist())
    # Place NOLBL last so its index is max (mirrors mfMAP convention)
    if nolbl in unique_sites:
        unique_sites.remove(nolbl)
    unique_sites.append(nolbl)          # NOLBL gets the highest index
    ps_mapping = {s: i for i, s in enumerate(unique_sites)}
    nolbl_idx  = ps_mapping[nolbl]

    label_df[ps_col] = label_df[ps_col].map(ps_mapping)

    # ---- Split: labeled tumors | unlabeled tumors | cell lines -----------
    nolbl_tumor_bc = label_df.barcode[
        (label_df[ps_col] == nolbl_idx) & (label_df['type'] == 'tumor')
    ].tolist()
    lbl_tumor_bc = label_df.barcode[
        (label_df[ps_col] != nolbl_idx) & (label_df['type'] == 'tumor')
    ].tolist()
    cell_bc = label_df.barcode[label_df['type'] == 'cell'].tolist()

    # cell-line dataset (used as nolbl_dataset / nolbl_loader)
    dcna_df  = dna_df[cell_bc]
    drna_df  = rna_df[cell_bc]
    dlabel   = label_df.loc[cell_bc]

    # Drop unlabeled tumors from the main matrix
    dna_df = dna_df.drop(nolbl_tumor_bc, axis=1)
    rna_df = rna_df.loc[:, list(dna_df.columns)]
    label_df = label_df.loc[list(dna_df.columns)]

    if FLAGS.separate_testing == 'yes':
        val_ratio = 0.09
        split_bc = label_df['barcode'].values if FLAGS.use_cell == 'yes' \
                   else label_df.loc[lbl_tumor_bc, 'barcode'].values
        split_lbl = label_df.loc[split_bc, ps_col].values

        split_lbl_counts = pd.Series(split_lbl).value_counts()
        safe_stratify = split_lbl if split_lbl_counts.min() >= 2 else None

        train_bc, val_bc, train_lbl, val_lbl = train_test_split(
            split_bc, split_lbl,
            test_size=val_ratio,
            random_state=FLAGS.random_seed,
            stratify=safe_stratify,
        )
        train_ds  = MultiOmiDataset(dna_df[train_bc], rna_df[train_bc], train_lbl)
        val_ds    = MultiOmiDataset(dna_df[val_bc],   rna_df[val_bc],   val_lbl)
        train_loader = DataLoader(train_ds, batch_size=FLAGS.batch_size,
                                  shuffle=True, num_workers=6)
        val_loader   = DataLoader(val_ds,   batch_size=FLAGS.batch_size,
                                  shuffle=True, num_workers=6)
    else:
        train_ds     = MultiOmiDataset(dna_df, rna_df, label_df[ps_col].values)
        train_loader = DataLoader(train_ds, batch_size=FLAGS.batch_size,
                                  shuffle=True, num_workers=6)

    nolbl_ds     = MultiOmiDataset(dcna_df, drna_df, dlabel[ps_col].values)
    nolbl_loader = DataLoader(nolbl_ds, batch_size=FLAGS.batch_size,
                              shuffle=True, num_workers=6)
    full_ds      = MultiOmiDataset(dna_df, rna_df, label_df[ps_col].values)
    full_loader  = DataLoader(full_ds, batch_size=FLAGS.batch_size, num_workers=6)

    if FLAGS.separate_testing == 'yes':
        return (nolbl_ds, nolbl_loader,
                full_ds, full_loader,
                train_ds, train_loader,
                val_ds, val_loader,
                dna_df, rna_df, label_df,
                nolbl_idx, ps_mapping)
    else:
        return (nolbl_ds, nolbl_loader,
                full_ds, full_loader,
                train_ds, train_loader,
                dna_df, rna_df, label_df,
                nolbl_idx, ps_mapping)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class MultiOmiDataset:
    def __init__(self, dna_df, rna_df, labels):
        self.dna_df = dna_df
        self.rna_df = rna_df
        self.labels = labels

    def __len__(self):
        return self.rna_df.shape[1]

    def __getitem__(self, index):
        dna = torch.Tensor(self.dna_df.iloc[:, index].values.copy())
        rna = torch.Tensor(self.rna_df.iloc[:, index].values.copy())
        return [dna, rna], self.labels[index]


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
def fc_layer(in_dim, out_dim, activation=1, dropout=False, dropout_p=0.5):
    if activation == 0:
        return nn.Sequential(nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim))
    elif activation == 2:
        return nn.Sequential(nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim),
                             nn.Sigmoid())
    else:
        if dropout:
            return nn.Sequential(nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim),
                                 nn.ReLU(), nn.Dropout(p=dropout_p))
        return nn.Sequential(nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim),
                             nn.ReLU())


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class mfMAP_ps(nn.Module):
    """VAE with separate DNA / RNA encoders and a primary-site classifier.

    The only architectural change vs. mfMAP is in what the classifier
    predicts (primary site, not tumor subtype).  All encoder/decoder layers
    are identical.
    """

    def __init__(self, input_dim_dna, input_dim_rna, classifier_out_dim):
        super().__init__()

        # ---- Encoder -------------------------------------------------
        self.e_fc1_dna      = fc_layer(input_dim_dna,  FLAGS.level_2_dim_dna)
        self.e_fc1_rna      = fc_layer(input_dim_rna,  FLAGS.level_2_dim_rna)
        self.e_fc2_dna      = fc_layer(FLAGS.level_2_dim_dna, FLAGS.level_3_dim_dna)
        self.e_fc2_rna      = fc_layer(FLAGS.level_2_dim_rna, FLAGS.level_3_dim_rna)
        self.e_fc3          = fc_layer(FLAGS.level_3_dim_dna + FLAGS.level_3_dim_rna,
                                       FLAGS.level_4_dim)
        self.e_fc4_mean     = fc_layer(FLAGS.level_4_dim, FLAGS.latent_space_dim, activation=0)
        self.e_fc4_log_var  = fc_layer(FLAGS.level_4_dim, FLAGS.latent_space_dim, activation=0)

        # ---- Decoder -------------------------------------------------
        self.d_fc4     = fc_layer(FLAGS.latent_space_dim, FLAGS.level_4_dim)
        self.d_fc3     = fc_layer(FLAGS.level_4_dim,
                                  FLAGS.level_3_dim_dna + FLAGS.level_3_dim_rna)
        self.d_fc2_dna = fc_layer(FLAGS.level_3_dim_dna, FLAGS.level_2_dim_dna)
        self.d_fc2_rna = fc_layer(FLAGS.level_3_dim_rna, FLAGS.level_2_dim_rna)
        self.d_fc1_dna = fc_layer(FLAGS.level_2_dim_dna, input_dim_dna, activation=2)
        self.d_fc1_rna = fc_layer(FLAGS.level_2_dim_rna, input_dim_rna, activation=2)

        # ---- Primary-site classifier (replaces subtype classifier) ---
        self.c_fc1 = fc_layer(FLAGS.latent_space_dim,   FLAGS.classifier_1_dim)
        self.c_fc2 = fc_layer(FLAGS.classifier_1_dim,   FLAGS.classifier_2_dim)
        self.c_fc3 = fc_layer(FLAGS.classifier_2_dim,   classifier_out_dim, activation=0)

        if FLAGS.parallel:
            for m in [self.e_fc1_dna, self.e_fc1_rna, self.e_fc2_dna,
                      self.e_fc2_rna, self.e_fc3, self.e_fc4_mean,
                      self.e_fc4_log_var]:
                m.to('cuda:0')
            for m in [self.d_fc4, self.d_fc3, self.d_fc2_dna, self.d_fc2_rna,
                      self.d_fc1_dna, self.d_fc1_rna,
                      self.c_fc1, self.c_fc2, self.c_fc3]:
                m.to('cuda:1')

    # ---- Encoder ---------------------------------------------------------
    def encode(self, x):
        dna2 = self.e_fc1_dna(x[0])
        rna2 = self.e_fc1_rna(x[1])
        dna3 = self.e_fc2_dna(dna2)
        rna3 = self.e_fc2_rna(rna2)
        h4   = self.e_fc3(torch.cat([dna3, rna3], dim=1))
        return self.e_fc4_mean(h4), self.e_fc4_log_var(h4)

    # ---- Reparameterization ----------------------------------------------
    def reparameterize(self, mean, log_var):
        sigma = torch.exp(0.5 * log_var)
        return mean + torch.randn_like(sigma) * sigma

    # ---- Decoder ---------------------------------------------------------
    def decode(self, z):
        h4   = self.d_fc4(z)
        h3   = self.d_fc3(h4)
        dna3 = h3.narrow(1, 0, FLAGS.level_3_dim_dna)
        rna3 = h3.narrow(1, FLAGS.level_3_dim_dna, FLAGS.level_3_dim_rna)
        return [self.d_fc1_dna(self.d_fc2_dna(dna3)),
                self.d_fc1_rna(self.d_fc2_rna(rna3))]

    # ---- Primary-site classifier -----------------------------------------
    def classifier(self, mean):
        return self.c_fc3(self.c_fc2(self.c_fc1(mean)))

    # ---- Forward ---------------------------------------------------------
    def forward(self, x, y):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        cls_input = mean
        if FLAGS.parallel:
            z         = z.to('cuda:1')
            cls_input = cls_input.to('cuda:1')
        recon_x = self.decode(z)
        pred_y  = self.classifier(cls_input)
        return z, recon_x, mean, log_var, pred_y


# ---------------------------------------------------------------------------
# Loss functions  (identical to mfMAP)
# ---------------------------------------------------------------------------
def dna_recon_loss(recon_x, x):
    return F.binary_cross_entropy(recon_x[0], x[0], reduction='mean')

def rna_recon_loss(recon_x, x):
    return F.binary_cross_entropy(recon_x[1], x[1], reduction='mean')

def kl_loss(mean, log_var):
    return -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

def classifier_er(pred_y, y, beta=0.5):
    bootstrap = -(1.0 - beta) * torch.sum(
        F.softmax(pred_y, dim=1) * F.log_softmax(pred_y, dim=1), dim=1)
    return torch.sum(bootstrap)

def classifier_hb_loss(pred_y, y, tmpidx, beta=0.5):
    bootstrap = -(1.0 - beta) * torch.sum(
        F.softmax(pred_y, dim=1) * F.log_softmax(pred_y, dim=1), dim=1)
    return beta * F.cross_entropy(pred_y[tmpidx], y[tmpidx], reduction='sum') \
           + torch.sum(bootstrap)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def run_train():
    if FLAGS.separate_testing == 'yes':
        (nolbl_ds, nolbl_loader,
         full_ds, full_loader,
         train_ds, train_loader,
         val_ds, val_loader,
         dna_df, rna_df, label_df,
         nolbl_idx, ps_mapping) = prepare_input(
            FLAGS.input_path, FLAGS.label_fn, FLAGS.input1_fn, FLAGS.input2_fn)
    else:
        (nolbl_ds, nolbl_loader,
         full_ds, full_loader,
         train_ds, train_loader,
         dna_df, rna_df, label_df,
         nolbl_idx, ps_mapping) = prepare_input(
            FLAGS.input_path, FLAGS.label_fn, FLAGS.input1_fn, FLAGS.input2_fn)

    ps_col = FLAGS.ps_col
    input_dim_dna = dna_df.shape[0]
    input_dim_rna = rna_df.shape[0]
    # Number of real primary-site classes (exclude NOLBL)
    classifier_out_dim = len(ps_mapping) - 1

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    FLAGS.parallel = torch.cuda.device_count() > 1 and FLAGS.parallel

    # ---- Output naming ---------------------------------------------------
    fn_tag = re.sub(r'features_|\.txt', '', '_'.join([FLAGS.input1_fn, FLAGS.input2_fn]))
    run_tag = 'ps_hb_{}_wait{}_st{}_uc{}_p1{}_bt{}'.format(
        fn_tag, FLAGS.patience, FLAGS.separate_testing,
        FLAGS.use_cell, FLAGS.p1_epoch_num, FLAGS.beta)
    out_dir = 'results/{}/'.format(FLAGS.organ)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join('ssd', FLAGS.organ), exist_ok=True)

    # ---- Model -----------------------------------------------------------
    if FLAGS.parallel:
        model = mfMAP_ps(input_dim_dna, input_dim_rna, classifier_out_dim)
    else:
        model = mfMAP_ps(input_dim_dna, input_dim_rna, classifier_out_dim).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print('Primary-site classifier  |  parameters: {:,}'.format(total_params))
    print('Primary-site mapping:', ps_mapping)

    if FLAGS.early_stopping:
        early_stop = Earlystopping(
            number=FLAGS.patience,
            path='ssd/{}/{}_{}D_checkpoint.pt'.format(
                FLAGS.organ, run_tag, FLAGS.latent_space_dim))

    optimizer = optim.Adam(model.parameters(), lr=FLAGS.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=50, min_lr=1e-5)
    train_writer = SummaryWriter(log_dir='logs/ps_train')
    val_writer   = SummaryWriter(log_dir='logs/ps_val')

    total_epochs = FLAGS.p1_epoch_num + FLAGS.p2_epoch_num
    loss_arr    = np.zeros((15, total_epochs + 1))
    metrics_arr = np.zeros((4,  total_epochs + 1))

    # ------------------------------------------------------------------
    def train_epoch(e_idx, e_total, k_recon, k_kl, k_cls):
        model.train()
        t_dna_r = t_rna_r = t_kl = t_cls = t_er = t_tot = t_correct = 0
        for sample in train_loader:
            data, y = sample
            y = y.detach().clone()
            for i in range(2):
                data[i] = data[i].to(device)
            y = y.to(device)

            optimizer.zero_grad()
            _, recon, mean, log_var, pred_y = model(data, y)

            if FLAGS.parallel:
                for i in range(2):
                    recon[i] = recon[i].to('cuda:0')
                pred_y = pred_y.to('cuda:0')

            l_dna = dna_recon_loss(recon, data)
            l_rna = rna_recon_loss(recon, data)
            l_kl  = kl_loss(mean, log_var)
            labeled = [i for i in range(len(y)) if y[i] != nolbl_idx]
            l_er  = classifier_er(pred_y, y, beta=FLAGS.beta)
            l_cls = classifier_hb_loss(pred_y, y, labeled, beta=FLAGS.beta)
            loss  = k_recon * (l_dna + l_rna) + k_kl * l_kl + k_cls * l_cls
            loss.backward()

            with torch.no_grad():
                _, predicted = torch.max(F.softmax(pred_y, dim=1), 1)
                t_correct += (predicted[labeled] == y[labeled]).sum().item()
                t_dna_r += l_dna.item()
                t_rna_r += l_rna.item()
                t_kl    += l_kl.item()
                t_cls   += l_cls.item()
                t_er    += l_er.item()
                t_tot   += loss.item()
            optimizer.step()

        n = len(train_ds)
        labeled_total = np.sum(np.array(train_ds.labels) != nolbl_idx)
        acc = t_correct / labeled_total * 100 if labeled_total else 0.0
        print('Epoch {:3d}/{:3d}  [train]  '
              'DNA:{:.3f}  RNA:{:.3f}  KL:{:.3f}  Cls:{:.3f}  ACC:{:.1f}%'.format(
                  e_idx + 1, e_total, t_dna_r/n, t_rna_r/n, t_kl/n, t_cls/n, acc))
        loss_arr[[0,1,2,3,4,11,13], e_idx] = [
            t_dna_r/n, t_rna_r/n, t_kl/n, t_cls/n, acc, t_tot/n, t_er/n]
        for tag, val in [('DNA recon', t_dna_r/n), ('RNA recon', t_rna_r/n),
                         ('KL',t_kl/n), ('Cls',t_cls/n), ('ACC',acc),
                         ('Total',t_tot/n)]:
            train_writer.add_scalar(tag, val, e_idx)
        return acc, t_tot/n, t_er/n

    # ------------------------------------------------------------------
    if FLAGS.separate_testing == 'yes':
        def val_epoch(e_idx, k_recon, k_kl, k_cls, get_metrics=False):
            model.eval()
            v_dna_r = v_rna_r = v_kl = v_cls = v_er = v_tot = v_correct = 0
            yl_store  = torch.tensor([0])
            pl_store  = torch.tensor([0])
            with torch.no_grad():
                for sample in val_loader:
                    data, y = sample
                    for i in range(2):
                        data[i] = data[i].to(device)
                    y = y.to(device)
                    _, recon, mean, log_var, pred_y = model(data, y)
                    if FLAGS.parallel:
                        for i in range(2):
                            recon[i] = recon[i].to('cuda:0')
                        pred_y = pred_y.to('cuda:0')

                    l_dna = dna_recon_loss(recon, data)
                    l_rna = rna_recon_loss(recon, data)
                    l_kl  = kl_loss(mean, log_var)
                    labeled = [i for i in range(len(y)) if y[i] != nolbl_idx]
                    l_er  = classifier_er(pred_y, y, beta=FLAGS.beta)
                    l_cls = classifier_hb_loss(pred_y, y, labeled, beta=FLAGS.beta)
                    loss  = k_recon*(l_dna+l_rna) + k_kl*l_kl + k_cls*l_cls

                    _, predicted = torch.max(F.softmax(pred_y, dim=1), 1)
                    v_correct += (predicted[labeled] == y[labeled]).sum().item()
                    yl_store  = torch.cat([yl_store,  y[labeled].cpu()])
                    pl_store  = torch.cat([pl_store,  predicted[labeled].cpu()])
                    v_dna_r += l_dna.item(); v_rna_r += l_rna.item()
                    v_kl    += l_kl.item();  v_cls   += l_cls.item()
                    v_er    += l_er.item();  v_tot   += loss.item()

            n = len(val_ds)
            labeled_total = np.sum(np.array(val_ds.labels) != nolbl_idx)
            acc = v_correct / labeled_total * 100 if labeled_total else 0.0
            out_y  = yl_store[1:].numpy()
            out_py = pl_store[1:].numpy()
            if get_metrics:
                metrics_arr[0, e_idx] = metrics.accuracy_score(out_y, out_py)
                metrics_arr[1, e_idx] = metrics.precision_score(out_y, out_py, average='weighted', zero_division=0)
                metrics_arr[2, e_idx] = metrics.recall_score(out_y, out_py, average='weighted', zero_division=0)
                metrics_arr[3, e_idx] = metrics.f1_score(out_y, out_py, average='weighted', zero_division=0)
            print('           [val]    '
                  'DNA:{:.3f}  RNA:{:.3f}  KL:{:.3f}  Cls:{:.3f}  ACC:{:.1f}%'.format(
                      v_dna_r/n, v_rna_r/n, v_kl/n, v_cls/n, acc))
            loss_arr[[5,6,7,8,9,12,14], e_idx] = [
                v_dna_r/n, v_rna_r/n, v_kl/n, v_cls/n, acc, v_tot/n, v_er/n]
            for tag, val in [('DNA recon', v_dna_r/n), ('RNA recon', v_rna_r/n),
                              ('KL',v_kl/n), ('Cls',v_cls/n), ('ACC',acc),
                              ('Total',v_tot/n)]:
                val_writer.add_scalar(tag, val, e_idx)
            return acc, out_py, v_cls/n, v_tot/n

    # ------------------------------------------------------------------
    def get_decoder_weight(dataset=None):
        if dataset is None:
            dataset = full_ds
        model.eval()
        pd_ = {name: param.data.cpu().numpy().T
               for name, param in model.named_parameters()
               if re.search(r'd_fc\S*\.0\.weight', name)}
        w_dna = pd_['d_fc4.0.weight']
        for key in ['d_fc3.0.weight', 'd_fc2_dna.0.weight', 'd_fc1_dna.0.weight']:
            if key == 'd_fc3.0.weight':
                tmp = pd_[key][:, :pd_['d_fc2_dna.0.weight'].shape[0]]
            else:
                tmp = pd_[key]
            w_dna = np.dot(w_dna, tmp)
        w_rna = pd_['d_fc4.0.weight']
        for key in ['d_fc3.0.weight', 'd_fc2_rna.0.weight', 'd_fc1_rna.0.weight']:
            if key == 'd_fc3.0.weight':
                tmp = pd_[key][:, pd_['d_fc2_dna.0.weight'].shape[0]:]
            else:
                tmp = pd_[key]
            w_rna = np.dot(w_rna, tmp)
        w_dna = pd.DataFrame(w_dna, columns=dataset.dna_df.index)
        w_rna = pd.DataFrame(w_rna, columns=dataset.rna_df.index)
        return w_dna, w_rna

    def save_output(prefix, data_loader=None, dataset=None):
        if data_loader is None: data_loader = full_loader
        if dataset is None:     dataset     = full_ds
        model.eval()
        z_store    = torch.zeros(1, FLAGS.latent_space_dim).to(device)
        recon_dna_store = torch.zeros(1, dataset.dna_df.shape[0]).to(device)
        recon_rna_store = torch.zeros(1, dataset.rna_df.shape[0]).to(device)
        pred_store = torch.tensor([0]).to(device)
        with torch.no_grad():
            for sample in data_loader:
                d, y = sample
                y = y.to(device)
                for j in range(2):
                    d[j] = d[j].to(device)
                _, recon, d_z, _, pred_y = model(d, y)
                for i in range(2):
                    recon[i] = recon[i].to(device)
                pred_y = pred_y.to(device)
                _, predicted = torch.max(F.softmax(pred_y, dim=1), 1)
                z_store       = torch.cat([z_store,       d_z],         0)
                recon_dna_store = torch.cat([recon_dna_store, recon[0]], 0)
                recon_rna_store = torch.cat([recon_rna_store, recon[1]], 0)
                pred_store    = torch.cat([pred_store, predicted])

        barcodes = dataset.dna_df.columns
        base = out_dir + run_tag + '_{}_{}D'.format(prefix, FLAGS.latent_space_dim)

        pd.DataFrame(z_store[1:].cpu().numpy(), index=barcodes).to_csv(
            base + '_latent_space.tsv', sep='\t')
        pd.DataFrame(recon_dna_store[1:].cpu().numpy().T,
                     index=dataset.dna_df.index, columns=barcodes).to_csv(
            base + '_recon_dna.tsv', sep='\t')
        pd.DataFrame(recon_rna_store[1:].cpu().numpy().T,
                     index=dataset.rna_df.index, columns=barcodes).to_csv(
            base + '_recon_rna.tsv', sep='\t')
        pd.DataFrame(pred_store[1:].cpu().numpy(), index=barcodes).to_csv(
            base + '_pred_primary_site.tsv', sep='\t')

        w_dna, w_rna = get_decoder_weight(dataset)
        w_dna.to_csv(base + '_decoder_w_dna.tsv', sep='\t')
        w_rna.to_csv(base + '_decoder_w_rna.tsv', sep='\t')

        if FLAGS.separate_testing == 'yes':
            np.savetxt(base + '_metrics.tsv',    metrics_arr, delimiter='\t')
            np.savetxt(base + '_val_index.tsv',
                       np.array(val_ds.dna_df.columns.tolist()),
                       delimiter='\t', fmt='%s')
        if FLAGS.output_loss_record:
            np.savetxt(base + '_loss_record.tsv', loss_arr, delimiter='\t')

    # ------------------------------------------------------------------ training loop
    best_train_acc = 0.0
    best_val_acc   = 0.0
    best_train_er  = math.inf

    print('\n--- UNSUPERVISED PRE-TRAINING ---')
    for ep in range(FLAGS.p1_epoch_num):
        train_epoch(ep, total_epochs, k_recon=1, k_kl=1, k_cls=0)
        if FLAGS.separate_testing == 'yes':
            val_epoch(ep, k_recon=1, k_kl=1, k_cls=0)

    print('\n--- SUPERVISED TRAINING (primary site) ---')
    for ep in range(FLAGS.p1_epoch_num, total_epochs):
        acc, tot_loss, er = train_epoch(ep, total_epochs, k_recon=1, k_kl=1, k_cls=1)
        if acc > best_train_acc:
            best_train_acc = acc
            torch.save(model.state_dict(),
                       'ssd/{}/{}_{}D_best_train_acc.pt'.format(
                           FLAGS.organ, run_tag, FLAGS.latent_space_dim))
        if er < best_train_er and acc >= best_train_acc:
            best_train_er = er
            torch.save(model.state_dict(),
                       'ssd/{}/{}_{}D_best_train_acc.pt'.format(
                           FLAGS.organ, run_tag, FLAGS.latent_space_dim))

        if FLAGS.separate_testing == 'yes':
            get_m = (ep == total_epochs - 1)
            val_acc, _, val_cls_loss, val_tot_loss = val_epoch(
                ep, k_recon=1, k_kl=1, k_cls=1, get_metrics=get_m)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(),
                           'ssd/{}/{}_{}D_best_val_acc.pt'.format(
                               FLAGS.organ, run_tag, FLAGS.latent_space_dim))
            if FLAGS.early_stopping:
                early_stop(model, val_acc)
                if early_stop.stop_now:
                    print('Early stopping at epoch', ep + 1)
                    break

        if scheduler.step(tot_loss) and best_train_acc >= 100:
            break

    # ---- Save outputs ----------------------------------------------------
    print('\nEncoding data into latent space ...')
    if FLAGS.separate_testing == 'yes':
        if FLAGS.early_stopping:
            loss_arr[10, 0] = FLAGS.p1_epoch_num + early_stop.best_epoch_num
            model.load_state_dict(torch.load(
                'ssd/{}/{}_{}D_checkpoint.pt'.format(
                    FLAGS.organ, run_tag, FLAGS.latent_space_dim)))
            save_output('early_stop_full',  full_loader,  full_ds)
            save_output('early_stop_nolbl', nolbl_loader, nolbl_ds)
        else:
            model.load_state_dict(torch.load(
                'ssd/{}/{}_{}D_best_val_acc.pt'.format(
                    FLAGS.organ, run_tag, FLAGS.latent_space_dim)))
            save_output('best_val_full',  full_loader,  full_ds)
            save_output('best_val_nolbl', nolbl_loader, nolbl_ds)
            model.load_state_dict(torch.load(
                'ssd/{}/{}_{}D_best_train_acc.pt'.format(
                    FLAGS.organ, run_tag, FLAGS.latent_space_dim)))
            save_output('best_train_full',  full_loader,  full_ds)
            save_output('best_train_nolbl', nolbl_loader, nolbl_ds)
    else:
        model.load_state_dict(torch.load(
            'ssd/{}/{}_{}D_best_train_acc.pt'.format(
                FLAGS.organ, run_tag, FLAGS.latent_space_dim)))
        save_output('best_train_full',  full_loader,  full_ds)
        save_output('best_train_nolbl', nolbl_loader, nolbl_ds)


# ---------------------------------------------------------------------------
run_train()