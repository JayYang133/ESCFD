import math
import random

from PIL import Image
import blobfile as bf
import numpy as np
from torch.utils.data import DataLoader, Dataset

import scanpy as sc
import torch
import sys
sys.path.append('..')
from VAE.VAE_model import VAE
from sklearn.preprocessing import LabelEncoder

def stabilize(expression_matrix):
    ''' Use Anscombes approximation to variance stabilize Negative Binomial data
    See https://f1000research.com/posters/4-1041 for motivation.
    Assumes columns are samples, and rows are genes
    '''
    from scipy import optimize
    phi_hat, _ = optimize.curve_fit(lambda mu, phi: mu + phi * mu ** 2, expression_matrix.mean(1), expression_matrix.var(1))

    return np.log(expression_matrix + 1. / (2 * phi_hat[0]))

def load_VAE(vae_path, num_gene, hidden_dim):
    autoencoder = VAE(
        num_genes=num_gene,
        device='cuda',
        seed=0,
        loss_ae='mse',
        hidden_dim=hidden_dim,
        decoder_activation='ReLU',
    )
    autoencoder.load_state_dict(torch.load(vae_path))
    return autoencoder


def load_data(
    *,
    data_dir,
    batch_size,
    vae_path=None,
    deterministic=False,
    train_vae=False,
    hidden_dim=128,
):
    """
    For a dataset, create a generator over (cells, kwargs) pairs.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param vae_path: the path to save autoencoder / read autoencoder checkpoint.
    :param deterministic: if True, yield results in a deterministic order.
    :param train_vae: train the autoencoder or use the autoencoder.
    :param hidden_dim: the dimensions of latent space. If use pretrained weight, set 128
    """
    if not data_dir:
        raise ValueError("unspecified data directory")

    adata = sc.read_10x_h5(data_dir)
    
    # preporcess the data. modify this part if use your own dataset. the gene expression must first norm1e4 then log1p
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.filter_cells(adata, min_genes=10)
    adata.var_names_make_unique()

    # if generate ood data, left this as the ood data
    # selected_cells = (adata.obs['organ'] != 'mammary') | (adata.obs['celltype'] != 'B cell')  
    # adata = adata[selected_cells, :]  
    classes = np.zeros(adata.n_obs, dtype=np.int64)

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    cell_data = adata.X.toarray()

    # turn the gene expression into latent space. use this if training the diffusion backbone.
    if not train_vae:
        num_gene = cell_data.shape[1]
        autoencoder = load_VAE(vae_path,num_gene,hidden_dim)
        cell_data = autoencoder(torch.tensor(cell_data).cuda(),return_latent=True)
        cell_data = cell_data.cpu().detach().numpy()
    
    dataset = CellDataset(
        cell_data
    )
    return dataset
    '''if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader'''


class CellDataset(Dataset):
    def __init__(
        self,
        cell_data

    ):
        super().__init__()
        self.data = cell_data


    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        arr = self.data[idx]
        return arr
        
'''class CellDataset(Dataset):
    def __init__(
        self,
        cell_data,
        class_name
    ):
        super().__init__()
        self.data = cell_data
        self.class_name = class_name

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        arr = self.data[idx]
        out_dict = {}
        if self.class_name is not None:
            out_dict["y"] = np.array(self.class_name[idx], dtype=np.int64)
        return arr, out_dict'''




'''import numpy as np
from torch.utils.data import DataLoader, Dataset

import scanpy as sc
import pandas as pd
import torch
import sys
sys.path.append('..')
from VAE.VAE_model import VAE
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder


class Encoder(nn.Module):
    def __init__(self, latent_dim=128, input_size=None):
        """
        The Encoder class
        """
        if latent_dim is None or input_size is None:
            raise ValueError('Must explicitly declare input size and latent space dimension')

        super(Encoder, self).__init__()
        self.inp_dim = input_size
        self.zdim = latent_dim

        # feed forward layers
        self.enc_sequential = nn.Sequential(
            nn.Linear(self.inp_dim, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),

            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),

            nn.Linear(256,128),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )

    def forward(self, x):
        return self.enc_sequential(x)

def load_data(
    *,
    data_dir,
    batch_size,
    deterministic=False,
    hidden_dim=128,
):
    """
    For a dataset, create a generator over (cells, kwargs) pairs.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param vae_path: the path to save autoencoder / read autoencoder checkpoint.
    :param deterministic: if True, yield results in a deterministic order.
    :param train_vae: train the autoencoder or use the autoencoder.
    :param hidden_dim: the dimensions of latent space. If use pretrained weight, set 128
    """
    if not data_dir:
        raise ValueError("unspecified data directory")

    # 读取数据
    adata = sc.read_10x_h5(data_dir)
    print(adata)
    adata.var_names_make_unique()
    sc.pp.filter_cells(adata, min_genes=10)
    sc.pp.filter_genes(adata, min_cells=3)

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    cell_data = adata.X.toarray()

    # 如果不训练 VAE 模型，加载预训练的 VAE 模型

    num_gene = cell_data.shape[1]
    encoder = Encoder(latent_dim=hidden_dim, input_size=num_gene).cuda()
    cell_data = encoder(torch.tensor(cell_data).cuda())
    cell_data = cell_data.cpu().detach().numpy()

    dataset = CellDataset(
        cell_data
    )
    return dataset
    

class CellDataset(Dataset):
    def __init__(
        self,
        cell_data

    ):
        super().__init__()
        self.data = cell_data


    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        arr = self.data[idx]
        return arr
'''