import anndata
import scipy.sparse as sp
import numpy as np
import scanpy as sc
import torch
from torch.utils.data.dataset import Dataset


class SingleCellData(Dataset):
    def __init__(self, data_path, num_gene, shared_gene_mask=None, filter_mask=None, normalized=True):
        self.data_path = data_path
        self.num_gene = num_gene
        self.gene_mask = shared_gene_mask
        self.filter_mask = filter_mask
        self.normalized = normalized
        self.anndata = anndata.read_h5ad(data_path)
        self.num_class = len(self.anndata.obs['labels'].unique().tolist())
        
        if self.filter_mask is None:
            self.filter_mask, _ = sc.pp.filter_genes(self.anndata, min_counts=1, inplace=False)
        else:
            print("use share filter")
        self.anndata = self.anndata[:, self.filter_mask]

        if self.gene_mask is None:
            self.gene_mask = self.select_gene(data=self.anndata.X, num_gene=self.num_gene)
        else:
            print("use share gene")
        if self.normalized:
            anndata_norm = self.anndata.copy()
            sc.pp.normalize_per_cell(anndata_norm, counts_per_cell_after=1_000_000)
            sc.pp.log1p(anndata_norm)
            anndata_norm.X = anndata_norm.X.toarray()
            anndata_norm.X -= anndata_norm.X.mean(axis=0)
            anndata_norm.X /= anndata_norm.X.std(axis=0)
            if np.isnan(anndata_norm.X).any():
                print("Detect nan, fix by nan to num")
                anndata_norm.X = np.nan_to_num(anndata_norm.X)
                assert(not np.isnan(anndata_norm.X).any())

            self.anndata_preprocessed = anndata_norm[:, self.gene_mask].copy()
        else:
            self.anndata_preprocessed = self.anndata[:, self.gene_mask].copy()      
        self.X = torch.tensor(self.anndata_preprocessed.X)
        self.id_to_batch, self.batch_to_id = self.get_batch_map()
        self.id_to_cell, self.cell_to_id = self.get_cell_map()
        
        
        self.cell_label_tensor = torch.tensor([self.cell_to_id[e] for e in self.anndata.obs['labels']])
        self.batch_id_tensor = torch.tensor([self.batch_to_id[e] for e in self.anndata.obs['batch_id']])
        
    def __getitem__(self, index):
        return self.X[index], self.cell_label_tensor[index], self.batch_id_tensor[index]
            
    def __len__(self):
        return len(self.anndata)
    
    def get_batch_map(self):
        num_batch_type = len(self.anndata.obs['batch_id'].unique().tolist())
        id2batch = dict(zip(list(range(num_batch_type)), self.anndata.obs['batch_id'].unique().tolist()))
        batch2id = {v: k for k,v in id2batch.items()}
        return id2batch, batch2id
    
    def get_cell_map(self):
        num_cell_type = len(self.anndata.obs['labels'].unique().tolist())
        id2cell = dict(zip(list(range(num_cell_type)), self.anndata.obs['labels'].unique().tolist()))
        cell2id = {v:k for k,v in id2cell.items()}
        return id2cell, cell2id
    
    def select_gene(self, data,\
                    num_gene,\
                    threshold=0,\
                    atleast=10,\
                    decay=1,
                    xoffset=5,\
                    yoffset=0.02):
        
        if sp.issparse(data):
            zeroRate = 1 - np.squeeze(np.array((data > threshold).mean(axis=0)))
            A = data.multiply(data > threshold)
            A.data = np.log2(A.data)
            meanExpr = np.zeros_like(zeroRate) * np.nan
            detected = zeroRate < 1
            meanExpr[detected] = np.squeeze(np.array(A[:, detected].mean(axis=0))) / (
                1 - zeroRate[detected]
            )
        else:
            zeroRate = 1 - np.mean(data > threshold, axis=0)
            meanExpr = np.zeros_like(zeroRate) * np.nan
            detected = zeroRate < 1
            meanExpr[detected] = np.nanmean(
                np.where(data[:, detected] > threshold, np.log2(data[:, detected]), np.nan),
                axis=0,
            )

        lowDetection = np.array(np.sum(data > threshold, axis=0)).squeeze() < atleast
        # lowDetection = (1 - zeroRate) * data.shape[0] < atleast - .00001
        zeroRate[lowDetection] = np.nan
        meanExpr[lowDetection] = np.nan

        if self.num_gene is not None:
            up = 10
            low = 0
            for t in range(100):
                nonan = ~np.isnan(zeroRate)
                selected = np.zeros_like(zeroRate).astype(bool)
                selected[nonan] = (
                    zeroRate[nonan] > np.exp(-decay * (meanExpr[nonan] - xoffset)) + yoffset
                )
                if np.sum(selected) == num_gene:
                    break
                elif np.sum(selected) < num_gene:
                    up = xoffset
                    xoffset = (xoffset + low) / 2
                else:
                    low = xoffset
                    xoffset = (xoffset + up) / 2
            print("Chosen offset: {:.2f}".format(xoffset))
        else:
            nonan = ~np.isnan(zeroRate)
            selected = np.zeros_like(zeroRate).astype(bool)
            selected[nonan] = (
                zeroRate[nonan] > np.exp(-decay * (meanExpr[nonan] - xoffset)) + yoffset
            )

        return selected
