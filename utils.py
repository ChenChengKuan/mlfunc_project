import time
import os
import anndata
import torch
import torch.nn as nn
import numpy as np
import scanpy as sc
import scipy.sparse as sp
import copy
import json


def select_gene(data,\
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

        if num_gene is not None:
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

def extract_shared_gene(data_ref, data_trg, num_gene):
    data_ref = anndata.read_h5ad(data_ref)
    data_trg = anndata.read_h5ad(data_trg)
    sc.pp.filter_genes(data_ref, min_counts=1)
    shared_genes = data_ref.var_names[data_ref.var_names.isin(data_trg.var_names)]

    data_ref = data_ref[:, data_ref.var_names.isin(shared_genes)]
    data_trg = data_trg[:, data_trg.var_names.isin(shared_genes)]
    data_ref = data_ref[:, data_ref.var_names.argsort()].copy()
    data_trg = data_trg[:, data_trg.var_names.argsort()].copy()
    assert all(data_ref.var_names == data_trg.var_names)
    gene_mask = select_gene(data_ref.X, num_gene=num_gene, threshold=0)
    return data_ref, data_trg, gene_mask

class saver():
    def __init__(self, save_path, model_name, writer, args):
        self.save_path = save_path
        self.model_name = model_name
        self.args = args
        self.writer = writer
    
    def save_ckpt(self, model, model_metric, optimizer, epoch):
        output_path = os.path.join(self.save_path, self.model_name)
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        
        save_dict = {'model': model.state_dict(),
                     'model_metric': model_metric.state_dict() if model_metric else None,\
                     'optimizer': optimizer.state_dict()}
        torch.save(save_dict, os.path.join(output_path, "ckpt_{}.pt".format(epoch)))
        print("Save ckpt of model and optmizer")
        
    def save_log(self, log, best_log):
        output = {'args': self.args.__dict__, \
                  'log': log,\
                  'best':best_log}
        output_path = os.path.join(self.save_path, self.model_name, 'log.json')
        with open(output_path, 'w') as handle:
            json.dump(output, handle)
        print("Saved training log and experiment config")
    def save_embedding(self, model, embed_dim, dataloaders, device):
        start_idx = 0
        meta_list = []
        embedding = torch.zeros(len(dataloaders['train'].dataset) + len(dataloaders['test'].dataset), embed_dim)
        print(embedding.shape)
        for phase in ['train', 'test']:
            with torch.no_grad():
                for inputs, labels, batch_id in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = model(inputs)
                    # if model_metric is not None:
                    #     embed = model_metric(outputs, labels)
                    # else:
                    #     embed = outputs
                    
                    end_idx = start_idx + inputs.shape[0]
                    embedding[start_idx:end_idx] = outputs

                    for i in range(len(labels)):
                        meta_list.append("-".join([str(labels[i].item()),str(batch_id[i].item())]))
                    start_idx = end_idx
        assert len(meta_list) == embedding.shape[0]
        torch.save(embedding, os.path.join(self.save_path, 'embed.pt'))
        self.writer.add_embedding(embedding, metadata=meta_list)
        self.writer.close()

def train(dataloaders, model, model_metric, paired, criterion, optimizer, device, saver, num_epoch):
    since = time.time()
    logs = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0
    best_log = {'best_acc':best_acc, 'best_epoch':best_epoch}
    for epoch in range(num_epoch):
        print('Epoch {}/{}'.format(epoch, num_epoch - 1))
        print('-' * 10)
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            elif phase == 'test':
                model.eval()   # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels, batch_id in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    if model_metric is not None:
                        metric_output = model_metric(outputs, labels)
                        _, preds = torch.max(metric_output, 1)
                        loss = criterion(metric_output, labels)
                    else:
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            logs.append({'epoch':epoch,\
                         '{}_loss'.format(phase):epoch_loss,\
                         '{}_acc'.format(phase): epoch_acc.item()})
            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_epoch = epoch
                best_acc = epoch_acc
                best_log['best_epoch'] = epoch
                best_log['best_acc'] = epoch_acc.item()
            if phase == 'train':
                saver.save_ckpt(model, model_metric, optimizer, epoch)
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    saver.save_log(logs, best_log)
    if not paired:
        saver.save_embedding(model, outputs.shape[-1], dataloaders, device)
    return



def create_model(num_input, num_hidden_units, num_class, batch_norm=False, use_metric=False):
    model_tmp = []
    input_unit = num_input

    for hidden_unit in num_hidden_units:
        if not batch_norm:
            model_tmp.append(nn.Linear(int(input_unit), int(hidden_unit), bias=False))
            model_tmp.append(nn.ReLU())
        else:
            model_tmp.append(nn.Linear(int(input_unit), int(hidden_unit), bias=False))
            model_tmp.append(nn.ReLU())
            model_tmp.append(nn.BatchNorm1d(int(hidden_unit)))
        input_unit = hidden_unit

    if use_metric:
        return nn.Sequential(*model_tmp)
    else:
        model_tmp.append(nn.Linear(int(num_hidden_units[-1]), num_class))
        return nn.Sequential(*model_tmp)

