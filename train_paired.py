import random
import time
import copy
import argparse
import torch
import os
import json
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from data import PairedSingleCellData
from loss import ArcFaceLoss
from utils import train, create_model, saver, extract_shared_gene
from tensorboardX import SummaryWriter


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--dataset_train', type=str, required=True)
    parser.add_argument('--dataset_test', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_gene_selected', type=int, default=3000)
    parser.add_argument('--save_cell_map', default=False, action='store_true')
    
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--sgd_momentum', type=float, default=0.9)
    
    parser.add_argument('--use_metric', default=False, action='store_true')
    parser.add_argument('--layers', nargs='+', required=True)
    parser.add_argument('--use_batch_norm', default=False, action='store_true')

    parser.add_argument('--num_epoch', type=int, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    args = parser.parse_args()

    if args.seed is not None:
        print('Using seed:', args.seed)
        torch.manual_seed(args.seed)
        random.seed(args.seed)
    device = torch.device('cuda')

    results_path = os.path.join(args.save_path, "_".join([os.path.split(args.dataset_train)[-1].split(".")[0],\
                                                           os.path.split(args.dataset_test)[-1].split(".")[0]]))
    if not os.path.exists(results_path):
        os.makedirs(results_path, exist_ok=True)
    
    adata_train, adata_test, shared_gene = extract_shared_gene(data_ref=args.dataset_train, data_trg=args.dataset_test, num_gene=args.num_gene_selected)
    sc_dataset_train = PairedSingleCellData(adata=adata_train, gene_mask=shared_gene)
    sc_dataset_test = PairedSingleCellData(adata=adata_test,\
                                           gene_mask=shared_gene,\
                                           cell_map={'id2cell':sc_dataset_train.id_to_cell, 'cell2id':sc_dataset_train.cell_to_id})
    assert(all(sc_dataset_train.anndata.var_names == sc_dataset_test.anndata.var_names))


    if args.save_cell_map:
        with open(os.path.join(results_path, 'id2cell.json'), 'w') as handle:
            json.dump(sc_dataset_train.id_to_cell, handle)
    print(len(sc_dataset_train), len(sc_dataset_test))


    sc_dataloader_train = DataLoader(sc_dataset_train, batch_size=args.batch_size, shuffle=True)
    sc_dataloader_test = DataLoader(sc_dataset_test, batch_size=args.batch_size, shuffle=True)
    dataloader_dict = {'train':sc_dataloader_train, 'test':sc_dataloader_test}
    assert(len(sc_dataloader_test.dataset) != len(sc_dataloader_train.dataset))
    parameters = []
    model = create_model(num_input=args.num_gene_selected,\
                         num_hidden_units=args.layers,\
                         num_class=sc_dataset_train.num_class,\
                         batch_norm=args.use_batch_norm,\
                         use_metric=args.use_metric)
    model.to(device)
    print(model)
    parameters.append({'params':model.parameters()})
    criterion = torch.nn.CrossEntropyLoss()
    
    if args.use_metric:
        model_metric = ArcFaceLoss(in_features=int(args.layers[-1]), out_features=sc_dataset_train.num_class)
        model_metric.to(device)
        parameters.append({'params':model_metric.parameters()})
        model_name = 'MLP_{}_metric'.format("-".join(args.layers))
    else:
        model_metric = None
        model_name = 'MLP_{}'.format("-".join(args.layers))
    
    optimizer = torch.optim.SGD(parameters, lr=args.learning_rate, momentum=args.sgd_momentum)


    embedding_writer = SummaryWriter(results_path)

    saver = saver(save_path=results_path,\
                  model_name=model_name,\
                  writer=embedding_writer,\
                  args=args)
    results = train(dataloaders=dataloader_dict,\
                    model=model,\
                    model_metric=model_metric,\
                    paired=True,
                    criterion=criterion,\
                    optimizer=optimizer,\
                    device=device,\
                    saver=saver,\
                    num_epoch=args.num_epoch)
