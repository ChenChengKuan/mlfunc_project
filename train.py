import random
import time
import copy
import argparse
import torch
import os
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from data import SingleCellData
from loss import ArcFaceLoss
from utils import train, create_model



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--dataset_train', type=str, required=True)
    parser.add_argument('--dataset_test', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_gene_selected', type=int, default=3000)

    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--sgd_momentum', type=float, default=0.9)
    
    parser.add_argument('--use_metric', default=False, action='store_true')
    parser.add_argument('--layers', nargs='+', required=True)
    parser.add_argument('--use_batch_norm', default=False, action='store_true')

    parser.add_argument('--num_epoch', type=int)
    parser.add_argument('--save_path', type=str, required=True)
    args = parser.parse_args()

    if args.seed is not None:
        print('Using seed:', args.seed)
        torch.manual_seed(args.seed)
        random.seed(args.seed)
    device = torch.device('cuda')

    sc_dataset_train = SingleCellData(data_path=args.dataset_train, num_gene=args.num_gene_selected)

    sc_dataset_test = SingleCellData(data_path=args.dataset_test,\
                                     num_gene=args.num_gene_selected,\
                                     shared_gene_mask=sc_dataset_train.gene_mask,\
                                     filter_mask = sc_dataset_train.filter_mask)
    assert(np.array_equal(sc_dataset_train.gene_mask, sc_dataset_test.gene_mask))
    assert(np.array_equal(sc_dataset_train.filter_mask, sc_dataset_test.filter_mask))

    sc_dataloader_train = DataLoader(sc_dataset_train, batch_size=args.batch_size)
    sc_dataloader_test = DataLoader(sc_dataset_test, batch_size=args.batch_size)
    dataloader_dict = {'train':sc_dataloader_train, 'test':sc_dataloader_test}
    model = create_model(num_input=args.num_gene_selected,\
                         num_hidden_units=args.layers,\
                         num_class=sc_dataset_train.num_class,\
                         batch_norm=args.use_batch_norm,\
                         use_metric=args.use_metric)
    model.to(device)
    print(model)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.sgd_momentum)
    if args.use_metric:
        metric = ArcFaceLoss(in_features=int(args.layers[-1]), out_features=sc_dataset_train.num_class)
        metric.to(device)
    else:
        metric = None
    results = train(dataloaders=dataloader_dict,\
                    model=model,\
                    metric=metric,\
                    criterion=criterion,\
                    optim=optimizer,\
                    device=device,\
                    save_path=args.save_path,\
                    num_epoch=args.num_epoch)