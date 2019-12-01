import time
import os
import torch
import torch.nn as nn
import numpy as np
import copy
import json

class saver():
    def __init__(self, save_path, model_name, args):
        self.save_path = save_path
        self.model_name = model_name
        self.args = args
    
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
        output_path = os.path.join(self.save_path, self.model_name, 'log.txt')
        with open(output_path, 'w') as handle:
            json.dump(output, handle)
        print("Saved training log and experiment config")
    def save_embedding(self, model, dataloaders, device):
        pass

def train(dataloaders, model, model_metric, criterion, optimizer, device, saver, num_epoch):
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
    #saver_save_embedding(model, dataloaders, device)
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

