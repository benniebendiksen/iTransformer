import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from torch.optim.lr_scheduler import ReduceLROnPlateau


plt.switch_backend('agg')

def adjust_learning_rate(optimizer, epoch, args, val_loss=None, scheduler=None):
    scheduler.step(val_loss)
    new_lr = optimizer.param_groups[0]['lr']
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr  # Explicitly update the learning rate in param_groups
    print(f"Updating learning rate to {new_lr} using ReduceLROnPlateau")
    # if args.lradj == 'type1':
    #     lr = args.learning_rate * ((1/3) ** ((epoch - 1) // 1))
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = lr
    #     print('Updating learning rate to {}'.format(lr))
    #
    # elif args.lradj == 'type2':
    #     lr_adjust = {
    #         2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
    #         10: 5e-7, 15: 1e-7, 20: 5e-8
    #     }
    #     if epoch in lr_adjust:
    #         lr = lr_adjust[epoch]
    #         for param_group in optimizer.param_groups:
    #             param_group['lr'] = lr
    #         print('Updating learning rate to {}'.format(lr))
    #
    # elif args.lradj == 'plateau' and scheduler is not None and val_loss is not None:
    #     scheduler.step(val_loss)


# def adjust_learning_rate(optimizer, epoch, args):
#     # lr = args.learning_rate * (0.2 ** (epoch // 2))
#     if args.lradj == 'type1':
#         lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
#         lr_adjust = {epoch: args.learning_rate * ((1/3) ** ((epoch - 1) // 1))}
#     elif args.lradj == 'type2':
#         lr_adjust = {
#             2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
#             10: 5e-7, 15: 1e-7, 20: 5e-8
#         }
#     if epoch in lr_adjust.keys():
#         lr = lr_adjust[epoch]
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = lr
#         print('Updating learning rate to {}'.format(lr))


# class EarlyStopping:
#     def __init__(self, patience=7, verbose=False, delta=0):
#         self.patience = patience
#         self.verbose = verbose
#         self.counter = 0
#         self.best_score = None
#         self.early_stop = False
#         self.val_loss_min = np.Inf
#         self.train_loss_prev = np.Inf  # Track previous training loss
#         self.delta = delta
#
#     def __call__(self, val_loss, model, path, train_loss=None):
#         score = -val_loss
#         train_loss_decreased = False
#
#         # Check if training loss decreased (if provided)
#         if train_loss is not None:
#             train_loss_decreased = train_loss < self.train_loss_prev
#             self.train_loss_prev = train_loss
#
#         if self.best_score is None:
#             # First call, simply save model
#             self.best_score = score
#             self.save_checkpoint(val_loss, model, path)
#         elif score < self.best_score + self.delta:
#             # Validation didn't improve enough
#             if train_loss is not None and train_loss_decreased:
#                 # Training loss is decreasing, so we reset counter and save model
#                 if self.verbose:
#                     print(
#                         f'Validation loss did not improve, but train loss decreased ({self.train_loss_prev:.6f} --> {train_loss:.6f}). Resetting counter and saving model...')
#                 self.save_checkpoint(val_loss, model, path)
#                 self.counter = 0
#             else:
#                 # Neither validation nor training loss improved, increment counter
#                 self.counter += 1
#                 print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
#                 if self.counter >= self.patience:
#                     self.early_stop = True
#         else:
#             # Validation improved, save model and reset counter
#             self.best_score = score
#             self.save_checkpoint(val_loss, model, path)
#             self.counter = 0
#
#     def save_checkpoint(self, val_loss, model, path):
#         if self.verbose:
#             print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
#         torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
#         self.val_loss_min = val_loss

class EarlyStopping:
    def __init__(self, patience=1, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)
