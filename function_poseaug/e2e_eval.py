from __future__ import print_function, absolute_import, division

import time
import pickle
import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader

from common.data_loader import PoseDataSet
from progress.bar import Bar
from utils.data_utils import fetch
from utils.loss import mpjpe, p_mpjpe, compute_PCK, compute_AUC
from utils.utils import AverageMeter
from function_poseaug.poseaug_viz import plot_joints
from matplotlib import pyplot as plt

import seaborn
def draw(data, ax):
    seaborn.heatmap(data, xticklabels=range(1,15), square=True, yticklabels=range(1,15), vmin=0.0, vmax=1.0,
                    cbar=False, ax=ax)

####################################################################
# ### evaluate p1 p2 pck auc dataset with test-flip-augmentation
####################################################################
def evaluate(data_loader, model_pos_eval, model_port_eval, device, args=None, summary=None, writer=None, key='', tag='', flipaug=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    epoch_p1 = AverageMeter()
    epoch_p2 = AverageMeter()
    epoch_p3 = AverageMeter()
    epoch_p4 = AverageMeter()
    epoch_auc = AverageMeter()
    epoch_pck = AverageMeter()

    # Switch to evaluate mode
    model_pos_eval.eval()
    model_port_eval.eval()
    end = time.time()

    bar = Bar('Eval posenet on {}'.format(key), max=len(data_loader))
    output_lst =[]
    for i, temp in enumerate(data_loader):
        targets_3d, inputs_2d, cam_param = temp[0], temp[1], temp[3]
        # Measure data loading time
        data_time.update(time.time() - end)
        num_poses = targets_3d.size(0)
        inputs_2d = inputs_2d.to(device)

        with torch.no_grad():
            if flipaug:  # flip the 2D pose Left <-> Right
                joints_left = [4, 5, 6, 10, 11, 12]
                joints_right = [1, 2, 3, 13, 14, 15]
                out_left = [4, 5, 6, 10, 11, 12]
                out_right = [1, 2, 3, 13, 14, 15]

                inputs_2d_flip = inputs_2d.detach().clone()
                inputs_2d_flip[:, :, 0] *= -1
                inputs_2d_flip[:, joints_left + joints_right, :] = inputs_2d_flip[:, joints_right + joints_left, :]
                outputs_3d_flip = model_pos_eval(inputs_2d_flip.view(num_poses, -1)).view(num_poses, -1, 3).cpu()
                outputs_3d_flip[:, :, 0] *= -1
                outputs_3d_flip[:, out_left + out_right, :] = outputs_3d_flip[:, out_right + out_left, :]

                outputs_3d = model_pos_eval(inputs_2d.view(num_poses, -1)).view(num_poses, -1, 3).cpu()
                outputs_3d = (outputs_3d + outputs_3d_flip) / 2.0

            else:
                pos_output = model_pos_eval(inputs_2d.view(num_poses, -1))
                pos_output_localized = pos_output[:, :, :] - pos_output[:, :1, :]  # the output is relative to the 0 joint
                ref_output = model_port_eval(pos_output_localized)
        # caculate the relative position.
        targets_3d = targets_3d[:, :, :] - targets_3d[:, :1, :]  # the output is relative to the 0 joint

        # compute p1 and p2
        p1score = mpjpe(pos_output_localized.cpu(), targets_3d.cpu()).item() * 1000.0
        epoch_p1.update(p1score, num_poses)
        p2score = p_mpjpe(pos_output_localized.cpu().numpy(), targets_3d.cpu().numpy()).item() * 1000.0
        epoch_p2.update(p2score, num_poses)

        p3score = mpjpe(ref_output.logits.view(num_poses, -1, 3).cpu(), targets_3d.cpu()).item() * 1000.0
        epoch_p3.update(p3score, num_poses)
        p4score = p_mpjpe(ref_output.logits.view(num_poses, -1, 3).cpu().numpy(), targets_3d.cpu().numpy()).item() * 1000.0
        epoch_p4.update(p4score, num_poses)

        # compute AUC and PCK
        # pck = compute_PCK(targets_3d.numpy(), outputs_3d.numpy())
        # epoch_pck.update(pck, num_poses)
        # auc = compute_AUC(targets_3d.numpy(), outputs_3d.numpy())
        # epoch_auc.update(auc, num_poses)

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot_joints(targets_3d.cpu(), inputs_2d.cpu(), pos_output.cpu(), ref_output.logits.view(num_poses, -1, 3).cpu() + pos_output[:,:1,:].cpu(), cam_param, i, args)

    bar.finish()
    return epoch_p1.avg, epoch_p2.avg, epoch_p3.avg, epoch_p4.avg


def save_fig(ref_output):
    fig, axs = plt.subplots(4,4, figsize=(10, 10))
    fig.suptitle(f"attn", fontsize=25)
    for i in range(0,4,1):
        for h in range(4):
            draw(ref_output.attentions[i][0, h].cpu(), axs[i,h%4])
            axs[h//4,h%4].set_title(f"head_{h+1}",fontsize=10)
            # plt.savefig(f"attn/Layer_{i}.png")
    plt.show()
    plt.savefig(f"attn/Layer.png")

def save_output(output_lst, outputs_3d):
    output_split = [val.detach().clone().cpu() for val in outputs_3d.split(1)]
    output_lst.extend(output_split)
