from __future__ import print_function, absolute_import, division

import time

import torch
import torch.nn as nn

from progress.bar import Bar
from utils.utils import AverageMeter, lr_decay, get_masked_input_and_labels, get_lr

'''
Code are modified from https://github.com/garyzhao/SemGCN
This train function is adopted from SemGCN for baseline training.
'''


def train(data_loader,
          model_pos,
          port,
          criterion,
          optimizer,
          scheduler,
          scheduler_name,
          device,
          lr_init,
          lr_now,
          step,
          decay,
          gamma,
          mask_val,
          mask_p,
          mask_remain_p,
          mask_random_p,
          lambda_ref,
          max_norm=True):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    epoch_loss_pos = AverageMeter()
    epoch_loss_ref = AverageMeter()

    # Switch to train mode
    torch.set_grad_enabled(True)
    port.train()
    model_pos.train()
    end = time.time()

    ## TODO parallel
    ## DONE wandb
    ## DONE lr scheduler
    ## apex
    ## DONE log dir HDD
    ## DONE warning suppression
    ## DONE MSE validation에 추가

    # bar = Bar('Train', max=len(data_loader))
    for i, (targets_3d, inputs_2d, _, _) in enumerate(data_loader):
        # Measure data loading time
        data_time.update(time.time() - end)
        num_poses = targets_3d.size(0)

        step += 1

        inputs_2d = inputs_2d.to(device)
        targets_3d = targets_3d.to(device)
        targets_3d = targets_3d[:, :, :] - targets_3d[:, :1, :]  # the output is relative to the 0 joint

        ## TODO gt, targets_3d 다른지 확인
        pos_output = model_pos(inputs_2d.view(num_poses, -1))
        pos_output = pos_output[:, :, :] - pos_output[:, :1, :]  # the output is relative to the 0 joint
        ref_output = port(pos_output).logits.view(num_poses, -1, 3)

        optimizer.zero_grad()
        pos_loss = criterion(pos_output, targets_3d)
        ref_loss = criterion(ref_output, targets_3d)
        (pos_loss + lambda_ref * ref_loss).mean().backward()
        if max_norm:
            nn.utils.clip_grad_norm_([model_pos.parameters(), port.parameters()], max_norm=1)
        optimizer.step()
        if scheduler_name == "cos_warmup":
            scheduler.step()

        epoch_loss_pos.update(pos_loss.item(), num_poses)
        epoch_loss_ref.update(ref_loss.item(), num_poses)

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # bar.suffix = '({batch}/{size}) Total: {ttl:} | ETA: {eta:} ' \
                     # '| Loss: {loss: .4f}' \
            # .format(batch=i + 1, size=len(data_loader), ttl=bar.elapsed_td, eta=bar.eta_td, loss=epoch_loss_3d_pos.avg)
        # bar.next()

    # bar.finish()
    if scheduler_name != "cos_warmup":
        scheduler.step()
    return epoch_loss_pos.avg, epoch_loss_ref.avg, get_lr(optimizer), step
