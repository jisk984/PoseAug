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


def pretrain_port(data_loader,
          port,
          criterion,
          optimizer,
          scheduler,
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
          max_norm=True):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    epoch_loss_3d_pos = AverageMeter()

    # Switch to train mode
    torch.set_grad_enabled(True)
    port.train()
    end = time.time()

    ## TODO parallel
    ## DONE wandb
    ## DONE lr scheduler
    ## apex
    ## DONE log dir HDD
    ## DONE warning suppression
    ## DONE MSE validation에 추가

    bar = Bar('Train', max=len(data_loader))
    for i, (targets_3d, _, _, _) in enumerate(data_loader):
        # Measure data loading time
        data_time.update(time.time() - end)
        num_poses = targets_3d.size(0)

        step += 1

        targets_3d = targets_3d.to(device)
        targets_3d = targets_3d[:, :, :] - targets_3d[:, :1, :]  # the output is relative to the 0 joint

        inp_masked, gt, _ = get_masked_input_and_labels(
            targets_3d,
            mask_value=mask_val,
            mask_p=mask_p,
            mask_remain_p=mask_remain_p,
            mask_random_p=mask_random_p
        )
        ## TODO gt, targets_3d 다른지 확인

        outputs_3d = port(inp_masked)

        optimizer.zero_grad()
        loss_3d_pos = criterion(outputs_3d.logits, gt)
        loss_3d_pos.backward()
        if max_norm:
            nn.utils.clip_grad_norm_(port.parameters(), max_norm=1)
        optimizer.step()
        scheduler.step()

        epoch_loss_3d_pos.update(loss_3d_pos.item(), num_poses)

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = '({batch}/{size}) Total: {ttl:} | ETA: {eta:} ' \
                     '| Loss: {loss: .4f}' \
            .format(batch=i + 1, size=len(data_loader), ttl=bar.elapsed_td, eta=bar.eta_td, loss=epoch_loss_3d_pos.avg)
        bar.next()

    bar.finish()
    return epoch_loss_3d_pos.avg, get_lr(optimizer), step
