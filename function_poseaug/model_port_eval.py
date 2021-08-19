from __future__ import print_function, absolute_import, division

import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from common.data_loader import PoseDataSet
from progress.bar import Bar
from utils.data_utils import fetch
from utils.loss import mpjpe, p_mpjpe, compute_PCK, compute_AUC
from utils.utils import AverageMeter, get_masked_input_and_labels


####################################################################
# ### evaluate p1 p2 pck auc dataset with test-flip-augmentation
####################################################################
def evaluate(data_loader,
             model_pos_eval,
             criterion,
             device,
             mask_val,
             mask_p,
             mask_remain_p,
             mask_random_p,
             summary=None, writer=None, key='', tag='', flipaug=''):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    mpjpe_gt = AverageMeter()
    pmpjpe_gt = AverageMeter()
    mse_gt = AverageMeter()
    mpjpe_gcn = AverageMeter()
    pmpjpe_gcn = AverageMeter()
    mse_gcn = AverageMeter()
    mpjpe_mlp = AverageMeter()
    pmpjpe_mlp = AverageMeter()
    mse_mlp = AverageMeter()
    mpjpe_stgcn = AverageMeter()
    pmpjpe_stgcn = AverageMeter()
    mse_stgcn = AverageMeter()
    mpjpe_videopose = AverageMeter()
    pmpjpe_videopose = AverageMeter()
    mse_videopose = AverageMeter()
    epoch_auc = AverageMeter()
    epoch_pck = AverageMeter()

    # Switch to evaluate mode
    model_pos_eval.eval()
    end = time.time()
    input_str = ["gt", "gcn", "mlp", "stgcn", "videopose"]
    metric_str = ["MPJPE", "P-MPJPE", "MSE"]

    bar = Bar('Eval posenet on {}'.format(key), max=len(data_loader), dynamic_ncols=True)
    for i, temp in enumerate(data_loader):
        targets_3d, inputs_2d, gcn, mlp, stgcn, videopose\
            = temp[0], temp[1], temp[2], temp[3], temp[4], temp[5]

        # Measure data loading time
        data_time.update(time.time() - end)
        num_poses = targets_3d.size(0)
        inputs_2d = inputs_2d.to(device)
        targets_3d = targets_3d.to(device)
        gcn = gcn.squeeze().to(device)
        mlp = mlp.squeeze().to(device)
        stgcn = stgcn.squeeze().to(device)
        videopose = videopose.squeeze().to(device)

        targets_3d = targets_3d[:, :, :] - targets_3d[:, :1, :]  # the output is relative to the 0 joint

        inp_masked, gt, _ = get_masked_input_and_labels(
            targets_3d,
            mask_value=mask_val,
            mask_p=mask_p,
            mask_remain_p=mask_remain_p,
            mask_random_p=mask_random_p
        )

        with torch.no_grad():
            if flipaug:  # flip the 2D pose Left <-> Right
                joints_left = [4, 5, 6, 10, 11, 12]
                joints_right = [1, 2, 3, 13, 14, 15]
                out_left = [4, 5, 6, 10, 11, 12]
                out_right = [1, 2, 3, 13, 14, 15]

                # inputs_2d_flip = inputs_2d.detach().clone()
                # inputs_2d_flip[:, :, 0] *= -1
                # inputs_2d_flip[:, joints_left + joints_right, :] = inputs_2d_flip[:, joints_right + joints_left, :]
                outputs_3d_flip = model_pos_eval(targets_3d.cuda()).logits.view(num_poses, -1, 3).cpu()
                outputs_3d_flip[:, :, 0] *= -1
                outputs_3d_flip[:, out_left + out_right, :] = outputs_3d_flip[:, out_right + out_left, :]

                outputs_3d = model_pos_eval(inputs_3d).logits.cpu()
                outputs_3d = (outputs_3d + outputs_3d_flip) / 2.0
            else:
                outputs_gt = model_pos_eval(gt).logits.cpu()
                outputs_gcn = model_pos_eval(gcn).logits.cpu()
                outputs_mlp = model_pos_eval(mlp).logits.cpu()
                outputs_stgcn = model_pos_eval(stgcn).logits.cpu()
                outputs_videopose = model_pos_eval(videopose).logits.cpu()
        targets_3d = targets_3d.cpu()
        # caculate the relative position.
        # targets_3d = targets_3d[:, :, :] - targets_3d[:, :1, :]  # the output is relative to the 0 joint
        # outputs_3d = outputs_3d[:, :, :] - outputs_3d[:, :1, :]  # the output is relative to the 0 joint

        # compute p1 and p2
        def eval_metric_(epoch_p1, epoch_p2, epoch_p3, outputs_3d, targets_3d):
            p1score = mpjpe(outputs_3d, targets_3d).item() * 1000.0
            epoch_p1.update(p1score, num_poses)
            p2score = p_mpjpe(outputs_3d.numpy(), targets_3d.numpy()).item() * 1000.0
            epoch_p2.update(p2score, num_poses)
            mse_loss = criterion(targets_3d, outputs_3d)
            epoch_p3.update(mse_loss, num_poses)

        eval_metric_(mpjpe_gt, pmpjpe_gt, mse_gt, outputs_gt, targets_3d)
        eval_metric_(mpjpe_gcn, pmpjpe_gcn, mse_gcn, outputs_gcn, targets_3d)
        eval_metric_(mpjpe_mlp, pmpjpe_mlp, mse_mlp, outputs_mlp, targets_3d)
        eval_metric_(mpjpe_stgcn, pmpjpe_stgcn, mse_stgcn, outputs_stgcn, targets_3d)
        eval_metric_(mpjpe_videopose, pmpjpe_videopose, mse_videopose, outputs_videopose, targets_3d)

        # compute AUC and PCK
        # pck = compute_PCK(targets_3d.numpy(), outputs_3d.numpy())
        # epoch_pck.update(pck, num_poses)
        # auc = compute_AUC(targets_3d.numpy(), outputs_3d.numpy())
        # epoch_auc.update(auc, num_poses)

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # bar.suffix = '({batch}/{size}) | Total: {ttl:} | ETA: {eta:} ' \
            # '| MPJPE: {e1: .4f} | P-MPJPE: {e2: .4f}, Loss: {mse: .4f}' \
            # .format(batch=i + 1, size=len(data_loader),
                    # ttl=bar.elapsed_td, eta=bar.eta_td, e1=epoch_p1.avg,
                    # e2=epoch_p2.avg, mse=mse_loss.cpu().item())
        bar.next()

    bar.finish()

    gt=[mpjpe_gt, pmpjpe_gt, mse_gt]
    gcn=[mpjpe_gcn, pmpjpe_gcn, mse_gcn]
    mlp=[mpjpe_mlp, pmpjpe_mlp, mse_mlp]
    stgcn=[mpjpe_stgcn, pmpjpe_stgcn, mse_stgcn]
    videopose=[mpjpe_videopose, pmpjpe_videopose, mse_videopose]
    metric_lst = [gt, gcn, mlp, stgcn, videopose]
    metrics = {}
    for i, metric in enumerate(metric_lst):
        metrics.update({input_str[i] + "/" + metric_str[j]:val.avg for j, val in enumerate(metric)})
    return metrics


#########################################
# overall evaluation function
#########################################
def evaluate_posenet(args, data_dict, model_pos, model_pos_eval, device, summary, writer, tag):
    """
    evaluate H36M and 3DHP
    test-augment-flip only used for 3DHP as it does not help on H36M.
    """
    with torch.no_grad():
        model_pos_eval.load_state_dict(model_pos.state_dict())
        h36m_p1, h36m_p2 = evaluate(data_dict['H36M_test'], model_pos_eval, device, summary, writer,
                                             key='H36M_test', tag=tag, flipaug='')  # no flip aug for h36m
        dhp_p1, dhp_p2 = evaluate(data_dict['mpi3d_loader'], model_pos_eval, device, summary, writer,
                                           key='mpi3d_loader', tag=tag, flipaug='_flip')
    return h36m_p1, h36m_p2, dhp_p1, dhp_p2

