from __future__ import print_function, absolute_import, division

import datetime
import os
import os.path as path
import random

import numpy as np
import wandb
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from transformers import get_cosine_schedule_with_warmup

from function_baseline.config import get_parse_args
from function_baseline.data_preparation import data_preparation
from function_baseline.model_pos_preparation import model_pos_preparation
from function_baseline.port import PORT
from function_poseaug.e2e_train import train
from function_poseaug.e2e_eval import evaluate
from utils.log import Logger, savefig
from utils.utils import save_ckpt
N_DATA = 1559752

"""
this code is used to pretrain the baseline model
1. Simple Baseline
2. VideoPose
3. SemGCN
4. ST-GCN
code are modified from https://github.com/garyzhao/SemGCN
"""


def main(args):
    print('==> Using settings {}'.format(args))
    device = torch.device("cuda")

    print('==> Loading dataset...')
    data_dict = data_preparation(args)

    print("==> Creating PoseNet model...")
    model_pos = model_pos_preparation(args, data_dict['dataset'], device).cuda()
    assert path.isfile(args.evaluate), '==> No checkpoint found at {}'.format(args.evaluate)
    print("==> Loading checkpoint '{}'".format(args.evaluate))
    ckpt = torch.load(args.evaluate)
    try:
        model_pos.load_state_dict(ckpt['state_dict'])
    except:
        model_pos.load_state_dict(ckpt['model_pos'])

    port = PORT(args).cuda()
    ckpt_port = torch.load(args.wandb_dir + "/wandb/run-20210801_101612-2bqfzhto/files/ckpt_epoch_0025.pth.tar")
    port.load_state_dict(ckpt_port['state_dict'])

    print("==> Prepare optimizer...")
    criterion = nn.MSELoss(reduction='mean').to(device)
    optimizer = torch.optim.Adam(port.parameters(), lr=args.lr)

    if args.scheduler == "cos_warmup":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=N_DATA//args.batch_size * args.epochs
        )
    elif args.scheduler == "exp_decay":
        scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: args.lr_gamma ** epoch)

    ckpt_dir_path = path.join(args.checkpoint, args.posenet_name, args.keypoints,
                                   datetime.datetime.now().strftime('%m%d%H%M%S') + '_' + args.note)
    os.makedirs(ckpt_dir_path, exist_ok=True)
    print('==> Making checkpoint dir: {}'.format(ckpt_dir_path))


    logger = Logger(os.path.join(ckpt_dir_path, 'log.txt'), args)
    logger.set_names(['epoch', 'lr', 'loss_train', 'error_h36m_p1', 'error_h36m_p2', 'error_3dhp_p1', 'error_3dhp_p2'])
    #################################################
    # ########## start training here
    #################################################
    start_epoch = 0
    error_best = None
    glob_step = 0
    lr_now = args.lr

    for epoch in range(start_epoch, args.epochs):
        print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr_now))

        # Train for one epoch
        p1, p2, lr_now, glob_step = train(data_dict['train_loader'], model_pos, port, criterion, optimizer, scheduler, args.scheduler, device,
                                            args.lr, lr_now, glob_step, args.lr_decay, args.lr_gamma,args.mask_val,
                                            args.mask_p, args.mask_remain_p, args.mask_random_p, args.lambda_ref,
                                            max_norm=args.max_norm)
        wandb.log({"p1_loss": p1, "p2_loss":p2,
                   "lr":lr_now}, step=epoch)
        # Evaluate
        if (epoch+1) % args.eval_period == 0:
            error_p1, error_p2, error_p3, error_p4 = evaluate(data_dict['H36M_test'], model_pos, port, device, args)
            print("pos", "MPJPE : ", error_p1, "P-MPJPE : ", error_p2, "ref","MPJPE : ", error_p3, "P-MPJPE : ", error_p4)
            wandb.log({"pos_MPJPE": error_p1, "pos_P-MPJPE": error_p2,
                       "ref_MPJPE": error_p3, "ref_P-MPJPE": error_p4}, step=epoch)

            if error_best is None or error_best > error_p4:
                error_best = error_p4
                save_ckpt({'state_dict': model_pos.state_dict(), 'epoch': epoch + 1}, args.ckpt_dir, suffix='best')
        # Update log file
        # logger.append([epoch + 1, lr_now, train_loss, error_h36m_p1, error_h36m_p2, error_h36m_p1, error_h36m_p2])

        # Update checkpoint

        if (epoch + 1) % args.snapshot == 0:
            save_ckpt({'state_dict': model_pos.state_dict(), 'epoch': epoch + 1}, args.ckpt_dir)

    logger.close()
    logger.plot(['loss_train', 'error_h36m_p1'])
    savefig(path.join(args.ckpt_dir, 'log.eps'))
    return



if __name__ == '__main__':
    args = get_parse_args()
    # fix random
    random_seed = args.random_seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    # copy from #https://pytorch.org/docs/stable/notes/randomness.html
    torch.backends.cudnn.deterministic = True
    cudnn.benchmark = True
    args.wandb_dir = "/mnt/data3"
    wandb.init(
        project=f"H36M_e2e",
        entity="chibros",
        config=args,
        sync_tensorboard=True,
        dir=args.wandb_dir,
    )
    args.ckpt_dir = wandb.run.dir
    main(args)
