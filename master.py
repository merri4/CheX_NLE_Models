# 이건 학습을 하는 게 아니라 모델을 불러오고 값을 통과시키기만 하면 되는거임.



import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm

# assert timm.__version__ == "0.3.2" # version check
from timm.models.layers import trunc_normal_
# from timm.data.mixup import Mixup
# from util.mixup_multi_label import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from util.multi_label_loss import SoftTargetBinaryCrossEntropy

import util.lr_decay as lrd
import util.misc as misc
from util.datasets import build_dataset, build_dataset_chest_xray
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import torch.nn.functional as F
import models_vit

from engine_finetune import train_one_epoch, evaluate_chestxray
from util.sampler import RASampler
# from apex.optimizers import FusedAdam
from libauc import losses
from torchvision import models
import timm.optim.optim_factory as optim_factory
from collections import OrderedDict


# 파싱

def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=14, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument("--train_list", default=None, type=str, help="file for train list")
    parser.add_argument("--val_list", default=None, type=str, help="file for val list")
    parser.add_argument("--test_list", default=None, type=str, help="file for test list")
    parser.add_argument('--eval_interval', default=10, type=int)
    parser.add_argument('--fixed_lr', action='store_true', default=False)
    parser.add_argument('--vit_dropout_rate', type=float, default=0,
                        help='Dropout rate for ViT blocks (default: 0.0)')
    parser.add_argument("--build_timm_transform", action='store_true', default=False)
    parser.add_argument("--aug_strategy", default='default', type=str, help="strategy for data augmentation")
    parser.add_argument("--dataset", default='MIMIC', type=str)

    parser.add_argument('--repeated-aug', action='store_true', default=False)

    parser.add_argument("--optimizer", default='adamw', type=str)

    parser.add_argument('--ThreeAugment', action='store_true')  # 3augment

    parser.add_argument('--src', action='store_true')  # simple random crop

    parser.add_argument('--loss_func', default=None, type=str)

    parser.add_argument("--norm_stats", default=None, type=str)

    parser.add_argument("--checkpoint_type", default=None, type=str)

    return parser



def main(args) :
    
    # 분산처리 할거면 하시고
    misc.init_distributed_mode(args)

    # job dir 및 파라미터 출력
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # 시드 고정
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ?
    cudnn.benchmark = True


    # 데이터셋 준비하기
    dataset_train = build_dataset_chest_xray(split='train', args=args)
    dataset_val = build_dataset_chest_xray(split='val', args=args)
    dataset_test = build_dataset_chest_xray(split='test', args=args)
    
    
    # sampler_train = torch.utils.data.RandomSampler(dataset_train)
    # print("Sampler_train = %s" % str(sampler_train))
    # sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    # print("Sampler_train = %s" % str(sampler_val))
    # sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    # print("Sampler_train = %s" % str(sampler_test))




    # dataloader에다가 train 담음
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        # sampler = sampler_train,
        batch_size = args.batch_size,
        num_workers = args.num_workers,
        pin_memory = args.pin_mem,
        drop_last = True,
    )


    # dataloader에다가 test 담음
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        # sampler = sampler_test,
        batch_size = args.batch_size,
        num_workers = args.num_workers,
        pin_memory = args.pin_mem,
        drop_last = False,
    )

    


    #### 모델 불러오기


    # 모델의 구조 생성
    if 'vit' in args.model:
        model = models_vit.__dict__[args.model](
            img_size=args.input_size, # 224
            num_classes=args.nb_classes, # 14
            drop_rate=args.vit_dropout_rate, # 0
            drop_path_rate=args.drop_path, # 0.1
            global_pool=args.global_pool, # True
        )

    elif 'densenet' in args.model or 'resnet' in args.model:
        model = models.__dict__[args.model](num_classes=args.nb_classes)
    
    else :
        raise NotImplementedError



    # pretrain weights load
    
    if 'vit' in args.model : # ViT면

        checkpoint = torch.load(args.finetune, map_location='cpu')
        print("Load pre-trained checkpoint from: %s" % args.finetune)

        checkpoint_model = checkpoint['model'] # model 부분만 뜯어서

        # 형태가 맞지 않으면 일부 shape 제거
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias'] :
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        
        # 기본적으로 실행?
        if args.global_pool:
            for k in ['fc_norm.weight', 'fc_norm.bias']: # fully-connected layer
                try:
                    del checkpoint_model[k]
                except:
                    pass

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        # manually initialize fc layer
        # normalize by truncation, fully connected layer
        trunc_normal_(model.head.weight, std=2e-5)
    
    # DenseNet이나 ResNet이면
    elif ('densenet' in args.model) or ('resnet' in args.model) :

        checkpoint = torch.load(args.finetune, map_location='cpu')
        print("Load pre-trained checkpoint from: %s" % args.finetune)
        if 'state_dict' in checkpoint.keys():
            checkpoint_model = checkpoint['state_dict']
        elif 'model' in checkpoint.keys():
            checkpoint_model = checkpoint['model']
        else:
            checkpoint_model = checkpoint
        if args.checkpoint_type == 'smp_encoder':
            state_dict = checkpoint_model

            new_state_dict = OrderedDict()

            for key, value in state_dict.items():
                if 'model.encoder.' in key:
                    new_key = key.replace('model.encoder.', '')
                    new_state_dict[new_key] = value
            checkpoint_model = new_state_dict

        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)


    # model_without_ddp : distributed data parallel 없는 통째로 모델

    if 'vit' in args.model:
        param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
            no_weight_decay_list=model_without_ddp.no_weight_decay(),
            layer_decay=args.layer_decay
        )
    else:
        param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)


    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()
    # criterion = torch.nn.BCEWithLogitsLoss() # 학습할 때만 필요


    model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))


    # 모델을 불러온다.
    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)


    # 평가만 할거면
    if args.eval:
        test_stats = evaluate_chestxray(data_loader_test, model, device, args)
        print(f"Average AUC of the network on the test set images: {test_stats['auc_avg']:.4f}")
        exit(0)


    with torch.no_grad() :
        


# 실행
if __name__ == '__main__' :

    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)








