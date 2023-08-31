# -*- coding: utf-8 -*-
'''

Train CIFAR10 with PyTorch and Vision Transformers!
written by @kentaroy47, @arutema47

'''

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

import torchvision
import torchvision.transforms as transforms

import os
import argparse
# import pandas as pd
import csv
import time
import logging

# from models import *
from utils.progress_bar import progress_bar
# from randomaug import RandAugment
# from models.vit import ViT
# from models.convmixer import ConvMixer

import timm
from lora_my import LoRA_ViT_timm, LoRA_ViT_timm_FC, Adaptive_LoRA_ViT_timm_FC, Adaptive_LoRA_ViT_timm_mixed

# parsers
parser = argparse.ArgumentParser(description='ViT Training')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset name')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate') # resnets.. 1e-3, Vit..1e-4
parser.add_argument('--opt', default="adam")
# parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--noaug', action='store_true', help='disable use randomaug')
parser.add_argument('--noamp', action='store_true', help='disable mixed precision training. for older pytorch versions')
parser.add_argument('--wandb', default=False, action='store_true', help='disable wandb')
parser.add_argument('--mixup', action='store_true', help='add mixup augumentations')
parser.add_argument('--net', default='vit_timm')
parser.add_argument('--bs', default='512')
parser.add_argument('--size', default="32")
parser.add_argument('--n_epochs', type=int, default='200')
parser.add_argument('--patch', default='4', type=int, help="patch for ViT")
parser.add_argument('--dimhead', default="512", type=int)
parser.add_argument('--convkernel', default='8', type=int, help="parameter for convmixer")
parser.add_argument('--log-dir', default='log_new', type=str, help='log save path')
parser.add_argument('--exp-name', default=None, type=str, help='experiment name')
# parser.add_argument('--save-dir', default='checkpoint_new', type=str, help='checkpoint save path')
parser.add_argument('--resume-path', default=None, type=str, help='checkpoint resume path')
parser.add_argument('--save-freq', default=50, type=int, help='checkpoint save frequency')
parser.add_argument('--LoRA', default=False, action='store_true', help='use LoRA')
parser.add_argument('--LoRA-FC', default=False, action='store_true', help='use LoRA on FC layers')
parser.add_argument('-r', default=4, type=int, help='LoRA rank')
parser.add_argument('--Adaptive-LoRA-FC', default=False, action='store_true', help='use Adaptive LoRA')
# parser.add_argument('--lr-scheduler', default='cosine', type=str, help='lr scheduler')
parser.add_argument('--Adaptive-LoRA-mixed', default=False, action='store_true', help='use Adaptive LoRA mixed')

# small orthinit
parser.add_argument('--init-scale', default=1, type=float, help='initialization scale')

args = parser.parse_args()

# take in args
usewandb = args.wandb
if usewandb:
    import wandb
    watermark = "{}_lr{}".format(args.net, args.lr)
    wandb.init(project="cifar10-challange",
            name=watermark)
    wandb.config.update(args)

bs = int(args.bs)
imsize = int(args.size)

use_amp = not args.noamp
aug = args.noaug

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


# create dir
print('==> Creating directory..')
if not os.path.exists(args.log_dir):
    os.mkdir(args.log_dir)
# if args.exp_name is None:
#     args.exp_name = time.strftime("%Y-%-m-%d %H:%M") + '_' + f'{args.net}_lr{args.lr}_bs{args.bs}'
# else: 
task_dir = os.path.join(args.log_dir, args.dataset)
if not os.path.exists(task_dir):
    os.mkdir(task_dir)
args.exp_name = time.strftime("%Y-%-m-%d %H:%M") + '_' + args.exp_name + f'_{args.net}_lr{args.lr}_bs{args.bs}'
exp_dir = os.path.join(task_dir, args.exp_name)
if not os.path.exists(exp_dir):
    os.mkdir(exp_dir)
# print(f'{exp_dir}/{args.net}-{args.patch}-ckpt-best.pth')
# exit()


# Data
print('==> Preparing data..')
if args.dataset in ["cifar10", "cifar100"]:
    if args.net=="vit_timm":
        size = 384
    else:
        size = imsize
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.Resize(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # # Add RandAugment with N, M(hyperparameter)
    # if aug:  
    #     N = 2; M = 14;
    #     transform_train.transforms.insert(0, RandAugment(N, M))

    # Prepare dataset
    if args.dataset == "cifar10":
        num_classes = 10
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    elif args.dataset == "cifar100":
        num_classes = 100
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    elif args.dataset == 'fer2013':
        num_classes = 7
        trainset = torchvision.datasets.FER2013(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.FER2013(root='./data', train=False, download=True, transform=transform_test)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=8)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)

    # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model factory..
print('==> Building model..')
if args.net=="vit_small":
    from models.vit_small import ViT
    net = ViT(
    image_size = size,
    patch_size = args.patch,
    num_classes = 10,
    dim = int(args.dimhead),
    depth = 6,
    heads = 8,
    mlp_dim = 512,
    dropout = 0.1,
    emb_dropout = 0.1
)
elif args.net=="vit_tiny":
    from models.vit_small import ViT
    net = ViT(
    image_size = size,
    patch_size = args.patch,
    num_classes = 10,
    dim = int(args.dimhead),
    depth = 4,
    heads = 6,
    mlp_dim = 256,
    dropout = 0.1,
    emb_dropout = 0.1
)
elif args.net=="simplevit":
    from models.simplevit import SimpleViT
    net = SimpleViT(
    image_size = size,
    patch_size = args.patch,
    num_classes = 10,
    dim = int(args.dimhead),
    depth = 6,
    heads = 8,
    mlp_dim = 512
)
elif args.net=="vit":
    # ViT for cifar10
    net = ViT(
    image_size = size,
    patch_size = args.patch,
    num_classes = 10,
    dim = int(args.dimhead),
    depth = 6,
    heads = 8,
    mlp_dim = 512,
    dropout = 0.1,
    emb_dropout = 0.1
)
elif args.net=="vit_timm":
    import timm
    net = timm.create_model("vit_base_patch16_384", pretrained=True)
    num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Original trainable parameters: {num_params}")

    if args.LoRA:
        net = LoRA_ViT_timm(vit_model=net, r=args.r, num_classes=num_classes)
        # args.n_epochs = 20
        num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(f"LoRA trainable parameters: {num_params}")
    elif args.LoRA_FC:
        net = LoRA_ViT_timm_FC(vit_model=net, r=args.r, num_classes=num_classes)
        num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(f"LoRA trainable parameters: {num_params}")
    elif args.Adaptive_LoRA_FC:
        ###### adaptive rank allocation
        if args.dataset == "cifar10":
            sr1 = np.array([7.6676348, 8.7078726, 13.94798877, 13.36670013, 19.56721734, 28.8925499, 
                   32.65807577, 30.33634341, 34.60073313, 23.48366018, 16.13951141, 9.22370525])
            sr2 = np.array([9.46316601, 17.58167134, 17.9991531, 20.33843591, 23.18977949, 27.35260371, 
                   21.32254756, 26.86263539, 32.05404861, 16.65734139, 6.55440992, 4.76064809])
        elif args.dataset == "cifar100":
            sr1 = np.array([8.42423471, 9.07851322, 13.43602677, 13.25423756, 21.78378579, 31.63520525, 
                35.71182006, 38.31577465, 46.78656648, 30.36281206, 20.72470383, 18.17922948])
            sr2 = np.array([11.84737225, 22.60192817, 19.99366874, 26.24931243, 30.16093538, 35.84856282, 
                33.64967826, 37.71307321, 55.02957429, 27.47488328, 16.9205525 , 24.94070038])

        ###### first 9
        # sr1 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0])
        # sr2 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0])

        ###### first 9 adaptive
        # if args.dataset == "cifar10":
        #     sr1 = np.array([7.6676348, 8.7078726, 13.94798877, 13.36670013, 19.56721734, 28.8925499, 
        #            32.65807577, 30.33634341, 34.60073313, 0, 0, 0])
        #     sr2 = np.array([9.46316601, 17.58167134, 17.9991531, 20.33843591, 23.18977949, 27.35260371, 
        #            21.32254756, 26.86263539, 32.05404861, 0, 0, 0])
        # elif args.dataset == "cifar100":
        #     sr1 = np.array([8.42423471, 9.07851322, 13.43602677, 13.25423756, 21.78378579, 31.63520525, 
        #         35.71182006, 38.31577465, 46.78656648, 0, 0, 0])
        #     sr2 = np.array([11.84737225, 22.60192817, 19.99366874, 26.24931243, 30.16093538, 35.84856282, 
        #         33.64967826, 37.71307321, 55.02957429, 0, 0, 0])
            

        ###### last 3
        # sr1 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
        # sr2 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1 ,1])

        ###### first 9 adaptive inverse
        # if args.dataset == "cifar10":
        #     sr1 = np.array([7.6676348, 8.7078726, 13.94798877, 13.36670013, 19.56721734, 28.8925499, 
        #            32.65807577, 30.33634341, 34.60073313, 0, 0, 0])
        #     sr2 = np.array([9.46316601, 17.58167134, 17.9991531, 20.33843591, 23.18977949, 27.35260371, 
        #            21.32254756, 26.86263539, 32.05404861, 0, 0, 0])
        # elif args.dataset == "cifar100":
        #     sr1 = np.array([8.42423471, 9.07851322, 13.43602677, 13.25423756, 21.78378579, 31.63520525, 
        #         35.71182006, 38.31577465, 46.78656648, 0, 0, 0])
        #     sr2 = np.array([11.84737225, 22.60192817, 19.99366874, 26.24931243, 30.16093538, 35.84856282, 
        #         33.64967826, 37.71307321, 55.02957429, 0, 0, 0])
        # sr1 = np.concatenate([1/sr1[:-3], np.zeros(3)])
        # sr2 = np.concatenate([1/sr2[:-3], np.zeros(3)])

        ####### last 9
        # sr1 = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        # sr2 = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1 ,1, 1])

        r1 = (12* args.r * sr1)/np.sum(sr1)
        r1 = r1.round().astype(int) # r1 = r1.astype(int)
        r2 = 12* args.r * sr2 / np.sum(sr2)
        r2 = r2.round().astype(int) # r2 = r2.astype(int)
        print('fc1 rank allocation:', r1)
        print('fc2 rank allocation:', r2)
        rs = np.stack([r1, r2], axis=1)
        net = Adaptive_LoRA_ViT_timm_FC(vit_model=net, r=rs, num_classes=num_classes)

        num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(f"LoRA trainable parameters: {num_params}")
    elif args.Adaptive_LoRA_mixed:
        ####### mixed
        sr1 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        sr2 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        r1 = (12* args.r * sr1)/np.sum(sr1)
        r1 = r1.round().astype(int) # r1 = r1.astype(int)
        r2 = 12* args.r * sr2 / np.sum(sr2)
        r2 = r2.round().astype(int) # r2 = r2.astype(int)
        print('fc1 rank allocation:', r1)
        print('fc2 rank allocation:', r2)
        rs = np.stack([r1, r2], axis=1)
        net = Adaptive_LoRA_ViT_timm_mixed(vit_model=net, r=rs, num_classes=num_classes)
    else:
        net.head = nn.Linear(net.head.in_features, num_classes)
    # for name, param in net.named_parameters():
    #     print(name, param.shape)
    # exit()


################################# logging #################################
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
    return trainable_params, all_param
# writeout args and trainable parameters to json
# num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
# total_params = sum(p.numel() for p in net.parameters())
train_params, total_params = print_trainable_parameters(net)
import json
with open(os.path.join(exp_dir, f'{args.net}_patch{args.patch}.json'), 'w') as f:
    n_params_dict = {"trainable_params": train_params, "total_params": total_params}
    json.dump(vars(args)|n_params_dict, f, indent=4)
    # f.write("\n")
    # json.dump(, f, indent=4)
###########################################################################


# For Multi-GPU
# if 'cuda' in device:
#     print(device)
#     print("using data parallel")
#     net = torch.nn.DataParallel(net) # make parallel
#     cudnn.benchmark = True
if 'cuda' in device:
    print(device)
    if torch.cuda.device_count() > 1:
        print("using data parallel")
        net = torch.nn.DataParallel(net) # make parallel
    cudnn.benchmark = True

    
if args.resume_path is not None:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(args.resume_path), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(args.resume_path)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

# Loss is CE
criterion = nn.CrossEntropyLoss()

if args.opt == "adam":
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
elif args.opt == "sgd":
    optimizer = optim.SGD(net.parameters(), lr=args.lr)  
    
# use cosine scheduling
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs)
# if args.LoRA:
#     scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, max_epochs=args.n_epochs)
# scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=10)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer)


##### Training
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        # Train with amp
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = net(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return train_loss/(batch_idx+1)

##### Validation
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    

    # Save checkpoint.
    acc = 100.*correct/total
    # if acc > best_acc:
    #     print('Saving best model..')
    #     state = {"model": net.state_dict(),
    #           "optimizer": optimizer.state_dict(),
    #           "scaler": scaler.state_dict()}
    #     # if not os.path.isdir(args.checkpoint_dir):
    #     #     os.mkdir(args.checkpoint_dir)
    #     torch.save(state, f'{exp_dir}/{args.net}-{args.patch}-ckpt-best.pth')
    #     best_acc = acc
    
    if (epoch+1) % args.save_freq == 0:
        print('Saving epoch{} model..'.format(epoch))
        state = {"model": net.state_dict(),
              "optimizer": optimizer.state_dict(),
              "scaler": scaler.state_dict()}
        torch.save(state, f'{exp_dir}/{args.net}-{args.patch}-ckpt-epoch{epoch}.pth')
    
    
    content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, val loss: {test_loss:.5f}, acc: {(acc):.5f}'
    print(content)
    with open(os.path.join(exp_dir, f'{args.net}_patch{args.patch}.txt'), 'a') as appender:
        appender.write(content + "\n")
    return test_loss, acc

list_loss = []
list_acc = []

if usewandb:
    wandb.watch(net)
    
# Check weights
fc_dict = {}
qkv_dict = {}
# for name, param in net.named_parameters():
#     print(name, param.shape)
#     if "weight" in name and param.shape == torch.Size([512, 512]):
#         fc_dict[name]=[]
#     if "weight" in name and param.shape == torch.Size([3*512, 512]):
#         qkv_dict[name]=[]

net.cuda()
for epoch in range(start_epoch, args.n_epochs):
    start = time.time()
    trainloss = train(epoch)
    val_loss, acc = test(epoch)
    
    # Check weights
    # if epoch % 2 == 0:
    #     for name, param in net.named_parameters():
    #         if "weight" in name and param.shape == torch.Size([512, 512]):
    #             fc_dict[name].append(param.detach().clone().cpu().numpy())
    #         if "weight" in name and param.shape == torch.Size([3*512, 512]):
    #             qkv_dict[name].append(param.detach().clone().cpu().numpy())

    scheduler.step() # step cosine scheduling

    list_loss.append(val_loss)
    list_acc.append(acc)
    
    # Log training..
    if usewandb:
        wandb.log({'epoch': epoch, 'train_loss': trainloss, 'val_loss': val_loss, "val_acc": acc, "lr": optimizer.param_groups[0]["lr"],
        "epoch_time": time.time()-start})

    # Write out csv..
    with open(os.path.join(exp_dir, f'{args.net}_patch{args.patch}.csv'), 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(list_loss) 
        writer.writerow(list_acc) 
    print(list_loss)

# for name, weights in fc_dict.items():
#     fc_dict[name] = np.stack(weights, axis=0)

# for name, weights in qkv_dict.items():
#     qkv_dict[name] = np.stack(weights, axis=0)

# np.savez(os.path.join(exp_dir, f'{args.net}_patch{args.patch}_fc'), **fc_dict)
# np.savez(os.path.join(exp_dir, f'{args.net}_patch{args.patch}_qkv'), **qkv_dict)


# writeout wandb
if usewandb:
    wandb.save("wandb_{}.h5".format(args.net))




