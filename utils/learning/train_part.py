import shutil
import numpy as np
import torch
import torch.nn as nn
import time

from collections import defaultdict
from utils.data.load_data import create_data_loaders
from utils.common.utils import save_reconstructions, ssim_loss
from utils.common.loss_function import SSIMLoss, ConsistencyLoss
from utils.model.unet import Unet

LAMBDA_CONS = 10

def train_epoch(args, epoch, model, data_loader, optimizer, device):
    model.train()
    start_epoch = start_iter = time.perf_counter()
    len_loader = len(data_loader)
    total_loss = 0.
    total_loss_ssim = 0.
    total_loss_cons = 0.
    for iter, data in enumerate(data_loader):
        input, target, maximum, _, _ = data
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        maximum = maximum.cuda(non_blocking=True)

        output = model(input)
        ssim_loss = SSIMLoss().to(device=device)
        cons_loss = ConsistencyLoss.to(device=device)
        loss_ssim = ssim_loss(output, target, maximum)
        loss_cons = cons_loss(output, target)
        loss = loss_ssim+LAMBDA_CONS*loss_cons
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss_ssim += loss_ssim.total()
        total_loss_cons +=loss_cons.total()
        total_loss += loss.item()
        loss_dict = dict()
        
        if iter % args.report_interval == 0:
            print(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
        start_iter = time.perf_counter()
    total_loss_ssim = total_loss_ssim / len_loader
    total_loss_cons = total_loss_cons / len_loader
    total_loss = total_loss / len_loader
    loss_dict['SSIM'] = total_loss_ssim
    loss_dict['Consistency'] = total_loss_cons
    loss_dict['Total'] = total_loss
    return loss_dict, time.perf_counter() - start_epoch


def validate(args, model, data_loader):
    model.eval()
    reconstructions = defaultdict(dict)
    targets = defaultdict(dict)
    inputs = defaultdict(dict)
    start = time.perf_counter()

    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            input, target, _, fnames, slices = data
            input = input.cuda(non_blocking=True)
            output = model(input)

            for i in range(output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()
                targets[fnames[i]][int(slices[i])] = target[i].numpy()
                inputs[fnames[i]][int(slices[i])] = input[i].cpu().numpy()

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    for fname in targets:
        targets[fname] = np.stack(
            [out for _, out in sorted(targets[fname].items())]
        )
    for fname in inputs:
        inputs[fname] = np.stack(
            [out for _, out in sorted(inputs[fname].items())]
        )
        metric_loss = sum([ssim_loss(targets[fname], reconstructions[fname]) for fname in reconstructions])
    num_subjects = len(reconstructions)
    return metric_loss, num_subjects, reconstructions, targets, inputs, time.perf_counter() - start


def save_model(args, exp_dir, epoch, model, optimizer, best_val_loss, is_new_best):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'exp_dir': exp_dir
        },
        f=exp_dir / 'model.pt'
    )
    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')


        
def train(args):
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print('Current cuda device: ', torch.cuda.current_device())
    
    model = Unet(in_chans = args.in_chans, out_chans = args.out_chans)
    model.to(device=device)
    
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    best_val_loss = 1.
    start_epoch = 0

    
    train_loader = create_data_loaders(data_path = args.data_path_train, args = args, shuffle=True)
    val_loader = create_data_loaders(data_path = args.data_path_val, args = args)

    val_loss_log = np.empty((0, 2))
    for epoch in range(start_epoch, args.num_epochs):
        print(f'Epoch #{epoch:2d} ............... {args.net_name} ...............')
        
        train_loss_dict, train_time = train_epoch(args, epoch, model, train_loader, optimizer)
        val_loss_dict, num_subjects, reconstructions, targets, inputs, val_time = validate(args, model, val_loader)

        val_loss_log = np.append(val_loss_log, np.array([[epoch, val_loss]]), axis=0)
        file_path = args.val_loss_dir / "val_loss_log"
        np.save(file_path, val_loss_log)
        print(f"loss file saved! {file_path}")

        val_loss = val_loss / num_subjects

        is_new_best = val_loss < best_val_loss
        best_val_loss = min(best_val_loss, val_loss)

        save_model(args, args.exp_dir, epoch + 1, model, optimizer, best_val_loss, is_new_best)
        print(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainTime = {train_time:.4f}s ValTime = {val_time:.4f}s'
        )
        print('Train')
        for keys,item in train_loss_dict:
            print(keys, "Loss : ", item, end = ' ')
        print('AvgSSIM :', 1-train_loss_dict['SSIM'])
        print('VALIDATION')
        for keys,item in train_loss_dict:
            print(keys, "Loss : ", item, end = ' ')
        print('AvgSSIM :', 1-val_loss_dict['SSIM'])
        if is_new_best:
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@NewRecord@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            start = time.perf_counter()
            save_reconstructions(reconstructions, args.val_dir, targets=targets, inputs=inputs)
            print(
                f'ForwardTime = {time.perf_counter() - start:.4f}s',
            )
