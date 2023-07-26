import argparse
from pathlib import Path
import h5py
import os, sys
import torch
from model.varnet import VarNet
from data.load_data import create_data_loaders
from collections import defaultdict
import numpy as np
if os.getcwd() + '/utils/model/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/model/')


    
def parse():
    parser = argparse.ArgumentParser(description='Test Varnet on FastMRI challenge Images',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--GPU_NUM', type=int, default=0, help='GPU number to allocate')
    parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('-n', '--net_name', type=Path, default='test_varnet', help='Name of network')
    parser.add_argument('-p', '--data_path', type=Path, default='/Data/leaderboard/', help='Directory of test data')
    parser.add_argument('-m', '--mask', type=str, default='acc4', choices=['acc4', 'acc8'], help='type of mask | acc4 or acc8')
    parser.add_argument('-o', '--output', type=str, default='./Dataset/reconstruct')
    parser.add_argument('-t', '--type', type=str, default='train')
    parser.add_argument('--cascade', type=int, default=1, help='Number of cascades | Should be less than 12')
    parser.add_argument('--chans', type=int, default=9, help='Number of channels for cascade U-Net')
    parser.add_argument('--sens_chans', type=int, default=4, help='Number of channels for sensitivity map U-Net')
    parser.add_argument("--input_key", type=str, default='kspace', help='Name of input key')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse()
    args.data_path = args.data_path / args.mask
    args.exp_dir = '../result' / args.net_name / 'checkpoints'
    args.forward_dir = '../result' / args.net_name / 'reconstructions_leaderboard' / args.mask
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)

    print ('Current cuda device ', torch.cuda.current_device())

    model = VarNet(num_cascades=args.cascade, 
                   chans=args.chans, 
                   sens_chans=args.sens_chans)
    model.to(device=device)
    
    checkpoint = torch.load(args.exp_dir / 'best_model.pt', map_location='cpu')
    forward_loader = create_data_loaders(data_path = '/Data/train', args = args, isforward = True)
    model.eval()
    reconstructions = defaultdict(dict)
    with torch.no_grad():
        for (mask, kspace, _, _, fnames, slices) in forward_loader:
            kspace = kspace.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            output = model(kspace, mask)

            for i in range(output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )

    args.output = args.output / args.type
    args.output.mkdir(exist_ok=True, parents=True)
    for fname, recons in reconstructions.items():
        with h5py.File(args.output / fname, 'w') as f:
            f.create_dataset('reconstruction', data=recons)
            image_path = '/Data/' + 'image/'+args.type+'/'+fname
            image_file = h5py.File(image_path, 'r')
            f.create_dataset('target', data=image_file['image_label'])
            f.create_dataset('input', data = image_file['image_input'])