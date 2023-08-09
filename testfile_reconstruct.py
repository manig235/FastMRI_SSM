import argparse
from pathlib import Path
import h5py
import os, sys
import torch
from utils.common.utils import kspace2image
if os.getcwd() + '/utils/model/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/model/')

from utils.learning.train_part import train

if os.getcwd() + '/utils/common/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/common/')
from utils.common.utils import seed_fix

if os.getcwd() + '/utils/data/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/data/')
from varnet import VarNet
from load_data import create_data_loaders
from collections import defaultdict
import numpy as np
def parse():
    parser = argparse.ArgumentParser(description='Test Varnet on FastMRI challenge Images',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--GPU_NUM', type=int, default=0, help='GPU number to allocate')
    parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('-n', '--net_name', type=Path, default=Path('test_varnet'), help='Name of network')
    parser.add_argument('-p', '--data_path', type=Path, default=Path('/Data/leaderboard/'), help='Directory of test data')
    parser.add_argument('-m', '--mask', type=str, default='acc4', choices=['acc4', 'acc8'], help='type of mask | acc4 or acc8')
    parser.add_argument('-o', '--output', type=Path, default=Path('../reconstruct'))
    parser.add_argument('-t', '--type', type=Path, default=Path('train'))
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
    forward_loader = create_data_loaders(data_path = '/Data'/args.type, args = args, isforward = True)
    model.eval()
    reconstructions = defaultdict(dict)
    iter = 0
    args.output = args.output / args.type / Path("image")
#     output_folder_list = os.listdir(args.output)
    prev_file = ''
    with torch.no_grad():
        for (mask, kspace, target_kspace, _, _, fnames, slices) in forward_loader:
            if prev_file == '':
                prev_file = fnames[0]
            kspace = kspace.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            output = model(kspace, mask).cpu().numpy()
            for i in range(output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i]
            if prev_file != fnames[0]:
                print(prev_file)
                print("writing")
                fname = prev_file
                reconstructions[fname] = np.stack(
                    [out for _, out in sorted(reconstructions[fname].items())]
                )
                args.output.mkdir(exist_ok=True, parents=True)
                with h5py.File(args.output / Path(fname), 'w') as f:
                    print(str(args.output / Path(fname)))
                    f.create_dataset('recons', data=reconstructions[fname])
                    image_path = Path('/Data')/args.type/Path('image/'+fname)

                    image_file = h5py.File(image_path, 'r')
                    attrs = dict(image_file.attrs)
                    f.create_dataset('target', data=image_file['image_label'])
                    f.create_dataset('input', data = image_file['image_input'])
                    f.create_dataset('grappa', data = image_file['image_grappa'])
                    f.attrs.create("max", attrs["max"])
                    f.attrs.create("norm", attrs["norm"])
                print(reconstructions.keys())
                reconstructions.pop(prev_file, None)
                prev_file = fnames[0]
                print(reconstructions.keys())
        print("writing last reconstruction")
        for fname in reconstructions:
            reconstructions[fname] = np.stack(
                [out for _, out in sorted(reconstructions[fname].items())]
            )
            args.output.mkdir(exist_ok=True, parents=True)
            with h5py.File(args.output / Path(fname), 'w') as f:
                print(str(args.output / Path(fname)))
                f.create_dataset('recons', data=reconstructions[fname])
                image_path = Path('/Data')/args.type/Path('image/'+fname)

                image_file = h5py.File(image_path, 'r')
                attrs = dict(image_file.attrs)
                f.create_dataset('target', data=image_file['image_label'])
                f.create_dataset('input', data = image_file['image_input'])
                f.create_dataset('grappa', data = image_file['image_grappa'])
                f.attrs.create("max", attrs["max"])
                f.attrs.create("norm", attrs["norm"])
        """
            if prev_file == '':
                prev_file = fnames[0]
            output = model(kspace, mask).cpu().numpy()
            output = np.array(output).astype('complex128')
            output[...,0] = output[...,0] + output[...,1]*1j
            output = output[...,0]
            iter+=1
            for i in range(output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i]
                fname = fnames[i]
                if prev_file != fname: 
                    recons = np.stack([out for _, out in sorted(reconstructions[prev_file].items())])
                    print(prev_file)
                    print(reconstructions[prev_file][0].shape)
                    args.output.mkdir(exist_ok=True, parents=True) 
                    with h5py.File(args.output / Path(prev_file), 'w') as f:
                        print(str(args.output / Path(prev_file)))
                        f.create_dataset('kspace_recons', data=recons)
                        image_path = Path('/Data')/args.type/Path('image/'+prev_file)
                        image_file = h5py.File(image_path, 'r')
                        f.create_dataset('target', data=image_file['image_label'])
                        f.create_dataset('input', data = image_file['image_input'])
                    reconstructions.pop(prev_file, None)
                    print(reconstructions.keys())
                    prev_file = fname
        recons = np.stack([out for _, out in sorted(reconstructions[fname].items())])
        print(fname)
        args.output.mkdir(exist_ok=True, parents=True)
        with h5py.File(args.output / Path(fname), 'w') as f:
            print(str(args.output / Path(fname)))
            f.create_dataset('kspace_recons', data=recons)
            image_path = Path('/Data')/args.type/Path('image/'+fname)
            image_file = h5py.File(image_path, 'r')
            f.create_dataset('target', data=image_file['image_label'])
            f.create_dataset('input', data = image_file['image_input'])
        reconstructions.pop(fname, None)
        """