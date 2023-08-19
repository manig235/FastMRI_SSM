import h5py
import random
from utils.data.transforms import DataTransform
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

class SliceData(Dataset):
    def __init__(self, root, transform, input_key, grappa_key, target_key, forward=False):
        self.transform = transform
        self.input_key = input_key
        self.target_key = target_key
        self.grappa_key = grappa_key
        self.forward = forward
        self.examples = []
        self.examples_2 = []
        
        files = list(Path(root).iterdir())
        for fname in sorted(files):
            num_slices = self._get_metadata(fname)

            self.examples += [
                (fname, slice_ind) for slice_ind in range(num_slices)
            ]
        root2 = str(root).replace('reconstruct', 'reconstruct_cascade8')
        files_2 = list(Path(root2).iterdir())
        for fname in sorted(files_2):
            num_slices = self._get_metadata(fname)

            self.examples_2 += [
                (fname, slice_ind) for slice_ind in range(num_slices)
            ]
    def _get_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            num_slices = hf[self.input_key].shape[0]
        return num_slices

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, dataslice = self.examples[i]
        fname_2, dataslice = self.examples_2[i]
#         if not self.forward:
#             base_path = Path('/Data/train/image')
#         else:
#             base_path = Path('/Data/val/image')
        with h5py.File(fname, "r") as hf:
            input_1 = hf[self.input_key][dataslice]
#            print(fname)
            if self.forward:
                target = -1
            else:
                target = hf[self.target_key][dataslice]
            attrs = dict(hf.attrs)
            grappa = hf[self.grappa_key][dataslice]
        with h5py.File(fname_2, "r") as hf:
            input_2= hf[self.input_key][dataslice]
#            print(fname_2)
        return self.transform(input_1, input_2,  grappa, target, attrs, fname.name, dataslice)


def create_data_loaders(data_path, args, shuffle=False, isforward=False):
    if isforward == False:
        max_key_ = args.max_key
        target_key_ = args.target_key
    else:
        max_key_ = -1
        target_key_ = -1
    data_storage = SliceData(
        root=data_path,
        transform=DataTransform(isforward, max_key_),
        input_key=args.input_key,
        grappa_key = args.grappa_key,
        target_key=target_key_,
        forward = isforward
    )

    data_loader = DataLoader(
        dataset=data_storage,
        batch_size=args.batch_size,
        shuffle=shuffle,
    )
    return data_loader
