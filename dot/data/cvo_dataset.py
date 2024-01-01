import os
import os.path as osp
from collections import OrderedDict

import lmdb
import torch
import numpy as np
import pickle as pkl
from einops import rearrange
from torch.utils.data import Dataset, DataLoader

from dot.utils.torch import get_alpha_consistency


class CVO_sampler_lmdb:
    """Data sampling"""

    all_keys = ["imgs", "imgs_blur", "fflows", "bflows", "delta_fflows", "delta_bflows"]

    def __init__(self, data_root, keys=None, split=None):
        if split == "extended":
            self.db_path = osp.join(data_root, "cvo_test_extended.lmdb")
        else:
            self.db_path = osp.join(data_root, "cvo_test.lmdb")
        self.split = split

        self.env = lmdb.open(
            self.db_path,
            subdir=os.path.isdir(self.db_path),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.env.begin(write=False) as txn:
            self.samples = pkl.loads(txn.get(b"__samples__"))
            self.length = len(self.samples)

        self.keys = self.all_keys if keys is None else [x.lower() for x in keys]
        self._check_keys(self.keys)

    def _check_keys(self, keys):
        # check keys are supported:
        for k in keys:
            assert k in self.all_keys, f"Invalid key value: {k}"

    def __len__(self):
        return self.length

    def sample(self, index):
        sample = OrderedDict()
        with self.env.begin(write=False) as txn:
            for k in self.keys:
                key = "{:05d}_{:s}".format(index, k)
                value = pkl.loads(txn.get(key.encode()))
                if "flow" in key and self.split in ["clean", "final"]:  # Convert Int to Floating
                    value = value.astype(np.float32)
                    value = (value - 2 ** 15) / 128.0
                if "imgs" in k:
                    k = "imgs"
                sample[k] = value
        return sample


class CVO(Dataset):
    all_keys = ["fflows", "bflows"]

    def __init__(self, data_root, keys=None, split="clean", crop_size=256):
        keys = self.all_keys if keys is None else [x.lower() for x in keys]
        self._check_keys(keys)
        if split == "final":
            keys.append("imgs_blur")
        else:
            keys.append("imgs")
        self.split = split
        self.sampler = CVO_sampler_lmdb(data_root, keys, split)

    def __getitem__(self, index):
        sample = self.sampler.sample(index)

        video = torch.from_numpy(sample["imgs"].copy())
        video = video / 255.0
        video = rearrange(video, "h w (t c) -> t c h w", c=3)

        fflow = torch.from_numpy(sample["fflows"].copy())
        fflow = rearrange(fflow, "h w (t c) -> t h w c", c=2)[-1]

        bflow = torch.from_numpy(sample["bflows"].copy())
        bflow = rearrange(bflow, "h w (t c) -> t h w c", c=2)[-1]

        if self.split in ["clean", "final"]:
            thresh_1 = 0.01
            thresh_2 = 0.5
        elif self.split == "extended":
            thresh_1 = 0.1
            thresh_2 = 0.5
        else:
            raise ValueError(f"Unknown split {self.split}")

        alpha = get_alpha_consistency(bflow[None], fflow[None], thresh_1=thresh_1, thresh_2=thresh_2)[0]

        data = {
            "video": video,
            "alpha": alpha,
            "flow": bflow
        }

        return data

    def _check_keys(self, keys):
        # check keys are supported:
        for k in keys:
            assert k in self.all_keys, f"Invalid key value: {k}"

    def __len__(self):
        return len(self.sampler)


def create_optical_flow_dataset(args):
    dataset = CVO(args.data_root, split=args.split)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    return dataloader