import os
from glob import glob
import random
import numpy as np
import torch
from torch.utils import data

from dot.utils.io import read_video, read_tracks


def create_point_tracking_dataset(args, batch_size=1, split="train", num_workers=None, verbose=False):
    dataset = Dataset(args, split, verbose)
    dataloader = DataLoader(args, dataset, batch_size, split, num_workers)
    return dataloader


class DataLoader:
    def __init__(self, args, dataset, batch_size=1, split="train", num_workers=None):
        num_workers = args.num_workers if num_workers is None else num_workers
        is_train = split == "train"
        self.sampler = data.distributed.DistributedSampler(dataset, args.world_size, args.rank) if is_train else None
        self.loader = data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=self.sampler,
        )
        self.epoch = -1
        self.reinit()

    def reinit(self):
        self.epoch += 1
        if self.sampler:
            self.sampler.set_epoch(self.epoch)
        self.iter = iter(self.loader)

    def next(self):
        try:
            return next(self.iter)
        except StopIteration:
            self.reinit()
            return next(self.iter)


def get_correspondences(track_path, src_step, tgt_step, num_tracks, height, width, vis_src_only):
    H, W = height, width
    tracks = torch.from_numpy(read_tracks(track_path))
    tracks[..., 0] = tracks[..., 0] / (W - 1)
    tracks[..., 1] = tracks[..., 1] / (H - 1)
    src_points = tracks[:, src_step]
    tgt_points = tracks[:, tgt_step]
    if vis_src_only:
        src_alpha = src_points[..., 2]
        vis_idx = torch.nonzero(src_alpha, as_tuple=True)[0]
        num_vis = vis_idx.shape[0]
        if num_vis == 0:
            return False, None
        samples = np.random.choice(num_vis, num_tracks, replace=num_tracks > num_vis)
        idx = vis_idx[samples]
    else:
        idx = np.random.choice(tracks.size(0), num_tracks, replace=num_tracks > tracks.size(0))
    return True, (src_points[idx], tgt_points[idx])


class Dataset(data.Dataset):
    def __init__(self, args, split="train", verbose=False):
        super().__init__()
        self.video_folder = os.path.join(args.data_root, "video")
        self.in_track_folder = os.path.join(args.data_root, args.in_track_name)
        self.out_track_folder = os.path.join(args.data_root, args.out_track_name)
        self.num_in_tracks = args.num_in_tracks
        self.num_out_tracks = args.num_out_tracks
        num_videos = len(glob(os.path.join(self.video_folder, "*")))
        self.video_steps = [
            len(glob(os.path.join(self.video_folder, str(video_idx), "*"))) for video_idx in range(num_videos)
        ]
        video_indices = list(range(num_videos))
        if split == "valid":
            video_indices = video_indices[:int(num_videos * args.valid_ratio)]
        elif split == "train":
            video_indices = video_indices[int(num_videos * args.valid_ratio):]
        self.video_indices = video_indices
        self.num_videos = len(video_indices)
        if verbose:
            print(f"Created {split} dataset of length {self.num_videos}")

    def __len__(self):
        return self.num_videos

    def __getitem__(self, idx):
        idx = idx % self.num_videos
        video_idx = self.video_indices[idx]
        time_steps = self.video_steps[video_idx]
        src_step = random.randrange(time_steps)
        tgt_step = random.randrange(time_steps - 1)
        tgt_step = (src_step + tgt_step) % time_steps

        video_path = os.path.join(self.video_folder, str(video_idx))
        src_frame = read_video(video_path, start_step=src_step, time_steps=1)[0]
        tgt_frame = read_video(video_path, start_step=tgt_step, time_steps=1)[0]
        _, H, W = src_frame.shape

        in_track_path = os.path.join(self.in_track_folder, f"{video_idx}.npy")
        out_track_path = os.path.join(self.out_track_folder, f"{video_idx}.npy")
        vis_src_only = False
        _, corr = get_correspondences(in_track_path, src_step, tgt_step, self.num_in_tracks, H, W, vis_src_only)
        src_points, tgt_points = corr

        vis_src_only = True
        success, corr = get_correspondences(out_track_path, src_step, tgt_step, self.num_out_tracks, H, W, vis_src_only)
        if not success:
            return self[idx + 1]
        out_src_points, out_tgt_points = corr

        data = {
            "src_frame": src_frame,
            "tgt_frame": tgt_frame,
            "src_points": src_points,
            "tgt_points": tgt_points,
            "out_src_points": out_src_points,
            "out_tgt_points": out_tgt_points,
        }

        return data
