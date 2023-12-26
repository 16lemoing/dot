# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
import glob
import torch
import pickle as pkl
import numpy as np
import os.path as osp
import mediapy as media
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from typing import Mapping, Tuple


def resize_video(video: np.ndarray, output_size: Tuple[int, int]) -> np.ndarray:
    """Resize a video to output_size."""
    # If you have a GPU, consider replacing this with a GPU-enabled resize op,
    # such as a jitted jax.image.resize.  It will make things faster.
    return media.resize_video(video, output_size)


def sample_queries_first(
    target_occluded: np.ndarray,
    target_points: np.ndarray,
    frames: np.ndarray,
) -> Mapping[str, np.ndarray]:
    """Package a set of frames and tracks for use in TAPNet evaluations.
    Given a set of frames and tracks with no query points, use the first
    visible point in each track as the query.
    Args:
      target_occluded: Boolean occlusion flag, of shape [n_tracks, n_frames],
        where True indicates occluded.
      target_points: Position, of shape [n_tracks, n_frames, 2], where each point
        is [x,y] scaled between 0 and 1.
      frames: Video tensor, of shape [n_frames, height, width, 3].  Scaled between
        -1 and 1.
    Returns:
      A dict with the keys:
        video: Video tensor of shape [1, n_frames, height, width, 3]
        query_points: Query points of shape [1, n_queries, 3] where
          each point is [t, y, x] scaled to the range [-1, 1]
        target_points: Target points of shape [1, n_queries, n_frames, 2] where
          each point is [x, y] scaled to the range [-1, 1]
    """
    valid = np.sum(~target_occluded, axis=1) > 0
    target_points = target_points[valid, :]
    target_occluded = target_occluded[valid, :]

    query_points = []
    for i in range(target_points.shape[0]):
        index = np.where(target_occluded[i] == 0)[0][0]
        x, y = target_points[i, index, 0], target_points[i, index, 1]
        query_points.append(np.array([index, y, x]))  # [t, y, x]
    query_points = np.stack(query_points, axis=0)

    return {
        "video": frames[np.newaxis, ...],
        "query_points": query_points[np.newaxis, ...],
        "target_points": target_points[np.newaxis, ...],
        "occluded": target_occluded[np.newaxis, ...],
    }


def sample_queries_strided(
    target_occluded: np.ndarray,
    target_points: np.ndarray,
    frames: np.ndarray,
    query_stride: int = 5,
) -> Mapping[str, np.ndarray]:
    """Package a set of frames and tracks for use in TAPNet evaluations.

    Given a set of frames and tracks with no query points, sample queries
    strided every query_stride frames, ignoring points that are not visible
    at the selected frames.

    Args:
      target_occluded: Boolean occlusion flag, of shape [n_tracks, n_frames],
        where True indicates occluded.
      target_points: Position, of shape [n_tracks, n_frames, 2], where each point
        is [x,y] scaled between 0 and 1.
      frames: Video tensor, of shape [n_frames, height, width, 3].  Scaled between
        -1 and 1.
      query_stride: When sampling query points, search for un-occluded points
        every query_stride frames and convert each one into a query.

    Returns:
      A dict with the keys:
        video: Video tensor of shape [1, n_frames, height, width, 3].  The video
          has floats scaled to the range [-1, 1].
        query_points: Query points of shape [1, n_queries, 3] where
          each point is [t, y, x] scaled to the range [-1, 1].
        target_points: Target points of shape [1, n_queries, n_frames, 2] where
          each point is [x, y] scaled to the range [-1, 1].
        trackgroup: Index of the original track that each query point was
          sampled from.  This is useful for visualization.
    """
    tracks = []
    occs = []
    queries = []
    trackgroups = []
    total = 0
    trackgroup = np.arange(target_occluded.shape[0])
    for i in range(0, target_occluded.shape[1], query_stride):
        mask = target_occluded[:, i] == 0
        query = np.stack(
            [
                i * np.ones(target_occluded.shape[0:1]),
                target_points[:, i, 1],
                target_points[:, i, 0],
            ],
            axis=-1,
        )
        queries.append(query[mask])
        tracks.append(target_points[mask])
        occs.append(target_occluded[mask])
        trackgroups.append(trackgroup[mask])
        total += np.array(np.sum(target_occluded[:, i] == 0))

    return {
        "video": frames[np.newaxis, ...],
        "query_points": np.concatenate(queries, axis=0)[np.newaxis, ...],
        "target_points": np.concatenate(tracks, axis=0)[np.newaxis, ...],
        "occluded": np.concatenate(occs, axis=0)[np.newaxis, ...],
        "trackgroup": np.concatenate(trackgroups, axis=0)[np.newaxis, ...],
    }


class TapVid(Dataset):
    def __init__(
        self,
        data_root,
        split="davis",
        query_mode="first",
        resize_to_256=True
    ):
        self.split = split
        self.resize_to_256 = resize_to_256
        self.query_mode = query_mode
        if self.split == "kinetics":
            all_paths = glob.glob(osp.join(data_root, "*_of_0010.pkl"))
            points_dataset = []
            for pickle_path in all_paths:
                with open(pickle_path, "rb") as f:
                    data = pkl.load(f)
                    points_dataset = points_dataset + data
            self.points_dataset = points_dataset
        else:
            with open(data_root, "rb") as f:
                self.points_dataset = pkl.load(f)
            if self.split == "davis":
                self.video_names = list(self.points_dataset.keys())
        print("found %d unique videos in %s" % (len(self.points_dataset), data_root))

    def __getitem__(self, index):
        if self.split == "davis":
            video_name = self.video_names[index]
        else:
            video_name = index
        video = self.points_dataset[video_name]
        frames = video["video"]

        if isinstance(frames[0], bytes):
            # TAP-Vid is stored and JPEG bytes rather than `np.ndarray`s.
            def decode(frame):
                byteio = io.BytesIO(frame)
                img = Image.open(byteio)
                return np.array(img)

            frames = np.array([decode(frame) for frame in frames])

        target_points = self.points_dataset[video_name]["points"]
        if self.resize_to_256:
            frames = resize_video(frames, [256, 256])
            target_points *= np.array([256, 256])
        else:
            target_points *= np.array([frames.shape[2], frames.shape[1]])

        target_occ = self.points_dataset[video_name]["occluded"]
        if self.query_mode == "first":
            converted = sample_queries_first(target_occ, target_points, frames)
        else:
            converted = sample_queries_strided(target_occ, target_points, frames)
        assert converted["target_points"].shape[1] == converted["query_points"].shape[1]

        trajs = torch.from_numpy(converted["target_points"])[0].permute(1, 0, 2).float() # T, N, D

        rgbs = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.
        visibles = torch.logical_not(torch.from_numpy(converted["occluded"]))[0].permute(1, 0)  # T, N
        query_points = torch.from_numpy(converted["query_points"])[0].float() # T, N
        tracks = torch.cat([trajs, visibles[..., None]], dim=-1)

        data = {
            "video": rgbs,
            "query_points": query_points,
            "tracks": tracks
        }

        return data

    def __len__(self):
        return len(self.points_dataset)


def create_point_tracking_dataset(args):
    data_root = osp.join(args.data_root, f"tapvid_{args.split}")
    if args.split in ["davis", "rgb_stacking"]:
        data_root = osp.join(data_root, f"tapvid_{args.split}.pkl")
    dataset = TapVid(data_root, args.split, args.query_mode)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    return dataloader