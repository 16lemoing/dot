import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
from joblib import Parallel, delayed
import random
from tqdm import tqdm
import tensorflow_datasets as tfds

from dot.utils.options.preprocess_options import PreprocessOptions
from dot.utils.io import create_folder, write_video, read_video, write_tracks
from dot.models import create_model
from dot.data.movi_f_tf_dataset import create_point_tracking_dataset


def main(args):
    create_folder(args.data_root, exist_ok=True)
    video_indices = list(range(args.num_videos))
    random.shuffle(video_indices)
    worker_video_indices = split_work(video_indices, args.num_workers)
    if args.extract_movi_f:
        Parallel(n_jobs=args.num_workers)(
            delayed(extract_movi_f)(args.data_root, args.download_path, indices) for indices in worker_video_indices
        )
    if args.save_tracks:
        Parallel(n_jobs=args.num_workers)(
            delayed(save_tracks)(args.data_root, indices) for indices in worker_video_indices
        )


def split_work(data, num_workers):
    k, m = divmod(len(data), num_workers)
    return [data[i * k + min(i, m):(i+1) * k + min(i + 1, m)] for i in range(num_workers)]


def extract_movi_f(data_root, download_path, video_indices):
    dataset = create_point_tracking_dataset(data_dir=download_path)
    loader = tfds.as_numpy(dataset)
    loader_iter = iter(loader)
    for video_idx in tqdm(video_indices):
        video_path = os.path.join(data_root, "video", str(video_idx))
        track_path = os.path.join(data_root, "ground_truth")
        success = create_folder(video_path, exist_ok=False)
        create_folder(track_path, exist_ok=False)
        if not success:
            continue
        data = next(loader_iter)
        video = (data['video'] + 1) / 2
        write_video(video, video_path, dtype="numpy", channels="last")
        traj = data['target_points'].astype(np.float32)
        vis = (1 - data['occluded']).astype(np.float32)
        tracks = np.concatenate((traj, vis[..., None]), axis=-1)
        write_tracks(tracks, os.path.join(track_path, str(video_idx)))


def save_tracks(data_root, video_indices):
    model = create_model(args).cuda()
    for video_idx in tqdm(video_indices):
        track_path = os.path.join(data_root, model.name)
        video_path = os.path.join(data_root, "video", str(video_idx))
        create_folder(track_path, exist_ok=True)
        if os.path.exists(os.path.join(track_path, str(video_idx) + ".npy")):
            continue
        video = read_video(video_path).cuda()[None]
        with torch.no_grad():
            tracks = model({"video": video}, mode="tracks_at_motion_boundaries", **vars(args))["tracks"]
        tracks = tracks[0].permute(1, 0, 2)
        tracks = tracks.cpu().numpy()
        write_tracks(tracks, os.path.join(track_path, str(video_idx)))


if __name__ == "__main__":
    args = PreprocessOptions().parse_args()
    main(args)
    print("Done.")
