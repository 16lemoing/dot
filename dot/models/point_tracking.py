from tqdm import tqdm
import torch
from torch import nn

from .optical_flow import OpticalFlow
from .shelf import CoTracker
from dot.utils.io import read_config
from dot.utils.torch import sample_points


class PointTracker(nn.Module):
    def __init__(self,  height, width, tracker_config, tracker_path, estimator_config, estimator_path):
        super().__init__()
        model_args = read_config(tracker_config)
        model_dict = {"cotracker": CoTracker}
        self.model = model_dict[model_args.name](model_args)
        if tracker_path is not None:
            device = next(self.model.parameters()).device
            self.model.load_state_dict(torch.load(tracker_path, map_location=device))
        self.optical_flow_estimator = OpticalFlow(height, width, estimator_config, estimator_path)

    def forward(self, data, mode, **kwargs):
        if mode == "tracks_at_motion_boundaries":
            return self.get_tracks_at_motion_boundaries(data, **kwargs)
        else:
            raise ValueError(f"Unknown mode {mode}")

    def get_tracks_at_motion_boundaries(self, data, num_tracks=8192, sim_tracks=2048, sample_mode="all", **kwargs):
        video = data["video"]
        N, S = num_tracks, sim_tracks
        B, T, _, H, W = video.shape
        assert B == 1
        assert N % S == 0

        # Define sampling strategy
        if sample_mode == "all":
            samples_per_step = [S // T for _ in range(T)]
            samples_per_step[0] += S - sum(samples_per_step)
            backward_tracking = True
            flip = False
        elif sample_mode == "first":
            samples_per_step = [0 for _ in range(T)]
            samples_per_step[0] += S
            backward_tracking = False
            flip = False
        elif sample_mode == "last":
            samples_per_step = [0 for _ in range(T)]
            samples_per_step[0] += S
            backward_tracking = False
            flip = True
        else:
            raise ValueError(f"Unknown sample mode {sample_mode}")

        if flip:
            video = video.flip(dims=[1])

        # Track batches of points
        tracks = []
        for _ in tqdm(range(N // S), desc="Track batch of points", leave=False):
            src_points = []
            for src_step, src_samples in enumerate(samples_per_step):
                if src_samples == 0:
                    continue
                tgt_step = src_step - 1 if src_step > 0 else src_step + 1
                data = {"src_frame": video[:, src_step], "tgt_frame": video[:, tgt_step]}
                pred = self.optical_flow_estimator(data, mode="motion_boundaries", **kwargs)
                motion_boundaries = pred["motion_boundaries"]
                src_points.append(sample_points(src_step, motion_boundaries[0], src_samples))
            src_points = torch.cat(src_points, dim=0)[None]
            traj, vis = self.model(video, src_points, backward_tracking)
            tracks.append(torch.cat([traj, vis[..., None]], dim=-1))
        tracks = torch.cat(tracks, dim=2)

        if flip:
            tracks = tracks.flip(dims=[1])

        return {"tracks": tracks}


