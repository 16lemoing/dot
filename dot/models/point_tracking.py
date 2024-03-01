from tqdm import tqdm
import torch
from torch import nn

from .optical_flow import OpticalFlow
from .shelf import CoTracker, CoTracker2, Tapir
from dot.utils.io import read_config
from dot.utils.torch import sample_points, sample_mask_points, get_grid


class PointTracker(nn.Module):
    def __init__(self,  height, width, tracker_config, tracker_path, estimator_config, estimator_path):
        super().__init__()
        model_args = read_config(tracker_config)
        model_dict = {
            "cotracker": CoTracker,
            "cotracker2": CoTracker2,
            "tapir": Tapir,
            "bootstapir": Tapir
        }
        self.name = model_args.name
        self.model = model_dict[model_args.name](model_args)
        if tracker_path is not None:
            device = next(self.model.parameters()).device
            self.model.load_state_dict(torch.load(tracker_path, map_location=device), strict=False)
        self.optical_flow_estimator = OpticalFlow(height, width, estimator_config, estimator_path)

    def forward(self, data, mode, **kwargs):
        if mode == "tracks_at_motion_boundaries":
            return self.get_tracks_at_motion_boundaries(data, **kwargs)
        elif mode == "flow_from_last_to_first_frame":
            return self.get_flow_from_last_to_first_frame(data, **kwargs)
        else:
            raise ValueError(f"Unknown mode {mode}")

    def get_tracks_at_motion_boundaries(self, data, num_tracks=8192, sim_tracks=2048, sample_mode="all", **kwargs):
        video = data["video"]
        N, S = num_tracks, sim_tracks
        B, T, _, H, W = video.shape
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
        motion_boundaries = {}
        cache_features = True
        for _ in tqdm(range(N // S), desc="Track batch of points", leave=False):
            src_points = []
            for src_step, src_samples in enumerate(samples_per_step):
                if src_samples == 0:
                    continue
                if not src_step in motion_boundaries:
                    tgt_step = src_step - 1 if src_step > 0 else src_step + 1
                    data = {"src_frame": video[:, src_step], "tgt_frame": video[:, tgt_step]}
                    pred = self.optical_flow_estimator(data, mode="motion_boundaries", **kwargs)
                    motion_boundaries[src_step] = pred["motion_boundaries"]
                src_boundaries = motion_boundaries[src_step]
                src_points.append(sample_points(src_step, src_boundaries, src_samples))
            src_points = torch.cat(src_points, dim=1)
            traj, vis = self.model(video, src_points, backward_tracking, cache_features)
            tracks.append(torch.cat([traj, vis[..., None]], dim=-1))
            cache_features = False
        tracks = torch.cat(tracks, dim=2)

        if flip:
            tracks = tracks.flip(dims=[1])

        return {"tracks": tracks}

    def get_flow_from_last_to_first_frame(self, data, sim_tracks=2048, **kwargs):
        video = data["video"]
        video = video.flip(dims=[1])
        src_step = 0  # We have flipped video over temporal axis so src_step is 0
        B, T, C, H, W = video.shape
        S = sim_tracks
        backward_tracking = False
        cache_features = True
        flow = get_grid(H, W, shape=[B]).cuda()
        flow[..., 0] = flow[..., 0] * (W - 1)
        flow[..., 1] = flow[..., 1] * (H - 1)
        alpha = torch.zeros(B, H, W).cuda()
        mask = torch.ones(H, W)
        pbar = tqdm(total=H * W // S, desc="Track batch of points", leave=False)
        while torch.any(mask):
            points, (i, j) = sample_mask_points(src_step, mask, S)
            idx = i * W + j
            points = points.cuda()[None].expand(B, -1, -1)

            traj, vis = self.model(video, points, backward_tracking, cache_features)
            traj = traj[:, -1]
            vis = vis[:, -1].float()

            # Update mask
            mask = mask.view(-1)
            mask[idx] = 0
            mask = mask.view(H, W)

            # Update flow
            flow = flow.view(B, -1, 2)
            flow[:, idx] = traj - flow[:, idx]
            flow = flow.view(B, H, W, 2)

            # Update alpha
            alpha = alpha.view(B, -1)
            alpha[:, idx] = vis
            alpha = alpha.view(B, H, W)

            cache_features = False
            pbar.update(1)
        pbar.close()
        return {"flow": flow, "alpha": alpha}
