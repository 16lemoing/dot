import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from einops import rearrange, repeat

from .optical_flow import OpticalFlow
from .point_tracking import PointTracker
from dot.utils.torch import get_grid


class DenseOpticalTracker(nn.Module):
    def __init__(self,
                 height=512,
                 width=512,
                 tracker_config="configs/cotracker2_patch_4_wind_8.json",
                 tracker_path="checkpoints/movi_f_cotracker2_patch_4_wind_8.pth",
                 estimator_config="configs/raft_patch_8.json",
                 estimator_path="checkpoints/cvo_raft_patch_8.pth",
                 refiner_config="configs/raft_patch_4_alpha.json",
                 refiner_path="checkpoints/movi_f_raft_patch_4_alpha.pth"):
        super().__init__()
        self.point_tracker = PointTracker(height, width, tracker_config, tracker_path, estimator_config, estimator_path)
        self.optical_flow_refiner = OpticalFlow(height, width, refiner_config, refiner_path)
        self.resolution = [height, width]

    def forward(self, data, mode, **kwargs):
        if mode == "flow_from_last_to_first_frame":
            return self.get_flow_from_last_to_first_frame(data, **kwargs)
        if mode == "tracks_for_queries":
            return self.get_tracks_for_queries(data, **kwargs)
        if mode == "tracks_from_first_to_every_other_frame":
            return self.get_tracks_from_first_to_every_other_frame(data, **kwargs)
        else:
            raise ValueError(f"Unknown mode {mode}")

    def get_flow_from_last_to_first_frame(self, data, **kwargs):
        B, T, C, h, w = data["video"].shape
        init = self.point_tracker(data, mode="tracks_at_motion_boundaries", **kwargs)["tracks"]
        init = torch.stack([init[..., 0] / (w - 1), init[..., 1] / (h - 1), init[..., 2]], dim=-1)
        data = {
            "src_frame": data["video"][:, -1],
            "tgt_frame": data["video"][:, 0],
            "src_points": init[:, -1],
            "tgt_points": init[:, 0]
        }
        pred = self.optical_flow_refiner(data, mode="flow_with_tracks_init", **kwargs)
        pred["src_points"] = data["src_points"]
        pred["tgt_points"] = data["tgt_points"]
        return pred

    def get_tracks_for_queries(self, data, **kwargs):
        time_steps = data["video"].size(1)
        query_points = data["query_points"]
        video = data["video"]
        S = query_points.size(1)
        B, T, C, h, w = video.shape
        assert B == 1
        H, W = self.resolution

        init = self.point_tracker(data, mode="tracks_at_motion_boundaries", **kwargs)["tracks"]
        init = torch.stack([init[..., 0] / (w - 1), init[..., 1] / (h - 1), init[..., 2]], dim=-1)

        if h != H or w != W:
            video = F.interpolate(video[0], size=(H, W), mode="bilinear")[None]

        feats = self.optical_flow_refiner({"video": video}, mode="feats", **kwargs)["feats"]

        grid = get_grid(H, W, device=video.device)
        src_steps = [int(v) for v in torch.unique(query_points[..., 0])]
        tracks = torch.zeros(B, T, S, 3, device=video.device)
        for src_step in tqdm(src_steps, desc="Refine source step", leave=False):
            src_points = init[:, src_step]
            src_feats = feats[:, src_step]
            tracks_from_src = []
            for tgt_step in tqdm(range(time_steps), desc="Refine target step", leave=False):
                if src_step == tgt_step:
                    flow = torch.zeros(B, H, W, 2, device=video.device)
                    alpha = torch.ones(B, H, W, device=video.device)
                else:
                    tgt_points = init[:, tgt_step]
                    tgt_feats = feats[:, tgt_step]
                    data = {
                        "src_feats": src_feats,
                        "tgt_feats": tgt_feats,
                        "src_points": src_points,
                        "tgt_points": tgt_points
                    }
                    pred = self.optical_flow_refiner(data, mode="flow_with_tracks_init", **kwargs)
                    flow, alpha = pred["flow"], pred["alpha"]
                    flow[..., 0] = flow[..., 0] / (W - 1)
                    flow[..., 1] = flow[..., 1] / (H - 1)
                tracks_from_src.append(torch.cat([flow + grid, alpha[..., None]], dim=-1))
            tracks_from_src = torch.stack(tracks_from_src, dim=1)
            cur = query_points[0, :, 0] == src_step
            cur_points = query_points[0, cur]
            cur_x = cur_points[..., 2] / (w - 1)
            cur_y = cur_points[..., 1] / (h - 1)
            cur_grid = torch.stack([cur_x, cur_y], dim=-1) * 2 - 1
            cur_grid = repeat(cur_grid, "s c -> t s r c", t=T, r=1)
            tracks_from_src = rearrange(tracks_from_src[0], "t h w c -> t c h w")
            cur_tracks = F.grid_sample(tracks_from_src, cur_grid, align_corners=True, mode="bilinear")
            cur_tracks = rearrange(cur_tracks[..., 0], "t c s -> t s c")
            cur_tracks[..., 0] = cur_tracks[..., 0] * (w - 1)
            cur_tracks[..., 1] = cur_tracks[..., 1] * (h - 1)
            cur_tracks[..., 2] = (cur_tracks[..., 2] > 0).float()
            tracks[:, :, cur] = cur_tracks
        return {"tracks": tracks}

    def get_tracks_from_first_to_every_other_frame(self, data, **kwargs):
        video = data["video"]
        B, T, C, h, w = video.shape
        assert B == 1
        H, W = self.resolution
        if h != H or w != W:
            video = F.interpolate(video[0], size=(H, W), mode="bilinear")[None]

        init = self.point_tracker(data, mode="tracks_at_motion_boundaries", **kwargs)["tracks"]
        init = torch.stack([init[..., 0] / (w - 1), init[..., 1] / (h - 1), init[..., 2]], dim=-1)

        grid = get_grid(H, W, device=video.device)
        grid[..., 0] *= (W - 1)
        grid[..., 1] *= (H - 1)
        src_step = 0
        src_points = init[:, src_step]
        src_frame = video[:, src_step]
        tracks = []
        for tgt_step in tqdm(range(T), desc="Refine target step", leave=False):
            if src_step == tgt_step:
                flow = torch.zeros(B, H, W, 2, device=video.device)
                alpha = torch.ones(B, H, W, device=video.device)
            else:
                tgt_points = init[:, tgt_step]
                tgt_frame = video[:, tgt_step]
                data = {
                    "src_frame": src_frame,
                    "tgt_frame": tgt_frame,
                    "src_points": src_points,
                    "tgt_points": tgt_points
                }
                pred = self.optical_flow_refiner(data, mode="flow_with_tracks_init", **kwargs)
                flow, alpha = pred["flow"], pred["alpha"]
            tracks.append(torch.cat([flow + grid, alpha[..., None]], dim=-1))
        tracks = torch.stack(tracks, dim=1)
        return {"tracks": tracks}

