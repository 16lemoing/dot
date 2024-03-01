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
        self.name = self.point_tracker.name + "_" + self.optical_flow_refiner.name
        self.resolution = [height, width]

    def forward(self, data, mode, **kwargs):
        if mode == "flow_from_last_to_first_frame":
            return self.get_flow_from_last_to_first_frame(data, **kwargs)
        elif mode == "tracks_for_queries":
            return self.get_tracks_for_queries(data, **kwargs)
        elif mode == "tracks_from_first_to_every_other_frame":
            return self.get_tracks_from_first_to_every_other_frame(data, **kwargs)
        elif mode == "tracks_from_every_cell_in_every_frame":
            return self.get_tracks_from_every_cell_in_every_frame(data, **kwargs)
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
        H, W = self.resolution

        init = self.point_tracker(data, mode="tracks_at_motion_boundaries", **kwargs)["tracks"]
        init = torch.stack([init[..., 0] / (w - 1), init[..., 1] / (h - 1), init[..., 2]], dim=-1)

        if h != H or w != W:
            video = video.reshape(B * T, C, h, w)
            video = F.interpolate(video, size=(H, W), mode="bilinear")
            video = video.reshape(B, T, C, H, W)

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
            for b in range(B):
                cur = query_points[b, :, 0] == src_step
                if torch.any(cur):
                    cur_points = query_points[b, cur]
                    cur_x = cur_points[..., 2] / (w - 1)
                    cur_y = cur_points[..., 1] / (h - 1)
                    cur_tracks = dense_to_sparse_tracks(cur_x, cur_y, tracks_from_src[b], h, w)
                    tracks[b, :, cur] = cur_tracks
        return {"tracks": tracks}

    def get_tracks_from_first_to_every_other_frame(self, data, **kwargs):
        video = data["video"]
        B, T, C, h, w = video.shape
        H, W = self.resolution

        if h != H or w != W:
            video = video.reshape(B * T, C, h, w)
            video = F.interpolate(video, size=(H, W), mode="bilinear")
            video = video.reshape(B, T, C, H, W)

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

    def get_tracks_from_every_cell_in_every_frame(self, data, cell_size=1, cell_time_steps=20, **kwargs):
        video = data["video"]
        B, T, C, h, w = video.shape
        H, W = self.resolution
        ch, cw, ct = h // cell_size, w // cell_size, min(T, cell_time_steps)

        if h != H or w != W:
            video = video.reshape(B * T, C, h, w)
            video = F.interpolate(video, size=(H, W), mode="bilinear")
            video = video.reshape(B, T, C, H, W)

        init = self.point_tracker(data, mode="tracks_at_motion_boundaries", **kwargs)["tracks"]
        init = torch.stack([init[..., 0] / (w - 1), init[..., 1] / (h - 1), init[..., 2]], dim=-1)

        feats = self.optical_flow_refiner({"video": video}, mode="feats", **kwargs)["feats"]

        grid = get_grid(H, W, device=video.device)
        visited_cells = torch.zeros(B, T, ch, cw, device=video.device)
        src_steps = torch.linspace(0, T - 1, T // ct).long()
        tracks = [[] for _ in range(B)]
        for k, src_step in enumerate(tqdm(src_steps, desc="Refine source step", leave=False)):
            if visited_cells[:, src_step].all():
                continue
            src_points = init[:, src_step]
            src_feats = feats[:, src_step]
            tracks_from_src = []
            for tgt_step in tqdm(range(T), desc="Refine target step", leave=False):
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
            for b in range(B):
                src_cell = visited_cells[b, src_step]
                if src_cell.all():
                    continue
                cur_y, cur_x = (1 - src_cell).nonzero(as_tuple=True)
                cur_x = (cur_x + 0.5) / cw
                cur_y = (cur_y + 0.5) / ch
                cur_tracks = dense_to_sparse_tracks(cur_x, cur_y, tracks_from_src[b], h, w)
                visited_cells[b] = update_visited(visited_cells[b], cur_tracks, h, w, ch, cw)
                tracks[b].append(cur_tracks)
        tracks = [torch.cat(t, dim=1) for t in tracks]
        return {"tracks": tracks}

def dense_to_sparse_tracks(x, y, tracks, height, width):
    h, w = height, width
    T = tracks.size(0)
    grid = torch.stack([x, y], dim=-1) * 2 - 1
    grid = repeat(grid, "s c -> t s r c", t=T, r=1)
    tracks = rearrange(tracks, "t h w c -> t c h w")
    tracks = F.grid_sample(tracks, grid, align_corners=True, mode="bilinear")
    tracks = rearrange(tracks[..., 0], "t c s -> t s c")
    tracks[..., 0] = tracks[..., 0] * (w - 1)
    tracks[..., 1] = tracks[..., 1] * (h - 1)
    tracks[..., 2] = (tracks[..., 2] > 0).float()
    return tracks

def update_visited(visited_cells, tracks, height, width, cell_height, cell_width):
    T = tracks.size(0)
    h, w = height, width
    ch, cw = cell_height, cell_width
    for tgt_step in range(T):
        tgt_points = tracks[tgt_step]
        tgt_vis = tgt_points[:, 2]
        visited = tgt_points[tgt_vis.bool()]
        if len(visited) > 0:
            visited_x, visited_y = visited[:, 0], visited[:, 1]
            visited_x = (visited_x / (w - 1) * cw).floor().long()
            visited_y = (visited_y / (h - 1) * ch).floor().long()
            valid = (visited_x >= 0) & (visited_x < cw) & (visited_y >= 0) & (visited_y < ch)
            visited_x = visited_x[valid]
            visited_y = visited_y[valid]
            tgt_cell = visited_cells[tgt_step].view(-1)
            tgt_cell[visited_y * cw + visited_x] = 1.
            tgt_cell = tgt_cell.view_as(visited_cells[tgt_step])
            visited_cells[tgt_step] = tgt_cell
    return visited_cells