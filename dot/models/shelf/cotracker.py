import torch
from torch import nn
from tqdm import tqdm

from .cotracker_utils.predictor import CoTrackerPredictor


class CoTracker(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model = CoTrackerPredictor(args.patch_size, args.wind_size)
        self.local_grid_size = 6
        self.grid_size = 6

    def forward(self, video, queries, backward_tracking, cache_features=False):
        return self.model(video, queries=queries, backward_tracking=backward_tracking, cache_features=cache_features)

    def add_context(self, query, rgbs):
        t = query[0, 0, 0]

        device = rgbs.device
        if self.local_grid_size > 0:
            xy_target = get_points_on_a_grid(
                self.local_grid_size,
                (50, 50),
                [query[0, 0, 2].cpu(), query[0, 0, 1].cpu()],
                device=device,
            )

            xy_target = torch.cat(
                [torch.ones_like(xy_target[:, :, :1]) * t, xy_target], dim=2
            )  #
            query = torch.cat([query, xy_target], dim=1).to(device)  #

        if self.grid_size > 0:
            xy = get_points_on_a_grid(self.grid_size, rgbs.shape[3:], device=device)
            xy = torch.cat([torch.ones_like(xy[:, :, :1]) * t, xy], dim=2).to(device)  #
            query = torch.cat([query, xy], dim=1).to(device)  #
        return query

    def remove_context(self, tracks, visibles):
        return tracks[:, :, 0], visibles[:, :, 0]

    def forward_star(self, video, queries, backward_tracking):
        tracks, visibles = [], []
        for query in tqdm(queries[0], desc="individual points"):
            query = query[None, None]
            queries_i = self.add_context(query, video)
            # tracks_i, visibles_i = self.forward(video, queries_i, backward_tracking)
            tracks_i, visibles_i = self.model._compute_sparse_tracks(video, queries_i, backward_tracking=backward_tracking)
            tracks_i, visibles_i = self.remove_context(tracks_i, visibles_i)
            tracks.append(tracks_i)
            visibles.append(visibles_i)
        tracks = torch.stack(tracks, dim=2)
        visibles = torch.stack(visibles, dim=2)
        return tracks, visibles


def meshgrid2d(B, Y, X, stack=False, norm=False, device="cpu"):
    # returns a meshgrid sized B x Y x X

    grid_y = torch.linspace(0.0, Y - 1, Y, device=torch.device(device))
    grid_y = torch.reshape(grid_y, [1, Y, 1])
    grid_y = grid_y.repeat(B, 1, X)

    grid_x = torch.linspace(0.0, X - 1, X, device=torch.device(device))
    grid_x = torch.reshape(grid_x, [1, 1, X])
    grid_x = grid_x.repeat(B, Y, 1)

    if stack:
        # note we stack in xy order
        # (see https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.grid_sample)
        grid = torch.stack([grid_x, grid_y], dim=-1)
        return grid
    else:
        return grid_y, grid_x


def get_points_on_a_grid(grid_size, interp_shape, grid_center=(0, 0), device="cpu"):
    if grid_size == 1:
        return torch.tensor([interp_shape[1] / 2, interp_shape[0] / 2], device=device)[
            None, None
        ]

    grid_y, grid_x = meshgrid2d(
        1, grid_size, grid_size, stack=False, norm=False, device=device
    )
    step = interp_shape[1] // 64
    if grid_center[0] != 0 or grid_center[1] != 0:
        grid_y = grid_y - grid_size / 2.0
        grid_x = grid_x - grid_size / 2.0
    grid_y = step + grid_y.reshape(1, -1) / float(grid_size - 1) * (
        interp_shape[0] - step * 2
    )
    grid_x = step + grid_x.reshape(1, -1) / float(grid_size - 1) * (
        interp_shape[1] - step * 2
    )

    grid_y = grid_y + grid_center[0]
    grid_x = grid_x + grid_center[1]
    xy = torch.stack([grid_x, grid_y], dim=-1).to(device)
    return xy
