import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import torch
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter

from dot.utils.plot import to_rgb


def detach(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu()
    return tensor


def number(tensor):
    if isinstance(tensor, torch.Tensor) and tensor.isnan().any():
        return torch.zeros_like(tensor)
    return tensor


class Logger():
    def __init__(self, args):
        self.writer = SummaryWriter(args.log_path)
        self.factor = args.log_factor
        self.world_size = args.world_size

    def log_scalar(self, name, scalar, global_iter):
        if scalar is not None:
            if type(scalar) == list:
                for i, x in enumerate(scalar):
                    self.log_scalar(f"{name}_{i}", x, global_iter)
            else:
                self.writer.add_scalar(name, number(detach(scalar)), global_iter)

    def log_scalars(self, name, scalars, global_iter):
        for s in scalars:
            self.log_scalar(f"{name}/{s}", scalars[s], global_iter)

    def log_image(self, name, tensor, mode, nrow, global_iter, pos=None, occ=None):
        tensor = detach(tensor)
        tensor = to_rgb(tensor, mode, pos, occ)
        grid = make_grid(tensor, nrow=nrow, normalize=False, value_range=[0, 1], pad_value=0)
        grid = torch.nn.functional.interpolate(grid[None], scale_factor=self.factor)[0]
        self.writer.add_image(name, grid, global_iter)

    def log_video(self, name, tensor, mode, nrow, global_iter, fps=4, pos=None, occ=None):
        tensor = detach(tensor)
        tensor = to_rgb(tensor, mode, pos, occ, is_video=True)
        grid = []
        for i in range(tensor.shape[1]):
            grid.append(make_grid(tensor[:, i], nrow=nrow, normalize=False, value_range=[0, 1], pad_value=0))
        grid = torch.stack(grid, dim=0)
        grid = torch.nn.functional.interpolate(grid, scale_factor=self.factor)
        grid = grid[None]
        self.writer.add_video(name, grid, global_iter, fps=fps)