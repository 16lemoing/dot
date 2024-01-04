import numpy as np
import torch
import torch.distributed as dist


def reduce(tensor, world_size):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.clone()
        dist.all_reduce(tensor, dist.ReduceOp.SUM)
        tensor.div_(world_size)
    return tensor


def expand(mask, num=1):
    # mask: ... H W
    # -----------------
    # mask: ... H W
    for _ in range(num):
        mask[..., 1:, :] = mask[..., 1:, :] | mask[..., :-1, :]
        mask[..., :-1, :] = mask[..., :-1, :] | mask[..., 1:, :]
        mask[..., :, 1:] = mask[..., :, 1:] | mask[..., :, :-1]
        mask[..., :, :-1] = mask[..., :, :-1] | mask[..., :, 1:]
    return mask


def differentiate(mask):
    # mask: ... H W
    # -----------------
    # diff: ... H W
    diff = torch.zeros_like(mask).bool()
    diff_y = mask[..., 1:, :] != mask[..., :-1, :]
    diff_x = mask[..., :, 1:] != mask[..., :, :-1]
    diff[..., 1:, :] = diff[..., 1:, :] | diff_y
    diff[..., :-1, :] = diff[..., :-1, :] | diff_y
    diff[..., :, 1:] = diff[..., :, 1:] | diff_x
    diff[..., :, :-1] = diff[..., :, :-1] | diff_x
    return diff


def sample_points(step, boundaries, num_samples):
    if boundaries.ndim == 3:
        points = []
        for boundaries_k in boundaries:
            points_k = sample_points(step, boundaries_k, num_samples)
            points.append(points_k)
        points = torch.stack(points)
    else:
        H, W = boundaries.shape
        boundary_points, _ = sample_mask_points(step, boundaries, num_samples // 2)
        num_boundary_points = boundary_points.shape[0]
        num_random_points = num_samples - num_boundary_points
        random_points = sample_random_points(step, H, W, num_random_points)
        random_points = random_points.to(boundary_points.device)
        points = torch.cat((boundary_points, random_points), dim=0)
    return points


def sample_mask_points(step, mask, num_points):
    num_nonzero = int(mask.sum())
    i, j = torch.nonzero(mask, as_tuple=True)
    if num_points < num_nonzero:
        sample = np.random.choice(num_nonzero, size=num_points, replace=False)
        i, j = i[sample], j[sample]
    t = torch.ones_like(i) * step
    x, y = j, i
    points = torch.stack((t, x, y), dim=-1)  # [num_points, 3]
    return points.float(), (i, j)


def sample_random_points(step, height, width, num_points):
    x = torch.randint(width, size=[num_points])
    y = torch.randint(height, size=[num_points])
    t = torch.ones(num_points) * step
    points = torch.stack((t, x, y), dim=-1)  # [num_points, 3]
    return points.float()


def get_grid(height, width, shape=None, dtype="torch", device="cpu", align_corners=True, normalize=True):
    H, W = height, width
    S = shape if shape else []
    if align_corners:
        x = torch.linspace(0, 1, W, device=device)
        y = torch.linspace(0, 1, H, device=device)
        if not normalize:
            x = x * (W - 1)
            y = y * (H - 1)
    else:
        x = torch.linspace(0.5 / W, 1.0 - 0.5 / W, W, device=device)
        y = torch.linspace(0.5 / H, 1.0 - 0.5 / H, H, device=device)
        if not normalize:
            x = x * W
            y = y * H
    x_view, y_view, exp = [1 for _ in S] + [1, -1], [1 for _ in S] + [-1, 1], S + [H, W]
    x = x.view(*x_view).expand(*exp)
    y = y.view(*y_view).expand(*exp)
    grid = torch.stack([x, y], dim=-1)
    if dtype == "numpy":
        grid = grid.numpy()
    return grid


def get_sobel_kernel(kernel_size):
    K = kernel_size
    sobel = torch.tensor(list(range(K))) - K // 2
    sobel_x, sobel_y = sobel.view(-1, 1), sobel.view(1, -1)
    sum_xy = sobel_x ** 2 + sobel_y ** 2
    sum_xy[sum_xy == 0] = 1
    sobel_x, sobel_y = sobel_x / sum_xy, sobel_y / sum_xy
    sobel_kernel = torch.stack([sobel_x.unsqueeze(0), sobel_y.unsqueeze(0)], dim=0)
    return sobel_kernel


def to_device(data, device):
    data = {k: v.to(device) for k, v in data.items()}
    return data


def get_alpha_consistency(bflow, fflow, thresh_1=0.01, thresh_2=0.5, thresh_mul=1):
    norm = lambda x: x.pow(2).sum(dim=-1).sqrt()
    B, H, W, C = bflow.shape

    mag = norm(fflow) + norm(bflow)
    grid = get_grid(H, W, shape=[B], device=fflow.device)
    grid[..., 0] = grid[..., 0] + bflow[..., 0] / (W - 1)
    grid[..., 1] = grid[..., 1] + bflow[..., 1] / (H - 1)
    grid = grid * 2 - 1
    fflow_warped = torch.nn.functional.grid_sample(fflow.permute(0, 3, 1, 2), grid, mode="bilinear", align_corners=True)
    flow_diff = bflow + fflow_warped.permute(0, 2, 3, 1)
    occ_thresh = thresh_1 * mag + thresh_2
    occ_thresh = occ_thresh * thresh_mul
    alpha = norm(flow_diff) < occ_thresh
    alpha = alpha.float()
    return alpha