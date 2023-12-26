import warnings
import torch

try:
    from dot.utils import torch3d
except ModuleNotFoundError:
    torch3d = None

if torch3d:
    TORCH3D_AVAILABLE = True
else:
    TORCH3D_AVAILABLE = False


def interpolate(src_points, tgt_points, grid, version="torch3d"):
    B, S, _ = src_points.shape
    H, W, _ = grid.shape

    # For each point in a regular grid, find indices of nearest visible source point
    grid = grid.view(1, H * W, 2).expand(B, -1, -1)  # B HW 2
    src_pos, src_alpha = src_points[..., :2], src_points[..., 2]
    if version == "torch" or (version == "torch3d" and not TORCH3D_AVAILABLE):
        if version == "torch3d":
            warnings.warn(
                "Torch3D is not available. For optimal speed and memory consumption, consider setting it up.",
                stacklevel=2,
            )
        dis = (grid ** 2).sum(-1)[:, None] + (src_pos ** 2).sum(-1)[:, :, None] - 2 * src_pos @ grid.permute(0, 2, 1)
        dis[src_alpha == 0] = float('inf')
        _, idx = dis.min(dim=1)
        idx = idx.view(B, H * W, 1)
    elif version == "torch3d":
        src_pos_packed = src_pos[src_alpha.bool()]
        tgt_points_packed = tgt_points[src_alpha.bool()]
        lengths = src_alpha.sum(dim=1).long()
        max_length = int(lengths.max())
        cum_lengths = lengths.cumsum(dim=0)
        cum_lengths = torch.cat([torch.zeros_like(cum_lengths[:1]), cum_lengths[:-1]])
        src_pos = torch3d.packed_to_padded(src_pos_packed, cum_lengths, max_length)
        tgt_points = torch3d.packed_to_padded(tgt_points_packed, cum_lengths, max_length)
        _, idx, _ = torch3d.knn_points(grid, src_pos, lengths2=lengths, return_nn=False)
        idx = idx.view(B, H * W, 1)

    # Use correspondences between source and target points to initialize the flow
    tgt_pos, tgt_alpha = tgt_points[..., :2], tgt_points[..., 2]
    flow = tgt_pos - src_pos
    flow = torch.cat([flow, tgt_alpha[..., None]], dim=-1)  # B S 3
    flow = flow.gather(dim=1, index=idx.expand(-1, -1, flow.size(-1)))
    flow = flow.view(B, H, W, -1)
    flow, alpha = flow[..., :2], flow[..., 2]
    flow[..., 0] = flow[..., 0] * (W - 1)
    flow[..., 1] = flow[..., 1] * (H - 1)
    return flow, alpha