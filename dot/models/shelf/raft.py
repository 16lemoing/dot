import torch
from torch import nn
import torch.nn.functional as F


from .raft_utils.update import BasicUpdateBlock
from .raft_utils.extractor import BasicEncoder
from .raft_utils.corr import CorrBlock
from .raft_utils.utils import coords_grid


class RAFT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.fnet = BasicEncoder(output_dim=256, norm_fn=args.norm_fnet, dropout=0, patch_size=args.patch_size)
        self.cnet = BasicEncoder(output_dim=256, norm_fn=args.norm_cnet, dropout=0, patch_size=args.patch_size)
        self.update_block = BasicUpdateBlock(hidden_dim=128, patch_size=args.patch_size, refine_alpha=args.refine_alpha)
        self.refine_alpha = args.refine_alpha
        self.patch_size = args.patch_size
        self.num_iter = args.num_iter

    def encode(self, frame):
        frame = frame * 2 - 1
        fmap = self.fnet(frame)
        cmap = self.cnet(frame)
        feats = torch.cat([fmap, cmap], dim=1)
        return feats.float()

    def initialize_feats(self, feats, frame):
        if feats is None:
            feats = self.encode(frame)
        fmap, cmap = feats.split([256, 256], dim=1)
        return fmap, cmap

    def initialize_flow(self, fmap, coarse_flow):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, _, h, w = fmap.shape
        src_pts = coords_grid(N, h, w, device=fmap.device)

        if coarse_flow is not None:
            coarse_flow = coarse_flow.permute(0, 3, 1, 2)
            # coarse_flow = torch.stack([coarse_flow[:, 0] * (w - 1), coarse_flow[:, 1] * (h - 1)], dim=1)
            tgt_pts = src_pts + coarse_flow
        else:
            tgt_pts = src_pts

        return src_pts, tgt_pts

    def initialize_alpha(self, fmap, coarse_alpha):
        N, _, h, w = fmap.shape
        if coarse_alpha is None:
            alpha = torch.ones(N, 1, h, w, device=fmap.device)
        else:
            alpha = coarse_alpha[:, None]
        return alpha.logit(eps=1e-5)

    def postprocess_alpha(self, alpha):
        alpha = alpha[:, 0]
        return alpha.sigmoid()

    def postprocess_flow(self, flow):
        # N, C, H, W = flow.shape
        # flow = torch.stack([flow[:, 0] / (W - 1), flow[:, 1] / (H - 1)], dim=1)
        flow = flow.permute(0, 2, 3, 1)
        return flow

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/P, W/P, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, self.patch_size, self.patch_size, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(self.patch_size * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, self.patch_size * H, self.patch_size * W)

    def upsample_alpha(self, alpha, mask):
        """ Upsample alpha field [H/P, W/P, 1] -> [H, W, 1] using convex combination """
        N, _, H, W = alpha.shape
        mask = mask.view(N, 1, 9, self.patch_size, self.patch_size, H, W)
        mask = torch.softmax(mask, dim=2)

        up_alpha = F.unfold(alpha, [3, 3], padding=1)
        up_alpha = up_alpha.view(N, 1, 9, 1, 1, H, W)

        up_alpha = torch.sum(mask * up_alpha, dim=2)
        up_alpha = up_alpha.permute(0, 1, 4, 2, 5, 3)
        return up_alpha.reshape(N, 1, self.patch_size * H, self.patch_size * W)

    def forward(self, src_frame=None, tgt_frame=None, src_feats=None, tgt_feats=None, coarse_flow=None, coarse_alpha=None,
                is_train=False):
        src_fmap, src_cmap = self.initialize_feats(src_feats, src_frame)
        tgt_fmap, _ = self.initialize_feats(tgt_feats, tgt_frame)

        corr_fn = CorrBlock(src_fmap, tgt_fmap)

        net, inp = torch.split(src_cmap, [128, 128], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        src_pts, tgt_pts = self.initialize_flow(src_fmap, coarse_flow)
        alpha = self.initialize_alpha(src_fmap, coarse_alpha) if self.refine_alpha else None

        flows_up = []
        alphas_up = []
        for itr in range(self.num_iter):
            tgt_pts = tgt_pts.detach()
            if self.refine_alpha:
                alpha = alpha.detach()

            corr = corr_fn(tgt_pts)

            flow = tgt_pts - src_pts
            net, up_mask, delta_flow, up_mask_alpha, delta_alpha = self.update_block(net, inp, corr, flow, alpha)

            # F(t+1) = F(t) + \Delta(t)
            tgt_pts = tgt_pts + delta_flow
            if self.refine_alpha:
                alpha = alpha + delta_alpha

            # upsample predictions
            flow_up = self.upsample_flow(tgt_pts - src_pts, up_mask)
            if self.refine_alpha:
                alpha_up = self.upsample_alpha(alpha, up_mask_alpha)

            if is_train or (itr == self.num_iter - 1):
                flows_up.append(self.postprocess_flow(flow_up))
                if self.refine_alpha:
                    alphas_up.append(self.postprocess_alpha(alpha_up))

        flows_up = torch.stack(flows_up, dim=1)
        alphas_up = torch.stack(alphas_up, dim=1) if self.refine_alpha else None
        if not is_train:
            flows_up = flows_up[:, 0]
            alphas_up = alphas_up[:, 0] if self.refine_alpha else None
        return flows_up, alphas_up
