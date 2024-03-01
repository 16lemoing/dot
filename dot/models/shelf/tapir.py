from torch import nn
import torch.nn.functional as F
from einops import rearrange

from .tapir_utils.tapir_model import TAPIR

class Tapir(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model = TAPIR(pyramid_level=args.pyramid_level,
                           softmax_temperature=args.softmax_temperature,
                           extra_convs=args.extra_convs)

    def forward(self, video, queries, backward_tracking, cache_features=False):
        # Preprocess video
        video = video * 2 - 1  # conversion from [0, 1] to [-1, 1]
        video = rearrange(video, "b t c h w -> b t h w c")

        # Preprocess queries
        queries = queries[..., [0, 2, 1]]

        # Inference
        outputs = self.model(video, queries, cache_features=cache_features)
        tracks, occlusions, expected_dist = outputs['tracks'], outputs['occlusion'], outputs['expected_dist']

        # Postprocess tracks
        tracks = rearrange(tracks, "b s t c -> b t s c")

        # Postprocess visibility
        visibles = (1 - F.sigmoid(occlusions)) * (1 - F.sigmoid(expected_dist)) > 0.5
        visibles = rearrange(visibles, "b s t -> b t s")

        return tracks, visibles