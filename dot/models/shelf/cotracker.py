from torch import nn

from .cotracker_utils.predictor import CoTrackerPredictor


class CoTracker(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model = CoTrackerPredictor(args.patch_size, args.wind_size)

    def forward(self, video, queries, backward_tracking, cache_features=False):
        return self.model(video, queries=queries, backward_tracking=backward_tracking, cache_features=cache_features)
