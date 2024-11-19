from torch import nn

from .cotracker3_utils.predictor import CoTrackerPredictor


class CoTracker3(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model = CoTrackerPredictor(
            window_len=args.wind_size,
            offline=True,
            v2=False
        )

    def forward(self, video, queries, backward_tracking, cache_features=False):
        return self.model(video, queries=queries, backward_tracking=backward_tracking)
