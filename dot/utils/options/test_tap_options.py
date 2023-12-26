from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        parser.add_argument("--split", type=str, choices=["davis", "rgb_stacking", "kinetics"], default="davis")
        parser.add_argument("--query_mode", type=str, default="first", choices=["first", "strided"])
        parser.add_argument('--plot_indices', type=int, nargs="+", default=[])
        parser.set_defaults(data_root="datasets/tap", name="test_tap", batch_size=1, num_workers=0, num_tracks=8192)
        return parser