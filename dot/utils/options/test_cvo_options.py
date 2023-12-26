from .base_options import BaseOptions, str2bool


class TestOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        parser.add_argument("--split", type=str, choices=["clean", "final", "extended"], default="clean")
        parser.add_argument("--filter", type=str2bool, nargs='?', const=True, default=True)
        parser.add_argument('--filter_indices', type=int, nargs="+",
                            default=[70, 77, 93, 96, 140, 143, 162, 172, 174, 179, 187, 215, 236, 284, 285, 293, 330,
                                     358, 368, 402, 415, 458, 483, 495, 534])
        parser.add_argument('--plot_indices', type=int, nargs="+", default=[])
        parser.set_defaults(data_root="datasets/kubric/cvo", name="test_cvo", batch_size=1, num_workers=0,
                            sample_mode="last")
        return parser