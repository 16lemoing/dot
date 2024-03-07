from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        parser.add_argument("--in_track_name", type=str, default="cotracker")
        parser.add_argument("--out_track_name", type=str, default="ground_truth")
        parser.add_argument("--num_in_tracks", type=int, default=2048)
        parser.add_argument("--num_out_tracks", type=int, default=2048)
        parser.add_argument("--batch_size_valid", type=int, default=4)
        parser.add_argument("--train_iter", type=int, default=1000000)
        parser.add_argument("--log_iter", type=int, default=10000)
        parser.add_argument("--log_factor", type=float, default=1.)
        parser.add_argument("--print_iter", type=int, default=100)
        parser.add_argument("--valid_iter", type=int, default=10000)
        parser.add_argument("--num_valid_batches", type=int, default=24)
        parser.add_argument("--save_iter", type=int, default=1000)
        parser.add_argument("--lr", type=float, default=0.0001)
        parser.add_argument("--world_size", type=int, default=1)
        parser.add_argument("--valid_ratio", type=float, default=0.01)
        parser.add_argument("--lambda_motion_loss", type=float, default=1.)
        parser.add_argument("--lambda_visibility_loss", type=float, default=1.)
        parser.add_argument("--optimizer_path", type=str, default=None)
        parser.set_defaults(data_root="datasets/kubric/movi_f", name="train", batch_size=8, refiner_path=None,
                            is_train=True, model="ofr")
        return parser