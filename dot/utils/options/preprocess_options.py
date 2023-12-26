from .base_options import BaseOptions, str2bool


class PreprocessOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        parser.add_argument("--extract_movi_f", type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument("--save_tracks", type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--download_path', type=str, default="gs://kubric-public/tfds")
        parser.add_argument('--num_videos', type=int, default=11000)
        parser.set_defaults(data_root="datasets/kubric/movi_f", name="preprocess", num_workers=2, num_tracks=2048,
                            model="pt")
        return parser