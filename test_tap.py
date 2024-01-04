import torch
from tqdm import tqdm
import numpy as np
import os.path as osp

from dot.utils.options.test_tap_options import TestOptions
from dot.data.tap_dataset import create_point_tracking_dataset
from dot.utils.metrics import tap_metrics, save_metrics
from dot.utils.torch import to_device
from dot.models import create_model
from dot.utils.io import create_folder, write_video
from dot.utils.plot import to_rgb


def main(args):
    model = create_model(args).cuda()
    dataset = create_point_tracking_dataset(args)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    metrics = []
    num_plot = len(args.plot_indices)
    plot_count = 0
    for index, gt in tqdm(enumerate(dataset), total=len(dataset), desc="Test video"):
        if num_plot > 0 and index not in args.plot_indices:
            continue

        if args.num_workers > 0 and index % args.num_workers != args.worker_idx:
            continue

        gt = to_device(gt, "cuda")

        with torch.no_grad():
            start.record()
            pred = model(gt, mode="tracks_for_queries", **vars(args))
            end.record()
            torch.cuda.synchronize()
            time = start.elapsed_time(end) / 1000

        metrics.append(tap_metrics.compute_metrics(gt, pred, time, args.query_mode))

        if num_plot > 0:
            plot_count += 1
            gt_video = to_rgb(gt["video"][0], "rgb", tracks=gt["tracks"][0], is_video=True)
            write_video(gt_video, osp.join(args.result_path, f"{index}/gt"))
            pred_video = to_rgb(gt["video"][0], "rgb", tracks=pred["tracks"][0], is_video=True)
            write_video(pred_video, osp.join(args.result_path, f"{index}/pred"))
            if plot_count == num_plot:
                break

    if num_plot == 0:
        metrics = {k: [m[k] for m in metrics] for k in metrics[0]}
        avg_metrics = {k: np.mean(v) for k, v in metrics.items()}

        create_folder(args.log_path)
        save_path = osp.join(args.log_path, f"tap_metrics_{args.split}_{args.model}_{args.worker_idx}.txt")
        save_metrics(metrics, save_path)
        print("AVG %s: " % args.model)
        print(" ".join(f"{k}:{v:.4f}" for k, v in avg_metrics.items()))


if __name__ == "__main__":
    args = TestOptions().parse_args()
    main(args)
    print("Done.")
