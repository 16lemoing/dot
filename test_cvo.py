import torch
from tqdm import tqdm
import os.path as osp
import numpy as np

from dot.utils.options.test_cvo_options import TestOptions
from dot.utils.io import create_folder, write_video, write_frame
from dot.utils.plot import to_rgb, plot_points
from dot.utils.torch import to_device
from dot.utils.metrics import cvo_metrics, save_metrics
from dot.models import create_model
from dot.data.cvo_dataset import create_optical_flow_dataset


def main(args):
    model = create_model(args).cuda()
    dataset = create_optical_flow_dataset(args)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    metrics = []
    for index, gt in tqdm(enumerate(dataset), total=len(dataset)):
        if len(args.plot_indices) > 0 and index not in args.plot_indices:
            continue

        if args.filter and index in args.filter_indices and args.split != "extended":
            continue

        if args.num_workers > 0 and index % args.num_workers != args.worker_idx:
            continue

        gt = to_device(gt, "cuda")

        with torch.no_grad():
            start.record()
            pred = model(gt, mode="flow_from_last_to_first_frame", **vars(args))
            end.record()
            torch.cuda.synchronize()
            time = start.elapsed_time(end) / 1000

        metrics.append(cvo_metrics.compute_metrics(gt, pred, time))

        if len(args.plot_indices) > 0:
            write_video(gt["video"][0], osp.join(args.result_path, f"{index}/video.mp4"))
            write_frame(to_rgb(pred["flow"], "flow")[0], osp.join(args.result_path, f"{index}/pred_flow.png"))
            write_frame(to_rgb(gt["flow"], "flow")[0], osp.join(args.result_path, f"{index}/gt_flow.png"))
            write_frame(to_rgb(pred["alpha"], "mask")[0], osp.join(args.result_path, f"{index}/pred_alpha.png"))
            write_frame(to_rgb(gt["alpha"], "mask")[0], osp.join(args.result_path, f"{index}/gt_alpha.png"))
            if "src_points" in pred and "tgt_points" in pred:
                src_frame = gt["video"][0, -1]
                tgt_frame = gt["video"][0, 0]
                src_points = pred["src_points"][0]
                tgt_points = pred["tgt_points"][0]
                points_path = osp.join(args.result_path, f"{index}/points.png")
                plot_points(src_frame, tgt_frame, src_points, tgt_points, points_path)
            if "coarse_flow" in pred:
                coarse_flow_rgb = to_rgb(pred["coarse_flow"], "flow")[0]
                write_frame(coarse_flow_rgb, osp.join(args.result_path, f"{index}/coarse_flow.png"))
            if "coarse_alpha" in pred:
                coarse_alpha_rgb = to_rgb(pred["coarse_alpha"], "mask")[0]
                write_frame(coarse_alpha_rgb, osp.join(args.result_path, f"{index}/coarse_alpha.png"))

    if len(args.plot_indices) == 0:
        metrics = {k: [m[k] for m in metrics] for k in metrics[0]}
        avg_metrics = {k: np.mean(v) for k, v in metrics.items()}

        create_folder(args.log_path)
        save_path = osp.join(args.log_path, f"cvo_metrics_{args.split}_{args.model}_{args.worker_idx}.txt")
        save_metrics(metrics, save_path)
        print("AVG %s: " % args.model)
        print(" ".join(f"{k}:{v:.4f}" for k, v in avg_metrics.items()))

if __name__ == "__main__":
    args = TestOptions().parse_args()
    main(args)
    print("Done.")