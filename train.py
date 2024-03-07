import os
import torch

torch.backends.cudnn.benchmark = True
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
from einops import repeat, rearrange

from dot.data.movi_f_dataset import create_point_tracking_dataset
from dot.utils.io import create_folder
from dot.utils.torch import reduce, to_device
from dot.utils.log import Logger
from dot.models import create_model
from dot.utils.options.train_options import TrainOptions


def checkpoint(model, optimizer, path, name):
    create_folder(path)
    model_path = os.path.join(path, f"{name}.pth")
    optimizer_path = os.path.join(path, f"{name}_optimizer.pth")
    torch.save(model.module.model.state_dict(), model_path)
    torch.save(optimizer.state_dict(), optimizer_path)


def sample(pred, gt):
    B, I, H, W, _ = pred["flow"].shape
    dense = torch.cat([pred["flow"], pred["alpha"][..., None]], dim=-1)
    dense = rearrange(dense, "b i h w c -> b (i c) h w")
    src_pos = gt["out_src_points"][..., :2]
    grid = src_pos[:, None] * 2 - 1
    sparse = torch.nn.functional.grid_sample(dense, grid, mode="nearest", align_corners=True, padding_mode="border")
    sparse = rearrange(sparse, "b (i c) h w -> b i (h w) c", i=I)
    delta_pos, tgt_alpha = sparse[..., :2], sparse[..., 2:]
    delta_pos[..., 0] = delta_pos[..., 0] / (W - 1)
    delta_pos[..., 1] = delta_pos[..., 1] / (H - 1)
    tgt_pos = src_pos[:, None] + delta_pos
    out_tgt_points = torch.cat([tgt_pos, tgt_alpha], dim=-1)
    pred = {
        "flow": pred["flow"][:, -1],
        "alpha": pred["alpha"][:, -1],
        "out_tgt_points": out_tgt_points
    }
    return pred


def step(loader, model, optimizer, logger, global_iter, args):
    if optimizer is not None:
        optimizer.zero_grad()
    loss = torch.tensor(0., requires_grad=True).cuda()

    gt = loader.next()
    gt = to_device(gt, args.rank)
    pred = model(gt, mode="flow_with_tracks_init", **vars(args))
    pred = sample(pred, gt)

    motion_loss = torch.tensor(0., requires_grad=True).cuda()
    visibility_loss = torch.tensor(0., requires_grad=True).cuda()
    gamma = 0.8
    num_iter = pred["out_tgt_points"].size(1)
    for i in range(num_iter):
        weight = gamma ** (num_iter - i - 1)
        pred_pos, pred_vis = pred["out_tgt_points"][:, i][..., :2], pred["out_tgt_points"][:, i][..., 2]
        gt_pos, gt_vis = gt["out_tgt_points"][..., :2], gt["out_tgt_points"][..., 2]
        motion_loss += weight * (gt_pos - pred_pos).abs().mean()
        visibility_loss += weight * torch.nn.functional.binary_cross_entropy(pred_vis, gt_vis)
    loss += motion_loss * args.lambda_motion_loss
    loss += visibility_loss * args.lambda_visibility_loss

    if optimizer is not None:
        loss.backward()
        optimizer.step()

    if args.rank == 0 and logger is not None:
        logger.log_scalar("motion_loss", motion_loss, global_iter)
        logger.log_scalar("visibility_loss", visibility_loss, global_iter)
        logger.log_scalar("loss", loss, global_iter)

        if global_iter % args.print_iter == 0:
            losses = f"Loss: {loss.item():.3E} ("
            losses += f"motion: {motion_loss.item():.3E}, "
            losses += f"visibility: {visibility_loss.item():.3E})"
            epoch = loader.epoch
            print(f"[E{epoch:02d} I{global_iter}/{args.train_iter}] {losses}")

        if global_iter % args.log_iter == 0:
            logger.log_image("pred_flow", pred["flow"], "flow", 2, global_iter)
            logger.log_image("pred_alpha", pred["alpha"], "mask", 2, global_iter)
            logger.log_image("tgt_frame", gt["tgt_frame"], "rgb", 2, global_iter)
            logger.log_image("src_frame", gt["src_frame"], "rgb", 2, global_iter)

    return loss


def setup(rank, world_size, master_port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = master_port
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def main(rank, args):
    print(f"Running DOT on rank {rank + 1} / {args.world_size}.")
    setup(rank, args.world_size, args.master_port)
    args.rank = rank
    logger = Logger(args) if args.rank == 0 else None

    # Prepare data
    train_dataset = create_point_tracking_dataset(
        args,
        batch_size=args.batch_size,
        split="train",
        verbose=args.rank == 0
    )
    if args.rank == 0:
        valid_dataset = create_point_tracking_dataset(
            args,
            batch_size=args.batch_size_valid,
            split="valid",
            verbose=args.rank == 0,
            num_workers=1
        )

    # Load model and optimizer
    model = create_model(args).cuda()
    model = DDP(model, device_ids=[args.rank], output_device=args.rank)
    model.train()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0, 0.99))
    if args.optimizer_path is not None:
        optimizer.load_state_dict(torch.load(args.optimizer_path))

    min_loss = None
    for global_iter in range(args.train_iter + 1):
        step(train_dataset, model, optimizer, logger, global_iter, args)

        if args.rank == 0 and global_iter % args.save_iter == 0:
            checkpoint(model, optimizer, args.checkpoint_path, "last")

        if args.valid_iter > 0 and global_iter % args.valid_iter == 0:
            if args.rank == 0:
                model.eval()

                losses = []
                with torch.no_grad():
                    for _ in range(args.num_valid_batches):
                        losses.append(step(valid_dataset, model.module, None, None, global_iter, args))
                    valid_dataset.reinit()  # Make sure we always use the same validation data

                loss = torch.stack(losses).mean()

                status = ""
                if min_loss is None or loss < min_loss:
                    min_loss = loss
                    checkpoint(model, optimizer, args.checkpoint_path, "best")
                    status = "(best)"
                logger.log_scalar("valid/loss", loss.item(), global_iter)
                print(f"[I{global_iter}/{args.train_iter}] Validation loss: {loss:.3E} {status}")
                model.train()

            dist.barrier()

    cleanup()


if __name__ == "__main__":
    args = TrainOptions().parse_args()
    mp.spawn(main,
             args=(args,),
             nprocs=args.world_size,
             join=True)
    print("Done.")
