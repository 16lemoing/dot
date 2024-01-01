import torch


def compute_metrics(gt, pred, time):
    epe_all, epe_occ, epe_vis = get_epe(pred["flow"], gt["flow"], gt["alpha"])
    iou = get_iou(gt["alpha"], pred["alpha"])
    metrics = {
        "epe_all": epe_all.cpu().numpy(),
        "epe_occ": epe_occ.cpu().numpy(),
        "epe_vis": epe_vis.cpu().numpy(),
        "iou": iou.cpu().numpy(),
        "time": time
    }
    return metrics


def get_epe(pred, label, vis):
    diff = torch.norm(pred - label, p=2, dim=-1, keepdim=True)
    epe_all = torch.mean(diff, dim=(1, 2, 3))
    vis = vis[..., None]
    epe_occ = torch.sum(diff * (1 - vis), dim=(1, 2, 3)) / torch.sum((1 - vis), dim=(1, 2, 3))
    epe_vis = torch.sum((diff * vis), dim=(1, 2, 3)) / torch.sum(vis, dim=(1, 2, 3))
    return epe_all, epe_occ, epe_vis


def get_iou(vis1, vis2):
    occ1 = (1 - vis1).bool()
    occ2 = (1 - vis2).bool()
    intersection = (occ1 & occ2).float().sum(dim=[1, 2])
    union = (occ1 | occ2).float().sum(dim=[1, 2])
    iou = intersection / union
    return iou