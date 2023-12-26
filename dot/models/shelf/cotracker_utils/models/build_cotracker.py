# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from dot.models.shelf.cotracker_utils.models.core.cotracker.cotracker import CoTracker


def build_cotracker(
    patch_size: int,
    wind_size: int,
):
    if patch_size == 4 and wind_size == 8:
        return build_cotracker_stride_4_wind_8()
    elif patch_size == 4 and wind_size == 12:
        return build_cotracker_stride_4_wind_12()
    elif patch_size == 8 and wind_size == 16:
        return build_cotracker_stride_8_wind_16()
    else:
        raise ValueError(f"Unknown model for patch size {patch_size} and window size {window_size}")


# model used to produce the results in the paper
def build_cotracker_stride_4_wind_8(checkpoint=None):
    return _build_cotracker(
        stride=4,
        sequence_len=8,
        checkpoint=checkpoint,
    )


def build_cotracker_stride_4_wind_12(checkpoint=None):
    return _build_cotracker(
        stride=4,
        sequence_len=12,
        checkpoint=checkpoint,
    )


# the fastest model
def build_cotracker_stride_8_wind_16(checkpoint=None):
    return _build_cotracker(
        stride=8,
        sequence_len=16,
        checkpoint=checkpoint,
    )


def _build_cotracker(
    stride,
    sequence_len,
    checkpoint=None,
):
    cotracker = CoTracker(
        stride=stride,
        S=sequence_len,
        add_space_attn=True,
        space_depth=6,
        time_depth=6,
    )
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f, map_location="cpu")
            if "model" in state_dict:
                state_dict = state_dict["model"]
        cotracker.load_state_dict(state_dict)
    return cotracker
