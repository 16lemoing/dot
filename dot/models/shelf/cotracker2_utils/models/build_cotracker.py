# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from dot.models.shelf.cotracker2_utils.models.core.cotracker.cotracker import CoTracker2


def build_cotracker(patch_size, wind_size):
    cotracker = CoTracker2(stride=patch_size, window_len=wind_size, add_space_attn=True)
    return cotracker
