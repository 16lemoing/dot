/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// clang-format off
#include <torch/extension.h>
// clang-format on
#include "knn/knn.h"
#include "packed_to_padded_tensor/packed_to_padded_tensor.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("packed_to_padded", &PackedToPadded);
  m.def("padded_to_packed", &PaddedToPacked);
#ifdef WITH_CUDA
  m.def("knn_check_version", &KnnCheckVersion);
#endif
  m.def("knn_points_idx", &KNearestNeighborIdx);
  m.def("knn_points_backward", &KNearestNeighborBackward);
}