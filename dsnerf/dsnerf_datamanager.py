# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Instruct-NeRF2NeRF Datamanager.
"""

from __future__ import annotations

from functools import cached_property
from dataclasses import dataclass, field
from typing import Dict, Tuple, Type
from dsnerf.dsnerf_dataset import DSNeRFDataset

from rich.progress import Console

import torch

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)

CONSOLE = Console(width=120)

@dataclass
class DSNeRFDataManagerConfig(VanillaDataManagerConfig):
    """Configuration for the DSNeRFDataManager."""

    _target: Type = field(default_factory=lambda: DSNeRFDataManager)
    """Specifies the pixel sampler used to sample pixels from images."""

class DSNeRFDataManager(VanillaDataManager):
    """Data manager for DSNeRF."""

    config: DSNeRFDataManagerConfig

    @cached_property
    def dataset_type(self):
        return DSNeRFDataset

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        batch = next(self.iter_train_image_dataloader)
        assert self.train_pixel_sampler is not None
        image_batch = {k: v for k, v in batch.items() if k in ["image_idx", "image"]}
        depth_batch = {k: v[0] for k, v in batch.items() if k in ["depth_indices", "depth_values"]}
        image_batch = self.train_pixel_sampler.sample(image_batch)
        image_ray_indices = image_batch["indices"]
        image_ray_bundle = self.train_ray_generator(image_ray_indices)

        N_points = depth_batch["depth_values"].shape[0]
        N_samples = image_ray_indices.shape[0]
        point_indices = (torch.rand(N_samples) * N_points).long()
        depth_ray_indices = depth_batch["depth_indices"][point_indices]
        depth_ray_bundle = self.train_ray_generator(depth_ray_indices)

        return image_ray_bundle, image_batch, depth_ray_bundle, depth_batch
