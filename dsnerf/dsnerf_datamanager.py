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
DS-NeRF Datamanager.
"""

from __future__ import annotations

from functools import cached_property
from dataclasses import dataclass, field
from typing import Dict, Tuple, Type
from dsnerf.dsnerf_dataset import DSNeRFDataset

from rich.progress import Console

import torch
import numpy as np

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

    def next_train(self, step: int) -> Tuple[RayBundle, Dict, RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        image_batch = next(self.iter_train_image_dataloader)
        assert self.train_pixel_sampler is not None
        assert isinstance(image_batch, dict)
        batch = self.train_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.train_ray_generator(ray_indices)

        N_samples = ray_indices.shape[0]
        depth_batch = self.get_depth_batch(N_samples)
        depth_ray_indices = depth_batch["depth_indices"]
        depth_ray_bundle = self.train_ray_generator(depth_ray_indices)

        return ray_bundle, batch, depth_ray_bundle, depth_batch
    
    def get_depth_batch(self, N_samples):
        point_indices = (np.random.rand(N_samples) * self.train_dataset.N_points).astype(np.int32)
        depth_indices = self.train_dataset.depth_indices[point_indices]
        depth_values = self.train_dataset.depth_values[point_indices]
        depth_batch = {
            "depth_indices": torch.IntTensor(depth_indices).to(self.device),
            "depth_values": torch.FloatTensor(depth_values).to(self.device),
        }
        return depth_batch