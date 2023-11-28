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
SPIn-NeRF Datamanager.
"""

from __future__ import annotations

from functools import cached_property
from dataclasses import dataclass, field
from typing import Dict, Tuple, Type, Union, Literal
from pathlib import Path
from spinnerf.spinnerf_dataset import SPInNeRFDataset

from rich.progress import Console

import torch
import numpy as np

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
    variable_res_collate
)
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.pixel_samplers import (
    PixelSampler,
    PatchPixelSamplerConfig,
)
from nerfstudio.data.utils.dataloaders import (
    CacheDataloader,
)
from nerfstudio.model_components.ray_generators import RayGenerator

CONSOLE = Console(width=120)

@dataclass
class SPInNeRFDataManagerConfig(VanillaDataManagerConfig):
    """Configuration for the SPInNeRFDataManager."""

    _target: Type = field(default_factory=lambda: SPInNeRFDataManager)
    """Specifies the pixel sampler used to sample pixels from images."""
    lpips_patch_size = 32

class SPInNeRFDataManager(VanillaDataManager):
    """Data manager for SPInNeRF."""

    config: SPInNeRFDataManagerConfig


    def __init__(
        self,
        config: VanillaDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,
    ):
        self.config = config
        self.device = device
        self.world_size = world_size
        self.local_rank = local_rank
        self.sampler = None
        self.test_mode = test_mode
        self.test_split = "test" if test_mode in ["test", "inference"] else "val"
        self.dataparser_config = self.config.dataparser
        if self.config.data is not None:
            self.config.dataparser.data = Path(self.config.data)
        else:
            self.config.data = self.config.dataparser.data
        self.dataparser = self.dataparser_config.setup()
        if test_mode == "inference":
            self.dataparser.downscale_factor = 1  # Avoid opening images
        self.includes_time = self.dataparser.includes_time
        self.train_dataparser_outputs: DataparserOutputs = self.dataparser.get_dataparser_outputs(split="train")

        self.train_dataset = self.create_train_dataset()
        self.train_dataset_bg = self.create_train_dataset(bg=True)
        self.train_dataset_lpips = self.create_train_dataset(lpips=True)
        self.eval_dataset = self.create_eval_dataset()
        self.exclude_batch_keys_from_device = self.train_dataset.exclude_batch_keys_from_device
        if self.config.masks_on_gpu is True:
            self.exclude_batch_keys_from_device.remove("mask")
        if self.config.images_on_gpu is True:
            self.exclude_batch_keys_from_device.remove("image")

        if self.train_dataparser_outputs is not None:
            cameras = self.train_dataparser_outputs.cameras
            if len(cameras) > 1:
                for i in range(1, len(cameras)):
                    if cameras[0].width != cameras[i].width or cameras[0].height != cameras[i].height:
                        CONSOLE.print("Variable resolution, using variable_res_collate")
                        self.config.collate_fn = variable_res_collate
                        break
        super().__init__(config, device, test_mode, world_size, local_rank, **kwargs)

    @cached_property
    def dataset_type(self):
        return SPInNeRFDataset
    
    def create_train_dataset(self, bg=False, lpips=False):
        """Sets up the data loaders for training"""
        return self.dataset_type(
            dataparser_outputs=self.train_dataparser_outputs,
            scale_factor=self.config.camera_res_scale_factor,
            bg = bg,
            lpips = lpips,
        )
    
    def _get_pixel_sampler(self, dataset, num_rays_per_batch: int, lpips=False) -> PixelSampler:
        """Infer pixel sampler to use."""
        if lpips:
            return PatchPixelSamplerConfig().setup(
                patch_size=self.config.lpips_patch_size, num_rays_per_batch=num_rays_per_batch
            )
        return super()._get_pixel_sampler(dataset, num_rays_per_batch)

    def setup_train(self):
        """Sets up the data loaders for training"""
        assert self.train_dataset is not None
        CONSOLE.print("Setting up training dataset...")
        self.train_image_dataloader = CacheDataloader(
            self.train_dataset,
            num_images_to_sample_from=self.config.train_num_images_to_sample_from,
            num_times_to_repeat_images=self.config.train_num_times_to_repeat_images,
            device=self.device,
            num_workers=self.world_size * 4,
            pin_memory=True,
            collate_fn=self.config.collate_fn,
            exclude_batch_keys_from_device=self.exclude_batch_keys_from_device,
        )
        self.iter_train_image_dataloader = iter(self.train_image_dataloader)
        self.train_pixel_sampler = self._get_pixel_sampler(self.train_dataset, self.config.train_num_rays_per_batch)
        self.train_ray_generator = RayGenerator(self.train_dataset.cameras.to(self.device))

        assert self.train_dataset_bg is not None
        CONSOLE.print("Setting up training dataset (bg)...")
        self.train_image_dataloader_bg = CacheDataloader(
            self.train_dataset_bg,
            num_images_to_sample_from=self.config.train_num_images_to_sample_from,
            num_times_to_repeat_images=self.config.train_num_times_to_repeat_images,
            device=self.device,
            num_workers=self.world_size * 4,
            pin_memory=True,
            collate_fn=self.config.collate_fn,
            exclude_batch_keys_from_device=self.exclude_batch_keys_from_device,
        )
        self.iter_train_image_dataloader_bg = iter(self.train_image_dataloader_bg)

        assert self.train_dataset_lpips is not None
        CONSOLE.print("Setting up training dataset (lpips)...")
        self.train_image_dataloader_lpips = CacheDataloader(
            self.train_dataset_lpips,
            num_images_to_sample_from=self.config.train_num_images_to_sample_from,
            num_times_to_repeat_images=self.config.train_num_times_to_repeat_images,
            device=self.device,
            num_workers=self.world_size * 4,
            pin_memory=True,
            collate_fn=self.config.collate_fn,
            exclude_batch_keys_from_device=self.exclude_batch_keys_from_device,
        )
        self.iter_train_image_dataloader_lpips = iter(self.train_image_dataloader_lpips)
        self.train_pixel_sampler_lpips = self._get_pixel_sampler(self.train_dataset_lpips, self.config.train_num_rays_per_batch)

    def next_train(self, step: int) -> Tuple[RayBundle, Dict, RayBundle, Dict, RayBundle, Dict, RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        image_batch = next(self.iter_train_image_dataloader)
        assert self.train_pixel_sampler is not None
        assert isinstance(image_batch, dict)
        batch = self.train_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.train_ray_generator(ray_indices)

        image_batch_bg = next(self.iter_train_image_dataloader_bg)
        batch_bg = self.train_pixel_sampler.sample(image_batch_bg)
        ray_indices_bg = batch_bg["indices"]
        ray_bundle_bg = self.train_ray_generator(ray_indices_bg)
        
        image_batch_lpips = next(self.iter_train_image_dataloader_lpips)
        batch_lpips = self.train_pixel_sampler_lpips.sample(image_batch_lpips)
        ray_indices_lpips = batch_lpips["indices"]
        ray_bundle_lpips = self.train_ray_generator(ray_indices_lpips)

        N_samples = ray_indices.shape[0]
        depth_batch = self.get_depth_batch(N_samples)
        depth_ray_indices = depth_batch["depth_indices"]
        depth_ray_bundle = self.train_ray_generator(depth_ray_indices)

        return ray_bundle, batch, depth_ray_bundle, depth_batch, ray_bundle_bg, batch_bg, ray_bundle_lpips, batch_lpips
    
    def get_depth_batch(self, N_samples):
        point_indices = (np.random.rand(N_samples) * self.train_dataset.N_points).astype(np.int32)
        depth_indices = self.train_dataset.depth_indices[point_indices]
        depth_values = self.train_dataset.depth_values[point_indices][..., None]
        depth_batch = {
            "depth_indices": torch.IntTensor(depth_indices).to(self.device),
            "depth_values": torch.FloatTensor(depth_values).to(self.device),
        }
        return depth_batch