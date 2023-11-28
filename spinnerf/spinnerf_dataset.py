# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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
SPInNeRF dataset.
"""

from typing import Dict
import torch
import numpy as np
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.utils.data_utils import get_depth_image_from_path


class SPInNeRFDataset(InputDataset):
    """Dataset that returns images and depths. If no depths are found, then we generate them with Zoe Depth.

    Args:
        dataparser_outputs: description of where and how to read input images.
        scale_factor: The scaling factor for the dataparser outputs.
    """

    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0, bg=False, lpips=False):
        super().__init__(dataparser_outputs, scale_factor)
        colmap_depth = np.load(self.metadata["colmap_depth_file"], allow_pickle=True)
        
        depth_indices = []
        depth_values = []
        for i in range(len(colmap_depth)):
            coords = colmap_depth[i]["coord"][:, [1, 0]]
            cameras = i * np.ones_like(coords[:, :1])
            indices = np.concatenate([cameras, coords], 1)
            depth_indices.append(indices)
            depth_values.append(colmap_depth[i]["depth"])
        self.depth_indices = np.concatenate(depth_indices, 0).astype(np.int32)
        self.depth_values = np.concatenate(depth_values, 0).astype(np.float32)
        self.N_points = self.depth_values.shape[0]

        self.depth_filenames = self.metadata["depth_filenames"]
        self.depth_unit_scale_factor = 1.0

        self.bg = bg
        self.lpips = lpips
    
    def get_data(self, image_idx: int) -> Dict:
        """Returns the ImageDataset data as a dictionary.

        Args:
            image_idx: The image index in the dataset.
        """
        data = super().get_data(image_idx)
        if self.bg:
            data["mask"] = (data["mask"] == False)
        elif self.lpips:
            data["mask"] = torch.ones_like(data["mask"]).bool()
        return data
    
    def get_metadata(self, data: Dict) -> Dict:
        if self.depth_filenames is None:
            return {"depth_image": self.depths[data["image_idx"]]}

        filepath = self.depth_filenames[data["image_idx"]]
        height = int(self._dataparser_outputs.cameras.height[data["image_idx"]])
        width = int(self._dataparser_outputs.cameras.width[data["image_idx"]])

        # Scale depth images to meter units and also by scaling applied to cameras
        scale_factor = self.depth_unit_scale_factor * self._dataparser_outputs.dataparser_scale
        depth_image = get_depth_image_from_path(
            filepath=filepath, height=height, width=width, scale_factor=scale_factor
        )

        return {"depth_image": depth_image}