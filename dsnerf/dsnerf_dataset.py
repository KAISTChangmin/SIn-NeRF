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
DSNeRF dataset.
"""

from typing import Dict
import numpy as np
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset


class DSNeRFDataset(InputDataset):
    """Dataset that returns images and depths. If no depths are found, then we generate them with Zoe Depth.

    Args:
        dataparser_outputs: description of where and how to read input images.
        scale_factor: The scaling factor for the dataparser outputs.
    """

    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0):
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

    def get_metadata(self, data: Dict) -> Dict:
        depth_indices = self.depth_indices
        depth_values = self.depth_values

        return {"depth_indices": depth_indices, "depth_values": depth_values}