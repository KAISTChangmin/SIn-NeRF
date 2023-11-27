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
Model for DSNeRF
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Type

from nerfstudio.model_components.losses import MSELoss
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig

@dataclass
class DSNeRFModelConfig(NerfactoModelConfig):
    """Configuration for the DSNeRFModel."""
    _target: Type = field(default_factory=lambda: DSNeRFModel)
    depth_loss_mult: float = 0.1
    """Multiplier for Depth loss."""

class DSNeRFModel(NerfactoModel):
    """Model for DSNeRF."""

    config: DSNeRFModelConfig

    def populate_modules(self):
        super().populate_modules()
        self.depth_loss = MSELoss()

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = super().get_metrics_dict(outputs, batch)
        if self.training:
            pred_depth = outputs["expected_depth"]
            gt_depth = batch["depth_values"].to(self.device)
            metrics_dict["depth_loss"] = self.depth_loss(pred_depth, gt_depth)
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        if self.training:
            loss_dict["depth_loss"] = self.config.depth_loss_mult * metrics_dict["depth_loss"]
        return loss_dict
