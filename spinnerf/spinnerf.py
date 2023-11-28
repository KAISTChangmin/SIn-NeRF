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
Model for SPInNeRF
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Type
import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.model_components.losses import (
    MSELoss,
    orientation_loss,
    pred_normal_loss,
    scale_gradients_by_distance_squared,
)

@dataclass
class SPInNeRFModelConfig(NerfactoModelConfig):
    """Configuration for the SPInNeRFModel."""
    _target: Type = field(default_factory=lambda: SPInNeRFModel)
    depth_loss_mult: float = 0.05
    disp_loss_mult: float = 1.0
    lpips_loss_mult: float = 0.01
    """Multiplier for losses."""
    lpips_patch_size: int = 32
    """Patch size to use for LPIPS loss."""

class SPInNeRFModel(NerfactoModel):
    """Model for SPInNeRF."""

    config: SPInNeRFModelConfig

    def populate_modules(self):
        super().populate_modules()
        self.depth_loss = MSELoss()
        self.disp_loss = MSELoss()
        self.lpips = LearnedPerceptualImagePatchSimilarity()
    
    def get_outputs(self, ray_bundle: RayBundle, detach_weight=False):
        # apply the camera optimizer pose tweaks
        if self.training:
            self.camera_optimizer.apply_to_raybundle(ray_bundle)
        ray_samples: RaySamples
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        field_outputs = self.field.forward(ray_samples, compute_normals=self.config.predict_normals)
        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)

        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        expected_depth = self.renderer_expected_depth(weights=weights, ray_samples=ray_samples)
        disp = 1. / torch.max(1e-10 * torch.ones_like(expected_depth), expected_depth)

        if detach_weight:
            for i, weight in enumerate(weights_list):
                weights_list[i] = weight.detach()
            weights = weights.detach()
                
        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        with torch.no_grad():
            depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "expected_depth": expected_depth,
            "disp": disp,
        }

        if self.config.predict_normals:
            normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
            pred_normals = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)
            outputs["normals"] = self.normals_shader(normals)
            outputs["pred_normals"] = self.normals_shader(pred_normals)
        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        if self.training and self.config.predict_normals:
            outputs["rendered_orientation_loss"] = orientation_loss(
                weights.detach(), field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions
            )

            outputs["rendered_pred_normal_loss"] = pred_normal_loss(
                weights.detach(),
                field_outputs[FieldHeadNames.NORMALS].detach(),
                field_outputs[FieldHeadNames.PRED_NORMALS],
            )

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])
        return outputs

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = super().get_metrics_dict(outputs, batch)
        if self.training:
            pred_disp = outputs["disp"]
            gt_disp = batch["depth_image"].to(self.device)
            metrics_dict["disp_loss"] = self.disp_loss(pred_disp, gt_disp)

            if "depth_pred" in outputs:
                pred_depth = outputs["depth_pred"]
                gt_depth = batch["depth_values"].to(self.device)
                metrics_dict["depth_loss"] = self.depth_loss(pred_depth, gt_depth)

            if "rgb_lpips" in outputs:
                pred_rgb_lpips = outputs["rgb_lpips"]
                gt_rgb_lpips = batch["image_lpips"].to(self.device)
                out_patches = (pred_rgb_lpips.view(-1, self.config.lpips_patch_size,self.config.lpips_patch_size, 3).permute(0, 3, 1, 2) * 2 - 1).clamp(-1, 1)
                gt_patches = (gt_rgb_lpips.view(-1, self.config.lpips_patch_size,self.config.lpips_patch_size, 3).permute(0, 3, 1, 2) * 2 - 1).clamp(-1, 1)
                metrics_dict["lpips_loss"] = self.lpips(out_patches, gt_patches)

        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)

        if self.training:
            loss_dict["disp_loss"] = self.config.disp_loss_mult * metrics_dict["disp_loss"]
            
            if "depth_loss" in metrics_dict:
                loss_dict["depth_loss"] = self.config.depth_loss_mult * metrics_dict["depth_loss"]
            
            if "lpips_loss" in metrics_dict:
                loss_dict["lpips_loss"] = self.config.lpips_loss_mult * metrics_dict["lpips_loss"]

        return loss_dict
    
    def forward(self, ray_bundle: RayBundle, detach_weight=False):
        """Run forward starting with a ray bundle. This outputs different things depending on the configuration
        of the model and whether or not the batch is provided (whether or not we are training basically)

        Args:
            ray_bundle: containing all the information needed to render that ray latents included
        """

        if self.collider is not None:
            ray_bundle = self.collider(ray_bundle)

        return self.get_outputs(ray_bundle, detach_weight=detach_weight)