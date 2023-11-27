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

from dataclasses import dataclass, field
from typing import Type
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from nerfstudio.utils import profiler

from dsnerf.dsnerf_datamanager import (
    DSNeRFDataManagerConfig,
)

@dataclass
class DSNeRFPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: DSNeRFPipeline)
    """target class to instantiate"""
    datamanager: DSNeRFDataManagerConfig = DSNeRFDataManagerConfig()
    """specifies the datamanager config"""

class DSNeRFPipeline(VanillaPipeline):
    """DSNeRF pipeline"""

    config: DSNeRFPipelineConfig

    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict. This will be responsible for
        getting the next batch of data from the DataManager and interfacing with the
        Model class, feeding the data to the model's forward function.

        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """
        ray_bundle, batch, depth_ray_bundle, depth_batch = self.datamanager.next_train(step)

        model_outputs = self._model(ray_bundle)
        depth_model_outputs = self._model(depth_ray_bundle)

        model_outputs["depth_pred"] = depth_model_outputs["expected_depth"]
        batch["depth_values"] = depth_batch["depth_values"]

        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)

        return model_outputs, loss_dict, metrics_dict

        