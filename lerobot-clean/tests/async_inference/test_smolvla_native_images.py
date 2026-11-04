# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from types import SimpleNamespace

import pytest

pytest.importorskip("grpc")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from lerobot.async_inference.configs import PolicyServerConfig  # noqa: E402
from lerobot.async_inference.helpers import TimedObservation  # noqa: E402
from lerobot.async_inference.policy_server import PolicyServer  # noqa: E402
from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.utils.feature_utils import hw_to_dataset_features  # noqa: E402


@pytest.mark.parametrize(
    "policy_type,padding,expected_shape",
    [
        ("smolvla", (512, 512), (1, 3, 480, 640)),
        ("smolvla", None, (1, 3, 256, 256)),
        ("act", None, (1, 3, 256, 256)),
    ],
)
def test_server_leaves_image_geometry_to_smolvla(policy_type, padding, expected_shape):
    """The actual server prediction path must use the policy's camera preparation strategy."""
    server = PolicyServer(PolicyServerConfig())
    server.policy_type = policy_type
    server.actions_per_chunk = 2
    server.lerobot_features = hw_to_dataset_features(
        {"joint.pos": float, "camera1": (480, 640, 3)}, "observation", use_video=False
    )
    captured = {}

    def predict(batch):
        captured.update(batch)
        return torch.zeros(1, 2, 1)

    server.policy = SimpleNamespace(
        config=SimpleNamespace(
            image_features={
                "observation.images.camera1": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256))
            },
            resize_imgs_with_padding=padding,
        ),
        predict_action_chunk=predict,
    )
    server.preprocessor = lambda batch: batch
    server.postprocessor = lambda actions: actions
    camera = np.zeros((480, 640, 3), dtype=np.uint8)
    camera[100:200, 200:400, 0] = 255
    actions = server._predict_action_chunk(
        TimedObservation(timestamp=0.0, timestep=1, observation={"joint.pos": 12.0, "camera1": camera})
    )

    image = captured["observation.images.camera1"]
    assert tuple(image.shape) == expected_shape
    if padding is not None:
        expected = torch.from_numpy(camera).permute(2, 0, 1).float().div(255).unsqueeze(0)
        torch.testing.assert_close(image, expected, rtol=0, atol=0)
    assert len(actions) == 2
    assert actions[0].get_timestep() == 1
