"""SO101 robot arm policy transforms for openpi.

SO101 is a 5-DOF + gripper single arm robot (6 total action dims).
Camera setup: front (third-person) + wrist.
"""
import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    """Parse image to uint8 HWC format."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class SO101Inputs(transforms.DataTransformFn):
    """Convert SO101 observations to pi0.5 model input format.

    State: 6-dim (5 joints + gripper)
    Images: front (base_0_rgb) + wrist (left_wrist_0_rgb)
    """
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation.images.front"])
        wrist_image = _parse_image(data["observation.images.wrist"])

        inputs = {
            "state": data["observation.state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        if "action" in data:
            inputs["actions"] = data["action"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class SO101Outputs(transforms.DataTransformFn):
    """Extract SO101 action dims from padded model output."""

    def __call__(self, data: dict) -> dict:
        # SO101 has 6 action dims (5 joints + gripper)
        return {"actions": np.asarray(data["actions"][:, :6])}
