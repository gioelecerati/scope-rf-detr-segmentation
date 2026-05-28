"""Configuration schema for the RF-DETR segmentation pipeline."""

from typing import Annotated, ClassVar, Literal

from pydantic import Field

from scope.core.pipelines.base_schema import (
    BasePipelineConfig,
    ModeDefaults,
    ui_field_config,
)


class RFDetrSegConfig(BasePipelineConfig):
    """Configuration for the RF-DETR instance segmentation pipeline.

    Real-time instance segmentation with Roboflow's RF-DETR. Predicts
    per-object masks for the 80 COCO classes and either overlays the
    colored masks ("masks" mode) or blurs the segmented regions for
    privacy ("privacy_blur" mode).
    """

    pipeline_id: ClassVar[str] = "rfdetr-segmentation"
    pipeline_name: ClassVar[str] = "RF-DETR Segmentation"
    pipeline_description: ClassVar[str] = (
        "Real-time instance segmentation using Roboflow RF-DETR. "
        "Overlays per-object masks, or blurs the segmented regions for privacy."
    )
    supports_prompts: ClassVar[bool] = False

    modes: ClassVar[dict[str, ModeDefaults]] = {
        "video": ModeDefaults(default=True),
    }

    # ── Load-time parameters ──────────────────────────────────────

    model_variant: Literal[
        "preview", "nano", "small", "medium", "large"
    ] = Field(
        default="nano",
        description=(
            "RF-DETR segmentation model size. Nano is fastest, Large is most "
            "accurate. All listed variants are Apache-2.0."
        ),
        json_schema_extra=ui_field_config(
            order=1,
            label="Model",
            is_load_param=True,
        ),
    )

    # ── Detection ─────────────────────────────────────────────────

    confidence_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.5,
        description="Minimum confidence score for detections.",
        json_schema_extra=ui_field_config(
            order=10,
            label="Confidence Threshold",
        ),
    )

    # ── Rendering ─────────────────────────────────────────────────

    render_mode: Literal["masks", "privacy_blur"] = Field(
        default="masks",
        description=(
            "How to render results. 'masks' overlays colored segmentation "
            "masks; 'privacy_blur' Gaussian-blurs the segmented regions and "
            "leaves the rest of the frame sharp."
        ),
        json_schema_extra=ui_field_config(
            order=11,
            label="Render Mode",
        ),
    )

    mask_opacity: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.5,
        description="Opacity of the mask overlay ('masks' mode).",
        json_schema_extra=ui_field_config(
            order=12,
            label="Mask Opacity",
        ),
    )

    show_labels: bool = Field(
        default=True,
        description="Show class labels with confidence ('masks' mode).",
        json_schema_extra=ui_field_config(
            order=13,
            label="Show Labels",
        ),
    )

    blur_strength: Annotated[int, Field(ge=1, le=100)] = Field(
        default=25,
        description="Gaussian blur radius for segmented regions ('privacy_blur' mode).",
        json_schema_extra=ui_field_config(
            order=14,
            label="Blur Strength",
        ),
    )
