"""RF-DETR instance segmentation pipeline.

Runs Roboflow's RF-DETR segmentation per frame and returns the frame either
with colored mask overlays ("masks") or with the segmented regions blurred
for privacy ("privacy_blur").

    segment (RF-DETR) -> overlay masks | blur masked regions
"""

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
from PIL import Image, ImageFilter

from scope.core.pipelines.interface import Pipeline, Requirements

from .schema import RFDetrSegConfig

if TYPE_CHECKING:
    from scope.core.pipelines.base_schema import BasePipelineConfig

logger = logging.getLogger(__name__)

_VARIANTS = {
    "preview": "RFDETRSegPreview",
    "nano": "RFDETRSegNano",
    "small": "RFDETRSegSmall",
    "medium": "RFDETRSegMedium",
    "large": "RFDETRSegLarge",
}


class RFDetrSegmentationPipeline(Pipeline):
    """Real-time instance segmentation with RF-DETR."""

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return RFDetrSegConfig

    def __init__(self, device: torch.device | None = None, **kwargs):
        import rfdetr
        import supervision as sv

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        variant: str = kwargs.get("model_variant", "nano")
        model_cls_name = _VARIANTS.get(variant, "RFDETRSegNano")
        model_cls = getattr(rfdetr, model_cls_name)

        logger.info(
            "Loading RF-DETR segmentation model: %s (device=%s)",
            model_cls_name,
            self.device,
        )
        self.model = model_cls()

        self._class_names: dict[int, str] = {}
        try:
            from rfdetr.util.coco_classes import COCO_CLASSES

            self._class_names = dict(COCO_CLASSES)
        except Exception:  # pragma: no cover - depends on rfdetr internals
            logger.warning("Could not import RF-DETR COCO class names; using ids")

        self._sv = sv
        self._label_annotator = sv.LabelAnnotator()

        logger.info("RF-DETR segmentation pipeline ready")

    def prepare(self, **kwargs) -> Requirements:
        return Requirements(input_size=1)

    @torch.no_grad()
    def __call__(self, **kwargs) -> dict:
        video = kwargs.get("video")
        if video is None:
            raise ValueError(
                "Input video cannot be None for RFDetrSegmentationPipeline"
            )

        confidence: float = kwargs.get("confidence_threshold", 0.5)
        render_mode: str = kwargs.get("render_mode", "masks")
        mask_opacity: float = kwargs.get("mask_opacity", 0.5)
        show_labels: bool = kwargs.get("show_labels", True)
        blur_strength: int = kwargs.get("blur_strength", 25)

        mask_annotator = self._sv.MaskAnnotator(opacity=mask_opacity)

        output_frames: list[torch.Tensor] = []

        for frame_tensor in video:
            frame_np = frame_tensor.squeeze(0).cpu().numpy().astype(np.uint8)
            pil_image = Image.fromarray(frame_np, mode="RGB")

            detections = self.model.predict(pil_image, threshold=confidence)

            if render_mode == "privacy_blur":
                annotated = self._privacy_blur(
                    frame_np, pil_image, detections, blur_strength
                )
            else:
                annotated = mask_annotator.annotate(
                    scene=frame_np.copy(), detections=detections
                )
                if show_labels and len(detections) > 0:
                    labels = []
                    for i in range(len(detections)):
                        class_id = int(detections.class_id[i])
                        name = self._class_names.get(class_id, str(class_id))
                        if detections.confidence is not None:
                            labels.append(f"{name} {detections.confidence[i]:.2f}")
                        else:
                            labels.append(name)
                    annotated = self._label_annotator.annotate(
                        scene=annotated, detections=detections, labels=labels
                    )

            frame_out = torch.from_numpy(annotated.astype(np.float32) / 255.0)
            output_frames.append(frame_out)

        return {"video": torch.stack(output_frames, dim=0)}

    @staticmethod
    def _privacy_blur(frame_np, pil_image, detections, blur_strength) -> np.ndarray:
        """Blur only the pixels covered by any segmentation mask."""
        masks = getattr(detections, "mask", None)
        if masks is None or len(detections) == 0:
            return frame_np
        union = np.any(np.asarray(masks).astype(bool), axis=0)  # (H, W)
        if not union.any():
            return frame_np
        blurred = np.array(
            pil_image.filter(ImageFilter.GaussianBlur(radius=int(blur_strength)))
        )
        return np.where(union[..., None], blurred, frame_np).astype(np.uint8)
