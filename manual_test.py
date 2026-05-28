"""Standalone end-to-end test of the RF-DETR segmentation Scope pipeline.

Downloads a sample image, runs both render modes on the GPU, prints detections,
and saves annotated output frames (masks overlay + privacy blur).
"""

import io

import numpy as np
import requests
import torch
from PIL import Image
from rfdetr.util.coco_classes import COCO_CLASSES

from scope_rfdetr_segmentation.pipelines.pipeline import RFDetrSegmentationPipeline

URL = "https://media.roboflow.com/notebooks/examples/dog-2.jpeg"

print("torch.cuda.is_available:", torch.cuda.is_available())
print("downloading test image...")
img = Image.open(io.BytesIO(requests.get(URL, timeout=60).content)).convert("RGB")
print("image size:", img.size)

print("loading segmentation pipeline (nano)...")
pipe = RFDetrSegmentationPipeline(model_variant="nano")
print("pipeline device:", pipe.device)

dets = pipe.model.predict(img, threshold=0.5)
print("num detections:", len(dets))
print("has masks:", getattr(dets, "mask", None) is not None)
for i in range(len(dets)):
    cid = int(dets.class_id[i])
    conf = float(dets.confidence[i])
    print(f"  - {COCO_CLASSES.get(cid, cid)}  {conf:.2f}")

frame = torch.from_numpy(np.array(img)).unsqueeze(0)

# masks overlay
out_masks = pipe(video=[frame], confidence_threshold=0.5, render_mode="masks")["video"]
Image.fromarray((out_masks[0].numpy() * 255).astype(np.uint8), "RGB").save(
    "rfdetr_seg_masks.jpg"
)
print("saved rfdetr_seg_masks.jpg", tuple(out_masks.shape))

# privacy blur
out_blur = pipe(
    video=[frame], confidence_threshold=0.5, render_mode="privacy_blur", blur_strength=25
)["video"]
Image.fromarray((out_blur[0].numpy() * 255).astype(np.uint8), "RGB").save(
    "rfdetr_seg_privacy_blur.jpg"
)
print("saved rfdetr_seg_privacy_blur.jpg", tuple(out_blur.shape))
