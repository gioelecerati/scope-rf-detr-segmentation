"""Scope plugin hook implementation for RF-DETR segmentation."""

from scope.core.plugins.hookspecs import hookimpl


@hookimpl
def register_pipelines(register):
    from .pipelines.pipeline import RFDetrSegmentationPipeline

    register(RFDetrSegmentationPipeline)
