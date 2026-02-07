"""Depth estimation module using MiDaS."""

from models.depth.midas import estimate_depth, get_object_depth, enrich_detections_with_depth

__all__ = ["estimate_depth", "get_object_depth", "enrich_detections_with_depth"]
