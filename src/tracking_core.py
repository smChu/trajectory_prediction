"""Shared YOLO/MARS bridge and drawing for the original nwojke DeepSORT."""

import colorsys
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# Use the checked-out original tracker rather than an installed substitute.
sys.path.insert(0, str(ROOT / "deep_sort"))

# Load torchvision (and PyTorch's Triton) before TensorFlow 2.21.0 to avoid
# the LLVM symbol clash when YOLO first imports torchvision.
import torchvision
import tensorflow as tf
from ultralytics import YOLO

from deep_sort.detection import Detection
from deep_sort.nn_matching import NearestNeighborDistanceMetric
from deep_sort.tracker import Tracker
from tools.generate_detections import create_box_encoder, extract_image_patch


def create_mars_encoder(model_path):
    """Use the original frozen-graph encoder in an isolated TF1-style graph."""
    with tf.Graph().as_default() as graph:
        encoder = create_box_encoder(
            str(model_path), input_name="net/images", output_name="net/features"
        )
        input_shape = graph.get_tensor_by_name("net/images:0").shape.as_list()
        output_shape = graph.get_tensor_by_name("net/features:0").shape.as_list()
        if output_shape != [None, 128]:
            raise ValueError(f"Expected MARS features [None, 128], got {output_shape}")
    return encoder, input_shape[1:3]


def make_detections(frame, result, encoder, patch_shape):
    """Convert YOLO xyxy boxes into original DeepSORT Detection objects."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []

    tlwh = boxes.xyxy.cpu().numpy().astype(np.float64, copy=True)
    tlwh[:, 2:] -= tlwh[:, :2]
    confidences = boxes.conf.cpu().numpy()

    # The original encoder substitutes a random image for an invalid crop.
    # Reject those boxes first so every feature comes from an actual crop.
    valid = [
        i for i, box in enumerate(tlwh)
        if np.isfinite(box).all()
        and np.isfinite(confidences[i])
        and np.all(box[2:] > 0)
        and extract_image_patch(frame, box, patch_shape) is not None
    ]
    if not valid:
        return []
    tlwh = tlwh[valid]
    confidences = confidences[valid]
    features = encoder(frame, tlwh)
    if features.shape != (len(tlwh), 128) or not np.isfinite(features).all():
        raise RuntimeError("MARS did not return finite 128-dimensional features")

    return [
        Detection(box, confidence, feature)
        for box, confidence, feature in zip(tlwh, confidences, features)
    ]


def track_color(track_id):
    """Return a stable, bright BGR color without changing tracking state."""
    hue = (int(track_id) * 0.618033988749895) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.75, 1.0)
    return tuple(round(channel * 255) for channel in (blue, green, red))


def draw_tracks(frame, tracker):
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        x1, y1, x2, y2 = np.rint(track.to_tlbr()).astype(int)
        color = track_color(track.track_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        cv2.putText(
            frame, f"ID {track.track_id}", (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
        )
