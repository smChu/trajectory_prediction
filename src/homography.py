"""Use the supplied VIRAT image-to-world homographies without changing scale.

README_homography.txt defines image origin at top left, x right, y down.
It specifies scene-dependent world coordinates but no physical unit (m/ft).
"""

from pathlib import Path
import re
import warnings

import cv2
import numpy as np


def load_virat_homography(source):
    """Return pixel-to-ground H, reference (width, height), and matrix path."""
    # Official Release 2.0: VIRAT_S_XXYYZZ_KK_SSSSSS_TTTTTT; XXYY is the scene.
    match = re.fullmatch(r"VIRAT_S_(\d{4})\d{2}_\d{2}_\d{6}_\d{6}", Path(source).stem)
    if match is None:
        raise ValueError(f"Cannot identify a VIRAT scene from filename: {source}")
    scene = match.group(1)
    directory = Path(__file__).resolve().parents[1] / "Resources/homographies"
    matrix_path = directory / f"VIRAT_{scene}_homography_img2world.txt"
    reference_path = directory / f"VIRAT_S_{scene}.jpg"
    for path in (matrix_path, reference_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    # These files contain three comma-separated rows, without a header.
    matrix = np.loadtxt(matrix_path, delimiter=",", dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError(f"Expected a finite 3x3 homography: {matrix_path}")
    if np.linalg.matrix_rank(matrix) != 3:
        raise ValueError(f"Homography is singular: {matrix_path}")
    reference = cv2.imread(str(reference_path))
    if reference is None:
        raise ValueError(f"Cannot read calibration reference: {reference_path}")
    # The JPG is the original scene reference. The PNG is a resized figure
    # with margins and grid labels, not the pixel coordinate reference.
    height, width = reference.shape[:2]
    # README explicitly says image -> world, so no inverse is used here.
    return matrix, (width, height), matrix_path


def pixel_to_ground(bottom_center, matrix):
    """Apply H @ [u, v, 1], then divide by its homogeneous denominator."""
    point = np.array([*bottom_center, 1.0], dtype=np.float64)
    projected = matrix @ point
    denominator_scale = np.linalg.norm(matrix[2]) * np.linalg.norm(point)
    if not np.isfinite(projected).all() or abs(projected[2]) <= 1e-12 * denominator_scale:
        raise ValueError("Non-finite projection or homogeneous denominator near zero")
    position = projected[:2] / projected[2]
    if not np.isfinite(position).all():
        raise ValueError("Non-finite ground position")
    return float(position[0]), float(position[1])


def ground_observations(tracks, matrix, frame_idx):
    """Return only confirmed tracks updated by a detection in this frame.

    frame_idx is zero-based within the input clip. Bounding boxes are the
    original DeepSORT posterior boxes in original video pixel coordinates.
    No history, prediction, or conversion to an assumed physical unit is added.
    """
    observations = []
    for track in tracks:
        if not track.is_confirmed() or track.time_since_update != 0:
            continue
        x, y, width, height = track.to_tlwh()
        bottom_center = (float(x + width / 2.0), float(y + height))
        try:
            if not np.isfinite([x, y, width, height]).all() or width <= 0 or height <= 0:
                raise ValueError("Invalid track bounding box")
            position = pixel_to_ground(bottom_center, matrix)
        except ValueError as error:
            warnings.warn(f"frame_idx={frame_idx} track_id={track.track_id}: {error}")
            continue
        observations.append({
            "frame_idx": frame_idx,
            "track_id": int(track.track_id),
            "bottom_center_pixel": bottom_center,
            "ground_position": position,
        })
    return observations
