"""Ground-space 95% covariance ellipses and original-video forecast overlays.

An ellipse boundary is transformed point by point, never by interpreting its
ground-space axes as pixels. Projective poles and invalid covariance are
reported and skipped rather than replaced with invented drawable geometry.
"""

from copy import deepcopy
from functools import lru_cache
from pathlib import Path
import warnings

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.estimate_r import inside_image, iou_matrix, unambiguous_matches


CHI2_95_2D = 5.991
CURRENT_COLOR = (255, 255, 0)
FUTURE_COLOR = (0, 255, 255)
ELLIPSE_COLOR = (255, 0, 255)
GT_COLOR = (0, 128, 255)  # Orange: original annotation bottom centers.


@lru_cache(maxsize=1)
def _trajectory_legend(predictor="ekf"):
    """Render trajectory labels once, then reuse pixels."""
    fonts = ("/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
             "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    font_path = next((path for path in fonts if Path(path).is_file()), None)
    font = ImageFont.truetype(font_path, 20) if font_path else ImageFont.load_default()
    labels = ("predicted trajectory", "GT trajectory")
    colors = (FUTURE_COLOR, GT_COLOR)
    width = 66 + max(int(font.getlength(label)) for label in labels)
    panel = Image.new("RGB", (width, 8 + 32 * len(labels)), (20, 20, 20))
    draw = ImageDraw.Draw(panel)
    for row, (label, color) in enumerate(zip(labels, colors)):
        y = 20 + row * 32
        draw.line((12, y, 44, y), fill=color[::-1], width=2)
        draw.text((54, y - 12), label, font=font, fill=(255, 255, 255))
    return np.asarray(panel)[:, :, ::-1].copy()


def draw_trajectory_legend(frame, predictor="ekf"):
    panel = _trajectory_legend(predictor)
    height = min(panel.shape[0], max(0, frame.shape[0] - 12))
    width = min(panel.shape[1], max(0, frame.shape[1] - 12))
    frame[12:12 + height, 12:12 + width] = panel[:height, :width]


def draw_performance_metrics(frame, metrics, processing_fps=None, method="ekf", y=92):
    """Show cumulative, completed-horizon GT metrics beneath the legend."""
    count = metrics["samples"]
    lines = [f"{method.upper()} / GT: cumulative (12 frames)",
             (f"ADE {metrics['ade']:.4f}   FDE {metrics['fde']:.4f}"
              if count else "ADE --   FDE --  (waiting for GT)"),
             f"Windows: {count} | scene units"]
    lines.append(f"Processing FPS (avg): {processing_fps:.2f}"
                 if processing_fps is not None else "Processing FPS: warming up")
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1
    width = max(cv2.getTextSize(line, font, scale, thickness)[0][0] for line in lines) + 20
    x = 12
    panel_height = 7 + len(lines) * 23
    right, bottom = min(frame.shape[1], x + width), min(frame.shape[0], y + panel_height)
    if right <= x or bottom <= y:
        return
    panel = np.full((panel_height, width, 3), 20, dtype=np.uint8)
    for row, line in enumerate(lines):
        cv2.putText(panel, line, (10, 19 + row * 23), font, scale,
                    (255, 255, 255), thickness, cv2.LINE_AA)
    frame[y:bottom, x:right] = panel[:bottom-y, :right-x]


class GroundTruthAssociation:
    """Confirm mutual-best, unambiguous IoU matches for three consecutive frames.

    Only current detection-matched tracks participate. Strict matches are
    used for metrics; a separate history can maintain display continuity.
    Neither mapping renames tracker IDs or enters EKF measurements.
    """

    def __init__(self):
        self.previous = {}
        self.display_ids = {}
        self.current = {}

    def step(self, frame_idx, tracks, targets, image_size):
        alive = {int(t.track_id) for t in tracks}
        self.display_ids = {tid: gid for tid, gid in self.display_ids.items() if tid in alive}
        tracks = [t for t in tracks if t.is_confirmed() and t.time_since_update == 0
                  and inside_image(t.to_tlwh(), image_size)]
        targets = [(gid, box) for gid, box in targets if inside_image(box, image_size)]
        current = {}
        associations = {}
        for i, j, _ in unambiguous_matches([t.to_tlwh() for t in tracks],
                                          [box for _, box in targets]):
            tid, gid = int(tracks[i].track_id), int(targets[j][0])
            # A confident competing identity invalidates display history
            # immediately, even while its own three-frame streak is pending.
            if tid in self.display_ids and self.display_ids[tid] != gid:
                del self.display_ids[tid]
            previous = self.previous.get(tid)
            count = previous[2] + 1 if previous and previous[:2] == (gid, frame_idx - 1) else 1
            current[tid] = (gid, frame_idx, count)
            if count >= 3:
                associations[tid] = gid
        self.previous = current
        self.display_ids.update(associations)
        self.current = associations
        return associations

    def for_display(self, tracks, targets):
        """Reuse a confirmed identity while its live track still overlaps GT.

        No synthetic GT or nearest-person fallback. Deleted/new tracker IDs
        do not inherit a mapping; a GT currently owned by another strict match
        cannot be borrowed. This relaxed history is never used for metrics.
        """
        boxes = dict(targets)
        owners = {gid: tid for tid, gid in self.current.items()}
        result = {}
        for track in tracks:
            tid = int(track.track_id)
            gid = self.display_ids.get(tid)
            box = boxes.get(gid)
            if not track.is_confirmed() or box is None or owners.get(gid, tid) != tid:
                continue
            if iou_matrix([track.to_tlwh()], [box])[0, 0] > 0:
                result[tid] = gid
        return result


class ForecastComparison:
    """Hold each evaluation forecast and its GT trail through the target frame.

    Live EKF predictions are supplied every frame but never overwrite an
    unexpired evaluation snapshot. The yellow overlay uses these same fixed
    snapshots, alongside their accumulating GT trails and evaluation sampling.
    A verified GT ID is locked only at snapshot creation.
    A missing/deleted ID leaves its old snapshot visible only until expiry;
    its locked GT path can continue independently of tracker ID switches.
    gt_id is the strict start-time match for evaluation. display_gt_id may
    additionally use confirmed history or a later verified display match;
    it never changes gt_id or retroactively makes a window evaluable.
    """

    def __init__(self):
        self.snapshots = {}
        self.frame_idx = None

    def step(self, frame_idx, forecasts, gt_targets, associations, display_associations=None):
        if self.frame_idx is not None and frame_idx <= self.frame_idx:
            raise ValueError("Comparison frame indices must increase")
        self.frame_idx = frame_idx
        self.snapshots = {
            tid: snapshot for tid, snapshot in self.snapshots.items()
            if frame_idx <= snapshot["end_frame_idx"]
        }
        for forecast in forecasts:
            tid = forecast["track_id"]
            if tid in self.snapshots:
                continue
            if forecast["frame_idx"] != frame_idx:
                raise ValueError("New comparison must use this frame's forecast")
            predictions = forecast["predictions"]
            if [p["frame_idx"] for p in predictions] != list(range(frame_idx + 1, frame_idx + 13)):
                raise ValueError("Comparison requires the next 12 consecutive frames")
            snapshot = deepcopy(forecast)
            snapshot["end_frame_idx"] = predictions[-1]["frame_idx"]
            snapshot["gt_id"] = associations.get(tid)
            snapshot["display_gt_id"] = snapshot["gt_id"]
            if snapshot["display_gt_id"] is None and display_associations is not None:
                snapshot["display_gt_id"] = display_associations.get(tid)
            snapshot["display_gt_source"] = (
                "start_match" if snapshot["gt_id"] is not None else
                "confirmed_history" if snapshot["display_gt_id"] is not None else "unmatched")
            snapshot["gt_path"] = []
            self.snapshots[tid] = snapshot
        targets = dict(gt_targets)
        for snapshot in self.snapshots.values():
            if snapshot["display_gt_id"] is None and display_associations is not None:
                gid = display_associations.get(snapshot["track_id"])
                if gid is not None:
                    snapshot["display_gt_id"] = gid
                    snapshot["display_gt_source"] = "confirmed_after_start"
            box = targets.get(snapshot["display_gt_id"])
            if box is None:
                continue
            x, y, width, height = box
            pixel = np.array([x + width / 2, y + height], dtype=float)
            if not np.isfinite(pixel).all() or min(width, height) <= 0:
                raise ValueError("Comparison requires a finite, positive-size GT box")
            snapshot["gt_path"].append({
                "frame_idx": frame_idx, "bottom_center_pixel": pixel.tolist(),
            })
        return list(self.snapshots.values())


def draw_gt_paths(frame, snapshots, thickness=1):
    """Draw annotated pixel bottom centers; never interpolate missing GT frames."""
    counts = {"gt_points": 0, "gt_segments": 0}
    limit = np.iinfo(np.int32).max - 1
    for snapshot in snapshots:
        previous = None
        for observation in snapshot.get("gt_path", []):
            pixel = np.asarray(observation["bottom_center_pixel"], dtype=float)
            if not np.isfinite(pixel).all() or np.any(np.abs(pixel) > limit):
                previous = None
                continue
            if previous is not None and observation["frame_idx"] == previous["frame_idx"] + 1:
                segment = _opencv_points(np.array([previous["bottom_center_pixel"], pixel]))
                cv2.polylines(frame, [segment], False, GT_COLOR, thickness, cv2.LINE_AA)
                counts["gt_segments"] += 1
            cv2.circle(frame, tuple(_opencv_points(pixel[None])[0, 0]),
                       1, GT_COLOR, -1, cv2.LINE_AA)
            counts["gt_points"] += 1
            previous = observation
    return counts


def uncertainty_ellipse(position, covariance, points=48):
    """Return major/minor semiaxes, orientation and a sampled ground contour."""
    if not 32 <= points <= 64:
        raise ValueError("Ellipse boundary needs 32..64 points")
    center = np.asarray(position, dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)
    if center.shape != (2,) or covariance.shape != (2, 2):
        raise ValueError("Expected a 2D center and 2x2 position covariance")
    if not np.isfinite(center).all() or not np.isfinite(covariance).all():
        raise ValueError("Non-finite ellipse center/covariance")
    scale = max(np.max(np.abs(covariance)), np.finfo(float).tiny)
    tolerance = 64 * np.finfo(float).eps * scale
    if np.max(np.abs(covariance - covariance.T)) > tolerance:
        raise ValueError("Asymmetric position covariance")
    covariance = (covariance + covariance.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    tolerance = 64 * np.finfo(float).eps * max(np.max(np.abs(eigenvalues)), np.finfo(float).tiny)
    if eigenvalues[0] < -tolerance:
        raise ValueError(f"Negative position-covariance eigenvalue: {eigenvalues[0]}")
    # Only negative values within floating-point roundoff tolerance are zeroed.
    eigenvalues = np.maximum(eigenvalues[::-1], 0.0)
    eigenvectors = eigenvectors[:, ::-1]
    axes = np.sqrt(CHI2_95_2D * eigenvalues)
    basis = eigenvectors * axes
    angles = np.linspace(0, 2 * np.pi, points, endpoint=False)
    boundary = center + (basis @ np.array([np.cos(angles), np.sin(angles)])).T
    if not np.isfinite(boundary).all():
        raise ValueError("Non-finite ellipse boundary")
    return {
        "center": center, "semiaxes": axes, "basis": basis,
        "angle_rad": float(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])),
        "boundary": boundary,
    }


def ground_to_pixels(points, inverse_homography):
    """Project an Nx2 array, returning pixels, validity and denominators.

    Invalid points remain NaN. Denominator signs are retained so paths are
    never joined across a projective pole. No pixel scaling/clamping is used.
    """
    points = np.asarray(points, dtype=np.float64)
    matrix = np.asarray(inverse_homography, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or matrix.shape != (3, 3):
        raise ValueError("Expected Nx2 ground points and 3x3 inverse homography")
    if not np.isfinite(matrix).all():
        raise ValueError("Non-finite inverse homography")
    homogeneous = np.column_stack((points, np.ones(len(points))))
    projected = homogeneous @ matrix.T
    weights = projected[:, 2]
    scale = np.linalg.norm(matrix[2]) * np.linalg.norm(homogeneous, axis=1)
    valid = np.isfinite(projected).all(axis=1) & (np.abs(weights) > 1e-12 * scale)
    pixels = np.full((len(points), 2), np.nan)
    np.divide(projected[:, :2], weights[:, None], out=pixels, where=valid[:, None])
    valid &= np.isfinite(pixels).all(axis=1)
    return pixels, valid, weights


def project_ellipse(ellipse, inverse_homography):
    """Return a finite closed contour only if it does not cross infinity."""
    matrix = np.asarray(inverse_homography, dtype=float)
    center_weight = float(matrix[2] @ np.r_[ellipse["center"], 1.0])
    amplitude = np.linalg.norm(matrix[2, :2] @ ellipse["basis"])
    scale = np.linalg.norm(matrix[2]) * np.linalg.norm(np.r_[ellipse["center"], 1.0])
    if abs(center_weight) <= amplitude + 1e-12 * scale:
        raise ValueError("Ellipse intersects the inverse-homography pole; no finite closed contour")
    pixels, valid, _ = ground_to_pixels(ellipse["boundary"], matrix)
    if not valid.all():
        raise ValueError("Invalid inverse projection on ellipse boundary")
    return pixels


def _opencv_points(pixels):
    limit = np.iinfo(np.int32).max - 1
    if not np.isfinite(pixels).all() or np.any(np.abs(pixels) > limit):
        raise ValueError("Projected geometry exceeds OpenCV integer-coordinate range")
    return np.rint(pixels).astype(np.int32).reshape(-1, 1, 2)


def collect_forecasts(ekfs, tracks):
    """Forecast ready confirmed EKFs, including a single missed detection frame."""
    forecasts = []
    for track in tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        tid = int(track.track_id)
        ekf = ekfs.filters.get(tid)
        if ekf is None:
            continue
        try:
            predictions = ekf.predict_future()
        except (ValueError, np.linalg.LinAlgError) as error:
            warnings.warn(f"Forecast frame_idx={ekf.frame_idx} track_id={tid}: {error}")
            continue
        if predictions:
            forecasts.append({
                "frame_idx": ekf.frame_idx, "track_id": tid,
                "has_measurement": ekf.last_measurement_frame == ekf.frame_idx,
                "current_position": ekf.x[:2].tolist(), "predictions": predictions,
            })
    return forecasts


def draw_forecasts(frame, forecasts, inverse_homography, ellipse_points=48, thickness=1,
                   color=FUTURE_COLOR, label=""):
    """Draw 12 real covariance contours and forecast points per ready track.

    Return per-frame counts for terminal verification; saved video contains the
    same overlay. OpenCV clips offscreen contours to the image viewport.
    """
    if not 32 <= ellipse_points <= 64 or thickness < 1:
        raise ValueError("Use 32..64 ellipse points and a positive line thickness")
    counts = {"tracks": len(forecasts), "predictions": 0, "ellipses": 0, "skipped_ellipses": 0}
    for forecast in forecasts:
        context = f"frame_idx={forecast['frame_idx']} track_id={forecast['track_id']}"
        predictions = forecast["predictions"]
        counts["predictions"] += len(predictions)
        # Draw uncertainty first so paths and centers remain visible on top.
        for prediction in predictions:
            if "predicted_covariance" not in prediction:
                continue
            try:
                ellipse = uncertainty_ellipse(
                    prediction["predicted_position"],
                    np.asarray(prediction["predicted_covariance"])[:2, :2], ellipse_points,
                )
                contour = _opencv_points(project_ellipse(ellipse, inverse_homography))
                #cv2.polylines(frame, [contour], True, ELLIPSE_COLOR, thickness, cv2.LINE_AA)
                #counts["ellipses"] += 1
            except (ValueError, np.linalg.LinAlgError) as error:
                counts["skipped_ellipses"] += 1
                warnings.warn(f"Ellipse {context} offset={prediction['offset']}: {error}")
        positions = [forecast["current_position"], *[p["predicted_position"] for p in predictions]]
        pixels, valid, weights = ground_to_pixels(positions, inverse_homography)
        valid &= (np.abs(pixels) <= np.iinfo(np.int32).max - 1).all(axis=1)
        if not valid.all():
            warnings.warn(f"Forecast {context}: some projected positions cannot be drawn")
        for i in range(1, len(pixels)):
            if valid[i - 1] and valid[i] and np.sign(weights[i - 1]) == np.sign(weights[i]):
                segment = _opencv_points(pixels[i - 1:i + 1])
                cv2.polylines(frame, [segment], False, color, thickness, cv2.LINE_AA)
            if valid[i]:
                point = tuple(_opencv_points(pixels[i:i + 1])[0, 0])
                cv2.circle(frame, point, 1, color, -1, cv2.LINE_AA)
        # Current-position marker: uncomment the following three lines to show it.
        # if valid[0]:
        #     point = tuple(_opencv_points(pixels[:1])[0, 0])
        #     cv2.circle(frame, point, 2, CURRENT_COLOR, -1, cv2.LINE_AA)
        if label and valid[-1]:
            x, y = _opencv_points(pixels[-1:])[0, 0]
            if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                cv2.putText(frame, label, (int(x) + 4, int(y) - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return counts
