"""Ground-space 95% covariance ellipses and original-video forecast overlays.

An ellipse boundary is transformed point by point, never by interpreting its
ground-space axes as pixels. Projective poles and invalid covariance are
reported and skipped rather than replaced with invented drawable geometry.
"""

import warnings

import cv2
import numpy as np


CHI2_95_2D = 5.991
CURRENT_COLOR = (255, 255, 0)
FUTURE_COLOR = (0, 255, 255)
ELLIPSE_COLOR = (255, 0, 255)


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
    """Use ready, confirmed tracks under the existing display-age condition.

    On a one-frame detection miss the maintained EKF prior can still forecast;
    has_measurement explicitly identifies this case. Older missed tracks keep
    their live EKF lifecycle but have no future overlay. Deleted tracks cannot
    enter through the current original-Tracker track list.
    """
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


def draw_forecasts(frame, forecasts, inverse_homography, ellipse_points=48, thickness=1):
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
            try:
                ellipse = uncertainty_ellipse(
                    prediction["predicted_position"],
                    np.asarray(prediction["predicted_covariance"])[:2, :2], ellipse_points,
                )
                contour = _opencv_points(project_ellipse(ellipse, inverse_homography))
                cv2.polylines(frame, [contour], True, ELLIPSE_COLOR, thickness, cv2.LINE_AA)
                counts["ellipses"] += 1
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
                cv2.polylines(frame, [segment], False, FUTURE_COLOR, thickness, cv2.LINE_AA)
            if valid[i]:
                point = tuple(_opencv_points(pixels[i:i + 1])[0, 0])
                cv2.circle(frame, point, 2, FUTURE_COLOR, -1, cv2.LINE_AA)
        # Current-position marker: uncomment the following three lines to show it.
        # if valid[0]:
        #     point = tuple(_opencv_points(pixels[:1])[0, 0])
        #     cv2.circle(frame, point, 4, CURRENT_COLOR, -1, cv2.LINE_AA)
        if valid[-1]:
            x, y = _opencv_points(pixels[-1:])[0, 0]
            if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                cv2.putText(frame, f"ID {forecast['track_id']}", (int(x) + 4, int(y) - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, FUTURE_COLOR, 1, cv2.LINE_AA)
    return counts
