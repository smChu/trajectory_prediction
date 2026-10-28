"""Calibrate diagonal EKF R from a development VIRAT clip and original GT.

GT is used only by this offline tool, never as an EKF measurement. No DIVA,
interpolation, tracker-ID equality assumption, or residual-based trimming.
"""

import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np

from homography import ground_observations, load_virat_homography, pixel_to_ground


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else _hash_stream(stream)


def _hash_stream(stream):
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def configuration(source, model, mars_model, imgsz):
    _, size, matrix_path = load_virat_homography(source)
    return {
        "homography_sha256": sha256(matrix_path), "image_size": list(size),
        "model_sha256": sha256(model), "mars_sha256": sha256(mars_model),
        "imgsz": int(imgsz), "person_class": 0, "confidence": 0.25,
        "tracker": {"n_init": 3, "max_age": 30, "max_iou_distance": 0.7,
                    "max_cosine_distance": 0.2},
    }


def default_calibration_path(source, model, imgsz):
    _, _, matrix_path = load_virat_homography(source)
    scene = matrix_path.stem.split("_")[1]
    return ROOT / "runs/calibration" / f"ekf_r_{scene}_{Path(model).stem}_imgsz{imgsz}.json"


def load_r_calibration(path, source, model, mars_model, imgsz):
    if not path.is_file():
        raise FileNotFoundError(
            f"R calibration missing: {path}. Run src/estimate_r.py on a development "
            "clip first, or explicitly supply provisional --ekf-r VAR_X VAR_Y."
        )
    data = json.loads(path.read_text())
    if data.get("schema") != 1 or data.get("configuration") != configuration(source, model, mars_model, imgsz):
        raise ValueError("R calibration does not match the homography, model, image size or tracker settings; recalibrate")
    r = np.asarray(data["r_diagonal"], dtype=float)
    if r.shape != (2,) or not np.isfinite(r).all() or np.any(r <= 0):
        raise ValueError("Calibration must contain two positive finite R variances")
    return r, data


def load_gt(source, frame_count):
    path = ROOT / "Resources/annotations" / f"{Path(source).stem}.viratdata.objects.txt"
    rows = np.loadtxt(path, dtype=np.int64, ndmin=2)
    if rows.shape[1] != 8:
        raise ValueError(f"Expected 8 columns in {path}, got {rows.shape}")
    people = rows[rows[:, 7] == 1]
    valid = ((people[:, 2] >= 0) & (people[:, 2] < frame_count)
             & (people[:, 5] > 0) & (people[:, 6] > 0))
    frames = defaultdict(list)
    seen = set()
    for row in people[valid]:
        tid, _, frame, x, y, width, height, _ = map(int, row)
        if (frame, tid) in seen:
            raise ValueError(f"Duplicate GT person/frame: {tid}/{frame}")
        seen.add((frame, tid))
        frames[frame].append((tid, np.array([x, y, width, height], dtype=float)))
    return frames, {
        "path": str(path.relative_to(ROOT)), "sha256": sha256(path),
        "person_rows": len(people), "excluded_invalid_rows": int((~valid).sum()),
        "person_ids": len(set(people[valid, 0].tolist())),
        "first_person_frame": min(frames) if frames else None,
        "last_person_frame": max(frames) if frames else None,
    }


def iou_matrix(boxes, targets):
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    targets = np.asarray(targets, dtype=float).reshape(-1, 4)
    low = np.maximum(boxes[:, None, :2], targets[None, :, :2])
    high = np.minimum(boxes[:, None, :2] + boxes[:, None, 2:], targets[None, :, :2] + targets[None, :, 2:])
    intersection = np.maximum(high - low, 0).prod(axis=2)
    union = boxes[:, 2:].prod(axis=1)[:, None] + targets[:, 2:].prod(axis=1)[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def unambiguous_matches(boxes, targets, minimum_iou=0.5, margin=0.1):
    scores = iou_matrix(boxes, targets)
    if not scores.size:
        return []
    pairs = []
    for i, row in enumerate(scores):
        j = int(np.argmax(row))
        if scores[i, j] < minimum_iou or int(np.argmax(scores[:, j])) != i:
            continue
        other_row = np.delete(row, j)
        other_col = np.delete(scores[:, j], i)
        if scores[i, j] - max(other_row.max(initial=0), other_col.max(initial=0)) < margin:
            continue
        pairs.append((i, j, float(scores[i, j])))
    return pairs


def inside_image(box, size):
    x, y, width, height = box
    return (np.isfinite(box).all() and min(width, height) > 0
            and x > 0 and y > 0 and x + width < size[0] - 1 and y + height < size[1] - 1)


def calibrate(args):
    output = args.output or default_calibration_path(args.source, args.model, args.imgsz)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing R calibration: {output}")
    # Reuse the exact existing YOLO/MARS bridge, including its import order.
    spec = importlib.util.spec_from_file_location("tracking_system", ROOT / "src/tracking system.py")
    tracking = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tracking)
    matrix, size, matrix_path = load_virat_homography(args.source)
    config = configuration(args.source, args.model, args.mars_model, args.imgsz)
    capture = cv2.VideoCapture(str(args.source))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {args.source}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    gt_frames, gt_info = load_gt(args.source, frame_count)
    encoder, patch_shape = tracking.create_mars_encoder(args.mars_model)
    model = tracking.YOLO(str(args.model))
    tracker = tracking.Tracker(tracking.NearestNeighborDistanceMetric("cosine", 0.2, budget=None),
                               max_iou_distance=0.7, max_age=30, n_init=3)
    records = []
    associations = defaultdict(set)
    previous = {}
    matched = 0
    frame_idx = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if (frame.shape[1], frame.shape[0]) != size:
                raise ValueError("Video resolution does not match calibration reference")
            result = model.predict(source=frame, classes=[0], device=args.device,
                                   imgsz=args.imgsz, save=False, verbose=False)[0]
            if float(model.predictor.args.conf) != config["confidence"]:
                raise ValueError("Unexpected YOLO default confidence; record/recalibrate this configuration")
            tracker.predict()
            tracker.update(tracking.make_detections(frame, result, encoder, patch_shape))
            observations = {o["track_id"]: o for o in ground_observations(tracker.tracks, matrix, frame_idx)}
            tracks = [t for t in tracker.tracks if t.track_id in observations and inside_image(t.to_tlwh(), size)]
            targets = [(tid, box) for tid, box in gt_frames.get(frame_idx, []) if inside_image(box, size)]
            current = {}
            for i, j, iou in unambiguous_matches([t.to_tlwh() for t in tracks], [box for _, box in targets]):
                tid, gid = int(tracks[i].track_id), int(targets[j][0])
                pair = (tid, gid)
                associations[tid].add(gid)
                count = previous.get(pair, 0) + 1
                current[pair] = count
                matched += 1
                if count < 3:
                    continue
                x, y, width, height = targets[j][1]
                try:
                    gt_ground = np.array(pixel_to_ground((x + width / 2, y + height), matrix))
                except ValueError:
                    continue
                observed = np.array(observations[tid]["ground_position"])
                records.append({"frame_idx": frame_idx, "track_id": tid, "gt_id": gid,
                                "iou": iou, "observed": observed.tolist(), "gt": gt_ground.tolist(),
                                "error": (observed - gt_ground).tolist()})
            previous = current  # A gap breaks the consecutive-match requirement.
            frame_idx += 1
            if frame_idx % 200 == 0:
                print(f"R calibration: {frame_idx}/{frame_count} frames, {len(records)} candidate samples", flush=True)
    finally:
        capture.release()
    if frame_idx != frame_count:
        raise RuntimeError(f"Incomplete decode: {frame_idx}/{frame_count} frames")
    unstable = {tid for tid, gids in associations.items() if len(gids) > 1}
    accepted = [record for record in records if record["track_id"] not in unstable]
    gt_ids = sorted({r["gt_id"] for r in accepted})
    if len(accepted) < 100 or len(gt_ids) < 3:
        raise RuntimeError(f"Insufficient reliable GT matches: {len(accepted)} samples, {len(gt_ids)} people; no R written")
    errors = np.array([record["error"] for record in accepted])
    variance = np.var(errors, axis=0, ddof=1)
    if not np.isfinite(errors).all() or not np.isfinite(variance).all():
        raise FloatingPointError("Non-finite calibration errors/variance")
    result = {
        "schema": 1, "configuration": config,
        "source": str(args.source.resolve()), "source_sha256": sha256(args.source),
        "frames": frame_idx, "annotation": gt_info, "homography": str(matrix_path),
        "units": "scene units squared; physical m/ft unspecified; dt=1 frame",
        "matching": {"same_frame_zero_based": True, "mutual_best_iou_minimum": 0.5,
                     "runner_up_margin": 0.1, "consecutive_frames": 3,
                     "exclude_border_boxes": True, "exclude_any_track_with_changed_gt_id": True,
                     "residual_trimming": False},
        "unambiguous_pairs": matched, "candidate_samples": len(records),
        "discarded_unstable_track_ids": sorted(unstable),
        "samples": len(accepted), "gt_object_ids": gt_ids,
        "matched_track_ids": sorted({r["track_id"] for r in accepted}),
        "mean_error": errors.mean(axis=0).tolist(), "sample_variance_ddof1": variance.tolist(),
        "variance_floor": 1e-6, "r_diagonal": np.maximum(variance, 1e-6).tolist(),
        "example_matches": accepted[::max(1, len(accepted) // 5)][:5],
        "limitations": "IoU-derived associations are conservative automatic correspondences, not identity ground truth. Samples are temporally correlated and R is a pooled diagonal approximation; nonzero mean error is reported, not corrected.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"Saved R calibration: {output.resolve()}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Development clip with original objects GT")
    parser.add_argument("--model", type=Path, default=ROOT / "weights/yolo26s.pt")
    parser.add_argument("--mars-model", type=Path, default=ROOT / "weights/mars-small128.pb")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=960, help="YOLO inference size (default: 960)")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    calibrate(parse_args())
