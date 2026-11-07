"""Original VIRAT GT evaluation; no GT ever enters tracking or prediction.

Prepare grouped calibration/validation/test splits, or run the unchanged
tracking pipeline with evaluation enabled. Distances are scene units, not m.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

# Allow direct script execution as well as python -m from the repository root.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import cv2
import numpy as np

from src.ekf import DEFAULT_Q
from src.estimate_r import ROOT, default_calibration_path, inside_image, load_gt, load_r_calibration, sha256
from src.homography import pixel_to_ground
def linear_forecast(samples, frame_idx):
    """Legacy evaluation baseline: last nine real times, last-point anchor."""
    samples = list(samples)[-9:]
    times = np.array([frame for frame, _ in samples], dtype=float)
    positions = np.array([position for _, position in samples], dtype=float)
    centered = times - times.mean() if len(times) else times
    if len(times) < 2 or centered @ centered <= 0:
        return None
    velocity = centered @ (positions - positions.mean(axis=0)) / (centered @ centered)
    return positions[-1] + (frame_idx + np.arange(1, 13) - times[-1])[:, None] * velocity



def group_id(source):
    return Path(source).stem.split("_")[2]


def split_groups(groups, reserved):
    remaining = sorted(set(groups) - {reserved},
                       key=lambda g: hashlib.sha256(f"virat-original-v1:{g}".encode()).hexdigest())
    if len(remaining) < 2:
        raise ValueError("Need distinct validation/test recordings")
    validation = set(remaining[:max(1, len(remaining) // 2)])
    return {g: "calibration" if g == reserved else "validation" if g in validation else "test"
            for g in groups}


def distance_summary(values):
    values = np.asarray(values, dtype=float)
    return {"samples": len(values),
            "mean_distance": float(values.mean()) if len(values) else None,
            "rmse_distance": float(np.sqrt(np.mean(values ** 2))) if len(values) else None}


class FrameProcessingSpeed:
    """Measure complete frame iterations, excluding initial warmup frames."""

    def __init__(self, warmup_frames=10):
        self.warmup_frames = warmup_frames
        self.seen = 0
        self.measured_frames = 0
        self.elapsed_seconds = 0.0

    def record(self, seconds):
        if not np.isfinite(seconds) or seconds <= 0:
            raise ValueError("Frame processing time must be positive and finite")
        self.seen += 1
        if self.seen > self.warmup_frames:
            self.measured_frames += 1
            self.elapsed_seconds += seconds

    def report(self):
        return {"measured_frames": self.measured_frames,
                "elapsed_seconds": self.elapsed_seconds,
                "processing_fps": (self.measured_frames / self.elapsed_seconds
                                   if self.measured_frames else None),
                "warmup_frames_excluded": min(self.seen, self.warmup_frames)}


def processing_speed_summary(reports):
    """Weight speeds by total measured time, never average per-video FPS."""
    speeds = [r["processing_speed"] for r in reports if "processing_speed" in r]
    frames = sum(s["measured_frames"] for s in speeds)
    seconds = sum(s["elapsed_seconds"] for s in speeds)
    return {"measured_frames": frames, "elapsed_seconds": seconds,
            "processing_fps": frames / seconds if seconds > 0 else None,
            "videos_with_timing": len(speeds)}


def window_summary(windows, method="ekf"):
    valid = [w for w in windows if w["status"] == "evaluated"]
    counts = Counter(w["status"] for w in windows)
    metrics = {}
    for name in (method, "linear"):
        metrics[name] = {
            "ade": float(np.mean([w[name]["ade"] for w in valid])) if valid else None,
            "fde": float(np.mean([w[name]["fde"] for w in valid])) if valid else None,
        }
    coverage = [x for w in valid for x in w["ellipse_contains_gt"] if x is not None]
    matched = sum(w["gt_id"] is not None for w in windows)
    return {
        "generated_windows": len(windows), "matched_windows": matched,
        "evaluated_windows": len(valid), "status_counts": dict(counts),
        "matched_fraction": matched / len(windows) if windows else None,
        "evaluated_fraction": len(valid) / len(windows) if windows else None,
        "metrics": metrics, "ellipse_valid_points": len(coverage),
        "ellipse_95_gt_coverage": float(np.mean(coverage)) if coverage else None,
    }


class QuantitativeEvaluation:
    """Score new fixed comparison windows, on a common complete-GT cohort.

    GT ID is locked at start, and never selected using future errors. Full
    horizon GT is used only to score; no interpolation or outlier trimming.
    Linear baseline fits the last nine real observations vs actual frames,
    anchored at the last observation, without using any GT or future frames.
    """

    def __init__(self, gt_frames, matrix, image_size, frame_count, method="ekf"):
        if method != "ekf":
            raise ValueError("Unknown prediction method")
        self.method = method
        self.gt = {f: dict(targets) for f, targets in gt_frames.items()}
        self.matrix, self.image_size = matrix, image_size
        self.frame_count = frame_count
        self.windows = []
        self._overlay_index = 0
        self._overlay_count = 0
        self._overlay_sum = np.zeros(2, dtype=float)
        self.errors = {"observed": []}
        if method == "ekf":
            self.errors["ekf"] = []
        self.frames = 0
        self.gt_person_frames = 0
        self.matched_person_frames = 0
        self.live_forecast_track_frames = 0
        self.previous_gt_tracks = {}
        self.track_gt = {}
        self.adjacent_gt_track_changes = 0
        self.track_gt_changes = 0

    def ground(self, frame, gid):
        box = self.gt.get(frame, {}).get(gid)
        if box is None:
            return None, "missing_gt"
        if not inside_image(box, self.image_size):
            return None, "gt_border_or_truncated"
        x, y, width, height = box
        try:
            return np.array(pixel_to_ground((x + width / 2, y + height), self.matrix)), None
        except ValueError:
            return None, "invalid_gt_projection"

    @staticmethod
    def linear_forecast(ekf, frame):
        return linear_forecast(ekf.samples, frame)

    def step(self, frame, snapshots, associations, observations, states, ekfs, live_forecasts):
        if frame != self.frames:
            raise ValueError("Evaluation requires each video frame exactly once, starting at zero")
        self.frames += 1
        self.gt_person_frames += len(self.gt.get(frame, {}))
        self.matched_person_frames += len(associations)
        self.live_forecast_track_frames += len(live_forecasts)
        current_gt_tracks = {gid: tid for tid, gid in associations.items()}
        self.adjacent_gt_track_changes += sum(
            gid in self.previous_gt_tracks and self.previous_gt_tracks[gid] != tid
            for gid, tid in current_gt_tracks.items())
        self.previous_gt_tracks = current_gt_tracks
        for tid, gid in associations.items():
            self.track_gt_changes += tid in self.track_gt and self.track_gt[tid] != gid
            self.track_gt[tid] = gid
        measured = {o["track_id"]: o for o in observations}
        for state in states:
            tid = state["track_id"]
            if not state["initialized"] or tid not in measured or tid not in associations:
                continue
            target, reason = self.ground(frame, associations[tid])
            if reason:
                continue
            self.errors["observed"].append(float(np.linalg.norm(
                np.asarray(measured[tid]["ground_position"]) - target)))
            if self.method == "ekf":
                self.errors["ekf"].append(float(np.linalg.norm(
                    np.array([state["estimated_px"], state["estimated_py"]]) - target)))
        for snapshot in snapshots:
            if snapshot["frame_idx"] != frame:
                continue
            gid, tid = snapshot["gt_id"], snapshot["track_id"]
            result = {"start_frame_idx": frame, "end_frame_idx": frame + 12,
                      "track_id": tid, "gt_id": gid, "status": "pending",
                      "has_measurement_at_start": snapshot["has_measurement"]}
            self.windows.append(result)
            predicted = np.array([p["predicted_position"] for p in snapshot["predictions"]])
            covariance = (np.array([p["predicted_covariance"] for p in snapshot["predictions"]])[:, :2, :2]
                          if self.method == "ekf" else [None] * 12)
            if predicted.shape != (12, 2) or not np.isfinite(predicted).all():
                result["status"] = "invalid_prediction"
                continue
            if gid is None:
                result["status"] = "unmatched_at_start"
                continue
            if frame + 12 >= self.frame_count:
                result["status"] = "video_end"
                continue
            targets = []
            for future in range(frame + 1, frame + 13):
                target, reason = self.ground(future, gid)
                if reason:
                    result["status"] = reason
                    break
                targets.append(target)
            if result["status"] != "pending":
                continue
            baseline = self.linear_forecast(ekfs.filters[tid], frame)
            if baseline is None or not np.isfinite(baseline).all():
                result["status"] = "invalid_linear_baseline"
                continue
            targets = np.array(targets)
            result.update(status="evaluated", gt_ground=targets.tolist(),
                          predicted_ground=predicted.tolist(), linear_ground=baseline.tolist())
            for method, positions in ((self.method, predicted), ("linear", baseline)):
                errors = np.linalg.norm(positions - targets, axis=1)
                result[method] = {"ade": float(errors.mean()), "fde": float(errors[-1]),
                                  "errors_by_offset": errors.tolist()}
            contains = []
            for residual, cov in zip(targets - predicted, covariance):
                if cov is None:
                    contains.append(None)
                    continue
                # A singular covariance has no invertible 2D ellipse. Do not
                # regularize it into an invented confidence region.
                eigenvalues = np.linalg.eigvalsh(cov) if np.isfinite(cov).all() else [0]
                if min(eigenvalues) <= 0:
                    contains.append(None)
                else:
                    contains.append(bool(residual @ np.linalg.solve(cov, residual) <= 5.991))
            result["ellipse_contains_gt"] = contains

    def overlay_metrics(self):
        """Cumulative EKF errors for horizons already reached in playback only."""
        while self._overlay_index < len(self.windows):
            item = self.windows[self._overlay_index]
            if item["end_frame_idx"] >= self.frames:
                break
            self._overlay_index += 1
            if item["status"] == "evaluated":
                self._overlay_sum += [item[self.method]["ade"], item[self.method]["fde"]]
                self._overlay_count += 1
        means = self._overlay_sum / self._overlay_count if self._overlay_count else (None, None)
        return {"samples": self._overlay_count, "ade": means[0], "fde": means[1]}

    def report(self, metadata):
        # Early quit/read truncation must not evaluate unprocessed future GT.
        windows = []
        for item in self.windows:
            if item["status"] == "evaluated" and item["end_frame_idx"] >= self.frames:
                item = {k: v for k, v in item.items() if k in (
                    "start_frame_idx", "end_frame_idx", "track_id", "gt_id", "has_measurement_at_start")}
                item["status"] = "run_ended_before_horizon"
            windows.append(item)
        return {"schema": 1, **metadata, "processed_frames": self.frames,
                "complete_video": self.frames == self.frame_count,
                "units": "scene-dependent world units; m/ft unspecified",
                "horizon_frames": 12, "overlay_start_stride_frames": 13,
                "evaluation_start_stride_frames": 13,
                "protocol": {"gt": "original VIRAT objects, class 1; no DIVA",
                             "matching": "mutual best IoU>=0.5, margin>=0.1, 3 consecutive frames; locked GT ID",
                             "cohort": "all 12 GT frames present, inside image and finite; no interpolation or error trimming",
                             "aggregation": "mean of per-window ADE/FDE; same windows for both methods",
                             "linear": "least squares over last 9 actual observations; anchored at last observation",
                             "coverage": ("empirical pointwise 95% ellipse coverage on evaluated windows"
                                          if self.method == "ekf" else f"not available for deterministic {self.method.upper()}"),
                             "limitations": "automatic ID association is not verified identity GT; conditional prediction metrics, not end-to-end recall; homography is approximate"},
                "gt_person_frames": self.gt_person_frames,
                "matched_person_frames": self.matched_person_frames,
                "association_fraction": self.matched_person_frames / self.gt_person_frames if self.gt_person_frames else None,
                "live_forecast_track_frames": self.live_forecast_track_frames,
                "association_changes": {"adjacent_gt_track_changes": self.adjacent_gt_track_changes,
                                        "track_gt_changes": self.track_gt_changes,
                                        "note": "diagnostic counts; not standard MOT IDSW"},
                "current_position": {k: distance_summary(v) for k, v in self.errors.items()},
                **window_summary(windows, self.method), "windows": windows}


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return path


def prepare_split(path, all_scenes=False):
    if path.exists():
        raise FileExistsError(f"Split already exists; use it or specify another --manifest: {path}")
    entries, calibrations, excluded, constraints = [], {}, [], {}
    scenes = sorted({p.stem.split("_")[2][:4] for p in
                     (ROOT / "Resources/videos_original").glob("VIRAT_S_*.mp4")}) if all_scenes else ["0100", "0101", "0102"]
    for scene in scenes:
        videos = sorted((ROOT / "Resources/videos_original").glob(f"VIRAT_S_{scene}*.mp4"))
        if not videos:
            continue
        calibration_path = default_calibration_path(videos[0], ROOT / "weights/yolo26s.pt", 960)
        if not calibration_path.is_file() and all_scenes:
            excluded.append({"scene": scene, "reason": "R calibration unavailable"})
            continue
        _, calibration = load_r_calibration(calibration_path, videos[0], ROOT / "weights/yolo26s.pt",
                                            ROOT / "weights/mars-small128.pb", 960)
        reserved = group_id(calibration["source"])
        groups = {group_id(v) for v in videos}
        if len(groups - {reserved}) >= 2:
            roles = split_groups(groups, reserved)
        elif all_scenes:
            # One recording cannot be simultaneously calibration and an
            # independent test. Do not split its adjacent clips to invent one.
            roles = {g: "calibration" if g == reserved else "validation" for g in groups}
            constraints[scene] = "No independent test recording after reserving R source; validation only" if len(groups) > 1 else "Only the R calibration recording exists; no independent validation/test"
        else:
            raise ValueError(f"Need distinct validation/test recordings in {scene}")
        calibrations[scene] = {"path": str(calibration_path.relative_to(ROOT)),
                               "sha256": sha256(calibration_path), "reserved_group": reserved,
                               "source": Path(calibration["source"]).name}
        for video in videos:
            annotation = ROOT / "Resources/annotations" / f"{video.stem}.viratdata.objects.txt"
            if not annotation.is_file():
                excluded.append({"source": str(video.relative_to(ROOT)), "reason": "original GT missing"})
                continue
            cap = cv2.VideoCapture(str(video))
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS); cap.release()
            try:
                _, gt_info = load_gt(video, frames)
            except (ValueError, OSError) as error:
                excluded.append({"source": str(video.relative_to(ROOT)), "reason": f"invalid original GT: {error}"})
                continue
            group = group_id(video)
            role = roles[group]
            entries.append({"source": str(video.relative_to(ROOT)), "scene": scene,
                            "group": group, "split": role, "frames": frames, "fps": fps,
                            "video_bytes": video.stat().st_size, "annotation_sha256": gt_info["sha256"]})
    for scene in calibrations:
        available = {e["split"] for e in entries if e["scene"] == scene}
        if not {"validation", "test"}.issubset(available):
            constraints.setdefault(scene, "Valid original GT does not provide both independent validation and test recordings")
    manifest = {"schema": 1, "protocol": "group by six-digit source recording; deterministic v1; existing R sources and siblings reserved",
                "scope": "all available scenes with valid imgsz960 R and original GT" if all_scenes else "0100/0101/0102 with existing imgsz960 R",
                "limitations": "clips already inspected during development are not a pristine blind test set",
                "scene_constraints": constraints, "excluded": excluded,
                "calibrations": calibrations, "videos": entries}
    save_json(path, manifest)
    return manifest


def run_batch(args):
    manifest = json.loads(args.manifest.read_text())
    selected = [e for e in manifest["videos"] if e["split"] == args.split]
    if args.limit_per_scene:
        selected = [entry for scene in sorted({e["scene"] for e in selected})
                    for entry in sorted((e for e in selected if e["scene"] == scene),
                                        key=lambda e: (e["frames"], e["source"]))[:args.limit_per_scene]]
    stamp = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d_%H%M%S_%f")
    directory = ROOT / "runs/evaluation" / f"{args.split}_{stamp}"
    directory.mkdir(parents=True)
    reports = []
    for entry in selected:
        source = ROOT / entry["source"]
        calibration = manifest["calibrations"][entry["scene"]]
        annotation = ROOT / "Resources/annotations" / f"{source.stem}.viratdata.objects.txt"
        if (source.stat().st_size != entry["video_bytes"] or sha256(annotation) != entry["annotation_sha256"]
                or sha256(ROOT / calibration["path"]) != calibration["sha256"]):
            raise ValueError("Dataset/calibration changed after preparing split; inspect before continuing")
        if entry["group"] == calibration["reserved_group"]:
            raise ValueError("R calibration recording cannot enter validation/test")
        output = directory / f"{source.stem}.json"
        command = [sys.executable, "-B", str(ROOT / "src/track.py"),
                   "--source", str(source), "--device", args.device, "--imgsz", "960",
                   "--no-display", "--no-save", "--evaluate", "--evaluation-output", str(output),
                   "--ekf-q", *map(str, args.ekf_q)]
        print(f"Evaluating {entry['scene']} {source.name}", flush=True)
        with output.with_suffix(".log").open("x") as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        report = json.loads(output.read_text())
        report.update(split=args.split, evaluation_label=args.split,
                      split_manifest_sha256=sha256(args.manifest))
        # Add batch provenance only to the file just produced in this unique
        # run directory; existing evaluations are never reused/overwritten.
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        reports.append(report)
    scenes = {}
    for scene in sorted({r["scene"] for r in reports}):
        subset = [r for r in reports if r["scene"] == scene]
        summary = window_summary([w for r in subset for w in r["windows"]])
        for key in ("gt_person_frames", "matched_person_frames", "processed_frames"):
            summary[key] = sum(r[key] for r in subset)
        summary["current_position"] = {}
        for method in ("observed", "ekf"):
            values = [r["current_position"][method] for r in subset]
            count = sum(v["samples"] for v in values)
            summary["current_position"][method] = {
                "samples": count,
                "mean_distance": sum(v["mean_distance"] * v["samples"] for v in values if v["samples"]) / count if count else None,
                "rmse_distance": float(np.sqrt(sum(v["rmse_distance"] ** 2 * v["samples"]
                                                   for v in values if v["samples"]) / count)) if count else None,
            }
        summary["complete_videos"] = sum(r["complete_video"] for r in subset)
        summary["association_fraction"] = (summary["matched_person_frames"] / summary["gt_person_frames"]
                                             if summary["gt_person_frames"] else None)
        summary["videos"] = len(subset)
        summary["processing_speed"] = processing_speed_summary(subset)
        scenes[scene] = summary
    summary = {"schema": 1, "split": args.split, "partial_split": bool(args.limit_per_scene),
               "manifest": str(args.manifest), "manifest_sha256": sha256(args.manifest),
               "q_diagonal": list(args.ekf_q), "units": "scene units; never pool distance across scenes",
               "q_status": "user-supplied fixed Q; no automatic optimization",
               "scene_constraints": manifest.get("scene_constraints", {}),
               "scenes": scenes, "reports": [r["source"] for r in reports]}
    save_json(directory / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"Saved: {directory / 'summary.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-split", action="store_true")
    mode.add_argument("--split", choices=("validation", "test"))
    parser.add_argument("--manifest", type=Path, default=ROOT / "runs/evaluation/original_gt_split.json")
    parser.add_argument("--limit-per-scene", type=int, help="Smoke evaluation: shortest N clips per scene, not the full split")
    parser.add_argument("--device", default="0")
    parser.add_argument("--all-scenes", action="store_true", help="Prepare a separate split for all scenes with available R")
    parser.add_argument("--ekf-q", type=float, nargs=5, default=DEFAULT_Q)
    args = parser.parse_args()
    if args.limit_per_scene is not None and args.limit_per_scene < 1:
        parser.error("--limit-per-scene must be positive")
    if args.all_scenes and not args.prepare_split:
        parser.error("--all-scenes is for --prepare-split; batch runs use the selected --manifest")
    if args.prepare_split:
        manifest = prepare_split(args.manifest, all_scenes=args.all_scenes)
        print(json.dumps({scene: dict(Counter(v["split"] for v in manifest["videos"] if v["scene"] == scene))
                          for scene in manifest["calibrations"]}, indent=2))
        print(f"Saved: {args.manifest}")
    else:
        run_batch(args)


if __name__ == "__main__":
    main()
