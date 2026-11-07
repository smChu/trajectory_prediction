"""Run YOLO/DeepSORT and frame-based EKF forecasts on original VIRAT videos."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from time import perf_counter
from zoneinfo import ZoneInfo

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch

from src.ekf import DEFAULT_Q, INITIALIZATION_OBSERVATIONS, TrackEKFs
from src.estimate_r import (TRACKER_MAX_AGE, YOLO_NMS_IOU, configuration,
                           default_calibration_path, load_gt, load_r_calibration, sha256)
from src.evaluate import FrameProcessingSpeed, QuantitativeEvaluation, group_id, save_json
from src.forecast import (ForecastComparison, GroundTruthAssociation, collect_forecasts,
                         draw_forecasts, draw_gt_paths, draw_performance_metrics,
                         draw_trajectory_legend)
from src.homography import ground_observations, load_virat_homography
from src.latency import PredictorLatency

ROOT = Path(__file__).resolve().parents[1]


def unique_path(path):
    requested = path
    index = 2
    while path.exists():
        path = requested.with_name(f"{requested.stem}_{index}{requested.suffix}")
        index += 1
    return path


def run(args):
    from src.tracking_core import (YOLO, NearestNeighborDistanceMetric, Tracker,
                                  create_mars_encoder, draw_tracks, make_detections)

    for path in (args.source, args.model, args.mars_model):
        if not path.is_file():
            raise FileNotFoundError(path)
    stamp = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d_%H%M%S_%f")
    output = unique_path(args.output or ROOT / "runs/tracking" / f"{args.source.stem}_{stamp}.mp4")
    evaluation_output = args.evaluation_output or unique_path(
        ROOT / "runs/evaluation" / f"{output.stem}.json")
    if args.evaluate and evaluation_output.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation: {evaluation_output}")
    homography, image_size, homography_path = load_virat_homography(args.source)
    inverse_homography = np.linalg.inv(homography)
    scene = homography_path.stem.split("_")[1]
    calibration = calibration_path = None
    if args.ekf_r is not None:
        r = args.ekf_r
        print("Using provisional R; not GT-calibrated", flush=True)
    else:
        calibration_path = args.r_calibration or default_calibration_path(
            args.source, args.model, args.imgsz)
        r, calibration = load_r_calibration(
            calibration_path, args.source, args.model, args.mars_model, args.imgsz)
    ekfs = TrackEKFs(r, q=args.ekf_q, p0=args.ekf_p0,
                     warmup_frames=args.ekf_warmup_frames, min_speed=args.ekf_min_speed,
                     initialization_image_size=image_size,
                     initialization_border_margin=args.ekf_init_border_margin)
    p0 = args.ekf_p0 if args.ekf_p0 is not None else [*r, 1., np.pi**2, .25]
    latency = (PredictorLatency(("ekf",), args.latency_warmup_frames)
               if args.measure_predictor_latency else None)
    comparisons, gt_association = ForecastComparison(), GroundTruthAssociation()
    encoder, patch_shape = create_mars_encoder(args.mars_model)
    model = YOLO(str(args.model))
    tracker = Tracker(NearestNeighborDistanceMetric("cosine", .2, budget=None),
                      max_iou_distance=.7, max_age=TRACKER_MAX_AGE, n_init=3)
    capture = cv2.VideoCapture(str(args.source))
    writer = None
    frames = 0
    speed = FrameProcessingSpeed(warmup_frames=10)
    window = "YOLO + Original DeepSORT + EKF (q / Esc: quit)"

    def synchronize_gpu():
        device = getattr(getattr(model, "predictor", None), "device", None)
        if device is not None and device.type == "cuda":
            torch.cuda.synchronize(device)

    try:
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {args.source}")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            raise RuntimeError("Video FPS is invalid")
        # GT is used only for evaluation/overlays, never for tracking or EKF.
        gt_frames, gt_info = load_gt(args.source, frame_count)
        evaluator = QuantitativeEvaluation(gt_frames, homography, image_size, frame_count)
        print(f"Scene: {scene}; position=scene units; dt=1 frame; horizon=12 frames", flush=True)
        while True:
            synchronize_gpu()
            frame_started = perf_counter()
            ok, frame = capture.read()
            if not ok:
                if frames == 0:
                    raise RuntimeError(f"Cannot read first frame: {args.source}")
                break
            if (frame.shape[1], frame.shape[0]) != image_size:
                raise ValueError(f"Video resolution must match homography reference {image_size}")
            result = model.predict(source=frame, classes=[0], device=args.device,
                                   imgsz=args.imgsz, iou=YOLO_NMS_IOU,
                                   save=False, verbose=False)[0]
            detections = make_detections(frame, result, encoder, patch_shape)
            tracker.predict()
            tracker.update(detections)
            observations = ground_observations(tracker.tracks, homography, frames)
            active_ids = [t.track_id for t in tracker.tracks if t.is_confirmed()]
            prediction_started = perf_counter()
            states = ekfs.step(frames, observations, active_ids)
            forecasts = collect_forecasts(ekfs, tracker.tracks)
            if latency is not None:
                latency.record(frames, {"ekf": perf_counter() - prediction_started},
                               {"ekf": [p["track_id"] for p in forecasts]}, len(active_ids))
            for state in states:
                values = " ".join(
                    f"{name}={state[key]:.4f}" if state[key] is not None else f"{name}=--"
                    for name, key in (("px", "estimated_px"), ("py", "estimated_py"),
                                      ("v", "estimated_v"), ("theta", "estimated_theta"),
                                      ("omega", "estimated_omega")))
                print(f"STATE method=ekf frame={frames} ID={state['track_id']} {values} "
                      f"measured={state['has_measurement']}", flush=True)
            targets = gt_frames.get(frames, [])
            associations = gt_association.step(frames, tracker.tracks, targets, image_size)
            snapshots = comparisons.step(frames, forecasts, targets, associations,
                                         gt_association.for_display(tracker.tracks, targets))
            evaluator.step(frames, snapshots, associations, observations, states, ekfs, forecasts)
            draw_forecasts(frame, snapshots, inverse_homography,
                           ellipse_points=args.ellipse_points,
                           thickness=args.forecast_line_thickness, label="EKF")
            draw_gt_paths(frame, snapshots, args.forecast_line_thickness)
            draw_tracks(frame, tracker)
            draw_trajectory_legend(frame)
            draw_performance_metrics(frame, evaluator.overlay_metrics(),
                                     speed.report()["processing_fps"])
            if writer is None and not args.no_save:
                output.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, image_size)
                if not writer.isOpened():
                    raise RuntimeError(f"Cannot write video: {output}")
            if writer is not None:
                writer.write(frame)
            frames += 1
            quit_requested = False
            if not args.no_display:
                cv2.imshow(window, frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    quit_requested = True
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    quit_requested = True
            synchronize_gpu()
            speed.record(perf_counter() - frame_started)
            if quit_requested or (args.max_frames is not None and frames >= args.max_frames):
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    if args.evaluate:
        same_recording = (group_id(args.source) == group_id(calibration["source"])) if calibration else None
        metadata = {
            "source": str(args.source.resolve()), "source_sha256": sha256(args.source),
            "scene": scene, "fps": fps, "predictor": "ekf", "horizon_seconds": 12 / fps,
            "annotation": gt_info,
            "configuration": configuration(args.source, args.model, args.mars_model, args.imgsz),
            "q_diagonal": list(args.ekf_q), "r_diagonal": list(map(float, r)),
            "p0_diagonal": list(map(float, p0)),
            "warmup_frames": args.ekf_warmup_frames, "min_speed": args.ekf_min_speed,
            "initialization_policy": {
                "image_size": list(image_size), "border_margin_pixels": args.ekf_init_border_margin,
                "max_accepted_frame_gap": ekfs.initialization_max_gap_frames,
                "observations": INITIALIZATION_OBSERVATIONS,
                "scope": "initial bootstrap only; established updates unchanged"},
            "r_calibration": str(calibration_path) if calibration_path else None,
            "r_calibration_sha256": sha256(calibration_path) if calibration_path else None,
            "same_recording_as_r_calibration": same_recording,
            "evaluation_label": ("development_same_R_recording" if same_recording else
                                 "provisional_R" if calibration is None else "unassigned_split"),
            "processing_speed": {
                **speed.report(),
                "scope": "frame read through tracking, EKF, GT evaluation, overlays, logging and enabled video/display; CUDA synchronized; excludes loading and final writer release",
                "display_enabled": not args.no_display, "video_save_enabled": not args.no_save,
                "state_logging_enabled": True, "imgsz": args.imgsz,
                "device": str(getattr(getattr(model, "predictor", None), "device", args.device))},
        }
        if latency is not None:
            metadata["predictor_latency"] = latency.report({"ekf": "cpu"})
        report = evaluator.report(metadata)
        save_json(evaluation_output, report)
        print(json.dumps({"metrics": report["metrics"], "evaluated_windows": report["evaluated_windows"],
                          "processing_speed": report["processing_speed"]}, allow_nan=False), flush=True)
        print(f"Evaluation saved: {evaluation_output.resolve()}", flush=True)
    print(f"Processed {frames} frames", flush=True)
    if writer is not None:
        print(f"Video saved: {output.resolve()}", flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=ROOT / "weights/yolo26s.pt")
    parser.add_argument("--mars-model", type=Path, default=ROOT / "weights/mars-small128.pb")
    parser.add_argument("--device", default="0", help="YOLO device: 0 or cpu")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--predictor", choices=("ekf",), default="ekf")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--max-frames", type=int, help="Limit a smoke run; partial results are marked incomplete")
    parser.add_argument("--evaluate", action="store_true", help="Save original-GT ADE/FDE and processing FPS")
    parser.add_argument("--evaluation-output", type=Path)
    parser.add_argument("--measure-predictor-latency", action="store_true")
    parser.add_argument("--latency-warmup-frames", type=int, default=50)
    noise = parser.add_mutually_exclusive_group()
    noise.add_argument("--r-calibration", type=Path)
    noise.add_argument("--ekf-r", type=float, nargs=2, metavar=("VAR_X", "VAR_Y"))
    parser.add_argument("--ekf-q", type=float, nargs=5, default=DEFAULT_Q)
    parser.add_argument("--ekf-p0", type=float, nargs=5)
    parser.add_argument("--ekf-warmup-frames", type=int, default=9)
    parser.add_argument("--ekf-min-speed", type=float, default=.01)
    parser.add_argument("--ekf-init-border-margin", type=float, default=1.)
    parser.add_argument("--ellipse-points", type=int, choices=range(32, 65), default=48)
    parser.add_argument("--forecast-line-thickness", type=int, default=1)
    args = parser.parse_args(argv)
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive")
    if args.latency_warmup_frames < 0 or args.forecast_line_thickness < 1:
        parser.error("Latency warmup must be nonnegative and line thickness must be positive")
    if (args.evaluation_output is not None or args.measure_predictor_latency) and not args.evaluate:
        parser.error("--evaluation-output and --measure-predictor-latency require --evaluate")
    return args


if __name__ == "__main__":
    run(parse_args())
