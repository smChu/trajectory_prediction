"""Track VIRAT pedestrians with YOLO26s and the original nwojke DeepSORT."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
# Import the checked-out original tracker, rather than an installed substitute.
sys.path.insert(0, str(ROOT / "deep_sort"))

import cv2
import numpy as np
# Load torchvision (and PyTorch's Triton) before TensorFlow 2.21.0. Reversing
# this order causes an LLVM symbol clash when YOLO first imports torchvision.
import torchvision
import tensorflow as tf
from ultralytics import YOLO

from deep_sort.detection import Detection
from deep_sort.nn_matching import NearestNeighborDistanceMetric
from deep_sort.tracker import Tracker
from tools.generate_detections import create_box_encoder, extract_image_patch
from homography import ground_observations, load_virat_homography
from ekf import DEFAULT_Q, INITIALIZATION_OBSERVATIONS, TrackEKFs
from estimate_r import default_calibration_path, load_r_calibration
from forecast import collect_forecasts, draw_forecasts


def create_mars_encoder(model_path):
    """Use the original frozen-graph encoder in an isolated TF1-style graph."""
    with tf.Graph().as_default() as graph:
        # The original loader imports the graph with name="net". Its default
        # unscoped tensor names do not refer to those imported tensors.
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


def draw_tracks(frame, tracker):
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        x1, y1, x2, y2 = np.rint(track.to_tlbr()).astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame, f"ID {track.track_id}", (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
        )


def run(args):
    for path in (args.source, args.model, args.mars_model):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output is None:
        timestamp = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%d_%H_%M")
        args.output = ROOT / "runs/tracking" / f"{args.source.stem}_{timestamp}.mp4"
    requested_output = args.output
    index = 2
    while args.output.exists():
        args.output = requested_output.with_name(
            f"{requested_output.stem}_{index}{requested_output.suffix}"
        )
        index += 1
    print(f"Output: {args.output}")
    print(f"YOLO imgsz: {args.imgsz}")

    homography, calibration_size, homography_path = load_virat_homography(args.source)
    scene = homography_path.stem.split("_")[1]
    print(f"VIRAT scene: {scene} (selected from video filename)")
    print(f"Homography: {homography_path} (pixel -> ground, no inverse)")
    print(f"Calibration image size: {calibration_size[0]}x{calibration_size[1]}")
    inverse_homography = np.linalg.inv(homography)
    print("Forecast overlay: inverse H maps ground -> original video pixels; "
          "yellow=next 12 frames, magenta=95% covariance contours")
    if args.forecast_line_thickness < 1:
        raise ValueError("Forecast line thickness must be positive")
    print("Ground unit: scene-dependent world units; m/ft not specified by VIRAT files")
    print("frame_idx is zero-based within this input clip")

    if args.ekf_r is not None:
        r = args.ekf_r
        print("EKF R: explicitly supplied PROVISIONAL variances; not GT-calibrated")
    else:
        calibration_path = args.r_calibration or default_calibration_path(
            args.source, args.model, args.imgsz
        )
        r, calibration = load_r_calibration(
            calibration_path, args.source, args.model, args.mars_model, args.imgsz
        )
        print(f"EKF R calibration: {calibration_path}; source={calibration['source']}; "
              f"samples={calibration['samples']}; GT people={len(calibration['gt_object_ids'])}")
        if Path(calibration["source"]).stem == args.source.stem:
            print("This is the R development clip; do not report it as independent evaluation")
    ekfs = TrackEKFs(r, q=args.ekf_q, p0=args.ekf_p0,
                     warmup_frames=args.ekf_warmup_frames, min_speed=args.ekf_min_speed)
    settings = ekfs.options
    p0 = settings["p0"] if settings["p0"] is not None else [*r, 1.0, np.pi**2, 0.25]
    print(f"EKF: dt=1 frame, bootstrap={INITIALIZATION_OBSERVATIONS} actual observations (two 5-point fits); "
          f"warmup={args.ekf_warmup_frames} elapsed frames after bootstrap (not a convergence test)")
    print(f"EKF Q diagonal (initial, not optimized): {list(args.ekf_q)}")
    print(f"EKF R diagonal: {list(r)}; P0 diagonal: {list(p0)}")
    print("EKF units: position=scene units, v=scene units/frame, theta=rad, omega=rad/frame; "
          "null motion components mean not yet observable")

    encoder, patch_shape = create_mars_encoder(args.mars_model)
    model = YOLO(str(args.model))
    metric = NearestNeighborDistanceMetric("cosine", 0.2, budget=None)
    tracker = Tracker(metric, max_iou_distance=0.7, max_age=30, n_init=3)
    capture = cv2.VideoCapture(str(args.source))
    writer = None
    frames = 0
    window = "YOLO26s + Original DeepSORT (q / Esc: quit)"

    try:
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {args.source}")
        while True:
            ok, frame = capture.read()
            if not ok:
                if frames == 0:
                    raise RuntimeError(f"Cannot read first frame: {args.source}")
                break
            frame_size = (frame.shape[1], frame.shape[0])
            if frame_size != calibration_size:
                raise ValueError(
                    f"Video size {frame_size} does not match calibration {calibration_size}; "
                    "use the corresponding original-resolution VIRAT video"
                )

            result = model.predict(
                source=frame, classes=[0], device=args.device,
                imgsz=args.imgsz, save=False, verbose=False,
            )[0]
            detections = make_detections(frame, result, encoder, patch_shape)
            # Empty detections must still advance the original track lifecycle.
            tracker.predict()
            tracker.update(detections)
            observations = ground_observations(tracker.tracks, homography, frames)
            if observations:
                for observation in observations:
                    print(observation, flush=True)
            else:
                print(f"frame_idx={frames} observations=[]", flush=True)
            states = ekfs.step(
                frames, observations,
                [track.track_id for track in tracker.tracks if track.is_confirmed()],
            )
            for state in states:
                print("EKF " + json.dumps(state, allow_nan=False), flush=True)
            forecasts = collect_forecasts(ekfs, tracker.tracks)
            overlay_counts = draw_forecasts(
                frame, forecasts, inverse_homography,
                ellipse_points=args.ellipse_points, thickness=args.forecast_line_thickness,
            )
            print("FORECAST " + json.dumps({
                "frame_idx": frames, "track_ids": [f["track_id"] for f in forecasts],
                **overlay_counts,
            }), flush=True)
            draw_tracks(frame, tracker)

            if writer is None:
                fps = capture.get(cv2.CAP_PROP_FPS)
                if not np.isfinite(fps) or fps <= 0:
                    raise RuntimeError("Video FPS is invalid; cannot save video")
                args.output.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(
                    str(args.output), cv2.VideoWriter_fourcc(*"mp4v"),
                    fps, (frame.shape[1], frame.shape[0]),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Cannot write video: {args.output}")
            # Save the annotated frame, instead of YOLO's untracked output.
            writer.write(frame)

            frames += 1
            if not args.no_display:
                cv2.imshow(window, frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    print(f"Processed {frames} frames")
    print(f"Saved: {args.output.resolve()}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="VIRAT video")
    parser.add_argument("--model", type=Path, default=ROOT / "weights/yolo26s.pt")
    parser.add_argument(
        "--mars-model", type=Path, default=ROOT / "weights/mars-small128.pb"
    )
    parser.add_argument("--device", default="0", help="YOLO device (default: 0)")
    parser.add_argument("--imgsz", type=int, default=960, help="YOLO inference size (default: 960)")
    parser.add_argument(
        "--output", type=Path,
        help="MP4 path (default: runs/tracking/<source>_DD_HH_MM.mp4, Asia/Seoul); "
        "append a number if the path already exists",
    )
    parser.add_argument("--no-display", action="store_true", help="Run without a GUI")
    noise = parser.add_mutually_exclusive_group()
    noise.add_argument("--r-calibration", type=Path, help="GT-derived R JSON (default: matching runs/calibration file)")
    noise.add_argument("--ekf-r", type=float, nargs=2, metavar=("VAR_X", "VAR_Y"),
                       help="Explicit provisional R variances; not GT-calibrated")
    parser.add_argument("--ekf-q", type=float, nargs=5, default=DEFAULT_Q,
                        metavar=("PX", "PY", "V", "THETA", "OMEGA"), help="Five Q diagonal variances per frame")
    parser.add_argument("--ekf-p0", type=float, nargs=5,
                        metavar=("PX", "PY", "V", "THETA", "OMEGA"), help="P0 variances (default: Rxx Ryy 1 pi^2 0.25)")
    parser.add_argument("--ekf-warmup-frames", type=int, default=9,
                        help="Elapsed warmup frames after 9-observation bootstrap; does not stop estimation")
    parser.add_argument("--ekf-min-speed", type=float, default=0.01,
                        help="Stationary threshold in scene units/frame for initialization")
    parser.add_argument("--ellipse-points", type=int, choices=range(32, 65), default=48,
                        help="Samples per 95%% ground ellipse boundary (32..64; default: 48)")
    parser.add_argument("--forecast-line-thickness", type=int, default=1,
                        help="Forecast trajectory/ellipse line thickness in pixels (default: 1)")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
