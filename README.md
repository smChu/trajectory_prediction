# Pedestrian Trajectory Prediction with YOLO26s, DeepSORT, and EKF

A lightweight framework for **pedestrian detection, multi-object tracking, and short-term trajectory prediction** in fixed-camera videos. This project combines **YOLO26s** for pedestrian detection, the original **DeepSORT** for identity-aware tracking, and a custom **Extended Kalman Filter (EKF)** for 12-frame trajectory forecasting.

The pipeline is evaluated on the **VIRAT Video Dataset**, using ground-truth annotations to measure prediction accuracy.

## Overview

```text
VIRAT Video
    |
    v
YOLO26s (Pedestrian Detection)
    |
    v
DeepSORT (Multi-Object Tracking)
    |
    v
Bounding-Box Bottom Center
    |
    v
Homography (Image -> Ground Plane)
    |
    v
EKF (State Estimation -> 12-Frame Prediction)
    |
    +--> Inverse Homography -> Video Visualization
    |
    +--> Ground-Truth Comparison -> ADE / FDE
```

The EKF maintains an independent motion state for each confirmed track:

\[
\mathbf{x}_k = [p_x,\ p_y,\ v,\ \theta,\ \omega]^T
\]

where \(p_x, p_y\) denote ground-plane position, \(v\) is speed, \(\theta\) is heading, and \(\omega\) is angular velocity. The motion model assumes approximately constant speed and turn rate over a short prediction horizon. State propagation uses **one video frame per time step**.

## Features

- Pedestrian detection using YOLO26s.
- Multi-object tracking with the original DeepSORT implementation and MARS appearance features.
- Homography-based transformation from image coordinates to the VIRAT scene ground plane.
- Per-track EKF state estimation and **12-frame future trajectory prediction**.
- Visualization of tracked pedestrians, predicted trajectories, and ground truth.
- Quantitative evaluation using **Average Displacement Error (ADE)**, **Final Displacement Error (FDE)**, and processing speed.
- Comparison against a linear trajectory extrapolation baseline.

## Installation

Python **3.10** is recommended. The project supports CUDA inference for YOLO; CPU inference is also available at a lower processing speed.

```bash
git clone https://github.com/smChu/trajectory_prediction.git
cd trajectory_prediction

python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

To check the installation:

```bash
python -m pip check
python -B -m unittest discover -s src -t . -p 'test_*.py'
```

## Dataset and Pretrained Models

Download the VIRAT video clips, their matching original object annotations, and the provided scene homographies. Prepare the YOLO26s and MARS pretrained weights separately; large datasets and model files are not included in this repository.

Expected directory layout:

```text
trajectory_prediction/
├── Resources/
│   ├── videos_original/
│   │   └── VIRAT_S_010208_07_000768_000791.mp4
│   ├── annotations/
│   │   └── VIRAT_S_010208_07_000768_000791.viratdata.objects.txt
│   └── homographies/
├── weights/
│   ├── yolo26s.pt
│   └── mars-small128.pb
├── deep_sort/
├── src/
└── requirements.txt
```

**Important:** Video filenames and original image resolution must match the corresponding annotation and scene homography files. Ground truth is used for offline calibration, visualization, and evaluation—not as an EKF observation during prediction.

## Quick Start

### 1. Calibrate the observation noise

The EKF uses an observation-noise covariance matrix \(R\) calibrated offline using a separate development clip and its ground-truth annotations.

```bash
python src/estimate_r.py \
    --source Resources/videos_original/VIRAT_S_010201_00_000000_000053.mp4 \
    --device 0 --imgsz 960
```

Use a development recording from the **same scene** as the evaluation recording, but keep calibration and evaluation recordings separate. Calibrated files are saved under `runs/calibration/` and selected automatically when their settings match the current run.

For a preliminary functionality test without calibrated \(R\), the `--ekf-r VAR_X VAR_Y` option accepts provisional variances. Such runs should not be treated as calibrated evaluation results.

### 2. Run tracking and trajectory prediction

```bash
python "src/tracking system.py" \
    --source Resources/videos_original/VIRAT_S_010208_07_000768_000791.mp4 \
    --device 0
```

Use `--device cpu` for CPU inference. The command displays the annotated video and saves the output to `runs/tracking/`.

### 3. Run quantitative evaluation

```bash
python "src/tracking system.py" \
    --source Resources/videos_original/VIRAT_S_010208_07_000768_000791.mp4 \
    --device 0 \
    --evaluate \
    --measure-predictor-latency
```

For headless evaluation without displaying or saving a video, append `--no-display --no-save`. Evaluation reports are written to `runs/evaluation/` in JSON format.

## Evaluation

We report the following metrics over valid 12-frame prediction windows:

| Metric | Description |
| --- | --- |
| **ADE** | Mean Euclidean distance between predicted and ground-truth positions across the prediction horizon. |
| **FDE** | Euclidean distance between predicted and ground-truth positions at the final prediction step. |
| **Processing FPS** | End-to-end video processing speed. |
| **Predictor latency** | EKF update and future prediction time, excluding detection, tracking, and rendering. |

A linear extrapolation baseline is provided for comparison. For multi-recording experiments, prepare independent splits and run:

```bash
python src/evaluate.py --prepare-split --all-scenes
python src/evaluate.py --split validation --device 0
python src/evaluate.py --split test --device 0
```

**Evaluation notes:** Ground-plane distances are expressed in the **native units of each VIRAT scene homography**, which are not automatically guaranteed to be meters. Distance results from different scenes should not be pooled without a verified common scale. Only windows with valid tracking-to-ground-truth associations and complete future ground truth are scored.

## Repository Structure

```text
src/
├── tracking system.py   # Main entry point
├── track.py             # Video processing and evaluation pipeline
├── tracking_core.py     # YOLO26s / MARS / DeepSORT integration
├── ekf.py               # EKF state estimation and forecasting
├── homography.py        # Image-to-ground coordinate transformation
├── forecast.py          # Trajectory visualization and GT association
├── estimate_r.py        # Offline observation-noise calibration
├── evaluate.py          # ADE/FDE and dataset split evaluation
├── latency.py           # Prediction latency measurement
└── test_*.py            # Unit tests
```

## References

- N. Wojke, A. Bewley, and D. Paulus, *Simple Online and Realtime Tracking with a Deep Association Metric*, ICIP, 2017. [Paper](https://arxiv.org/abs/1703.07402) | [Code](https://github.com/nwojke/deep_sort)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [VIRAT Video Dataset](https://viratdata.org/)

The DeepSORT tracker and MARS feature encoder are based on the original open-source implementation by Wojke et al. The EKF trajectory-prediction module and evaluation pipeline are implemented in this repository.

## License

This project is released under the [MIT License](LICENSE). See the included `deep_sort/LICENSE` for the license of the upstream tracking component.
