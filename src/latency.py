"""Prediction-only wall-clock statistics, with matched forecast workloads."""

import numpy as np


def summarize_latency(samples_ms, forecast_counts):
    values = np.asarray(samples_ms, dtype=float)
    if not len(values):
        return dict(frames=0, mean_ms=None, median_ms=None, p95_ms=None,
                    total_ms=0.0, forecast_tracks=0, amortized_ms_per_forecast_track=None)
    forecasts = sum(forecast_counts)
    return dict(frames=len(values), mean_ms=float(values.mean()),
                median_ms=float(np.median(values)), p95_ms=float(np.percentile(values, 95)),
                total_ms=float(values.sum()), forecast_tracks=forecasts,
                amortized_ms_per_forecast_track=float(values.sum() / forecasts) if forecasts else None)


class PredictorLatency:
    def __init__(self, methods, warmup_frames=50):
        if warmup_frames < 0 or not methods or len(set(methods)) != len(methods):
            raise ValueError("Need unique methods and a nonnegative latency warmup")
        self.methods = tuple(methods)
        self.warmup_frames = warmup_frames
        self.seen = 0
        self.samples = []

    def record(self, frame_idx, seconds, forecast_ids, active_tracks):
        if frame_idx != self.seen:
            raise ValueError("Latency requires consecutive frames starting at zero")
        if set(seconds) != set(self.methods) or set(forecast_ids) != set(self.methods):
            raise ValueError("Each method must have a duration and forecast IDs")
        if any(not np.isfinite(value) or value <= 0 for value in seconds.values()):
            raise ValueError("Prediction durations must be positive and finite")
        self.seen += 1
        if frame_idx < self.warmup_frames:
            return
        ids = {name: sorted(set(forecast_ids[name])) for name in self.methods}
        first_ids = ids[self.methods[0]]
        matched = bool(first_ids) and all(ids[name] == first_ids for name in self.methods)
        self.samples.append(dict(frame_idx=frame_idx, active_tracks=active_tracks,
                                 elapsed_ms={name: seconds[name] * 1000 for name in self.methods},
                                 forecast_ids=ids, matched_forecast_workload=matched))

    def report(self, devices):
        paired = [row for row in self.samples if row["matched_forecast_workload"]]
        methods = {}
        for name in self.methods:
            methods[name] = dict(device=devices[name])
            for label, rows in (("all_frames", self.samples), ("matched_forecast_frames", paired)):
                methods[name][label] = summarize_latency(
                    [row["elapsed_ms"][name] for row in rows],
                    [len(row["forecast_ids"][name]) for row in rows])
        return dict(
            scope="EKF state update plus 12-frame forecasting and covariance checks; excludes detection, "
                  "tracking, GT evaluation, overlays, logging and file/display IO",
            timing="wall clock on CPU; excludes model loading",
            units="milliseconds per video frame for all active tracks, not per person or per predicted point",
            warmup_frames_excluded=min(self.seen, self.warmup_frames),
            measured_frames=len(self.samples), matched_forecast_frames=len(paired),
            matching="after warmup, every method returns forecasts for exactly the same nonempty set of track IDs; "
                     "state/history updates for all active IDs remain included",
            methods=methods, samples=self.samples,
        )
