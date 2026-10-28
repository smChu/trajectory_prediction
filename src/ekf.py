"""Independent, frame-time EKFs for measured DeepSORT ground positions.

State: [px, py, speed, heading, angular_speed]. Positions use the unchanged
VIRAT scene units; speed is scene units/frame and angular speed rad/frame.
Only the last nine actual observations are retained for initialization.
"""

from collections import deque

import numpy as np


DEFAULT_Q = (1e-3, 1e-3, 1e-4, 1e-4, 1e-5)
INITIALIZATION_OBSERVATIONS = 9
OBSERVATION_MATRIX = np.eye(2, 5)


def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def transition(state, dt=1.0):
    px, py, speed, heading, angular_speed = state
    return np.array([
        px + speed * dt * np.cos(heading),
        py + speed * dt * np.sin(heading),
        speed, wrap_angle(heading + angular_speed * dt), angular_speed,
    ], dtype=np.float64)


def transition_jacobian(state, dt=1.0):
    speed, heading = state[2:4]
    return np.array([
        [1, 0, dt * np.cos(heading), -speed * dt * np.sin(heading), 0],
        [0, 1, dt * np.sin(heading), speed * dt * np.cos(heading), 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, dt],
        [0, 0, 0, 0, 1],
    ], dtype=np.float64)


def diagonal(values, size, name, allow_zero=False):
    values = np.asarray(values, dtype=np.float64)
    if (values.shape != (size,) or not np.isfinite(values).all()
            or np.any(values < 0 if allow_zero else values <= 0)):
        raise ValueError(f"{name} needs {size} finite {'nonnegative' if allow_zero else 'positive'} variances")
    return np.diag(values)


class PedestrianEKF:
    """Nine measured positions bootstrap motion; nine-frame warmup is separate.

    A stationary heading/angular speed is unobservable from positions. Its zero
    placeholder is marked unknown and retains large P0 uncertainty. When real
    motion first makes that component observable, rebootstrap conservatively
    from measured differences, never from predicted or GT positions.
    """

    def __init__(self, position, frame_idx, r, q=DEFAULT_Q, p0=None,
                 warmup_frames=9, min_speed=0.01):
        self.R = diagonal(r, 2, "R")
        self.Q = diagonal(q, 5, "Q", allow_zero=True)
        self.P0 = diagonal(
            p0 if p0 is not None else [*np.diag(self.R), 1.0, np.pi**2, 0.25],
            5, "P0",
        )
        if warmup_frames < 1 or not np.isfinite(min_speed) or min_speed <= 0:
            raise ValueError("warmup_frames and min_speed must be positive")
        self.warmup_frames = int(warmup_frames)
        self.min_speed = float(min_speed)
        self.x = np.zeros(5, dtype=np.float64)
        self.x[:2] = self._position(position)
        self.P = self.P0.copy()
        self.frame_idx = int(frame_idx)
        self.last_measurement_frame = int(frame_idx)
        self.samples = deque([(int(frame_idx), self.x[:2].copy())],
                             maxlen=INITIALIZATION_OBSERVATIONS)
        self.initialized_at = None
        self.heading_observed = False
        self.angular_speed_observed = False

    @staticmethod
    def _position(position):
        position = np.asarray(position, dtype=np.float64)
        if position.shape != (2,) or not np.isfinite(position).all():
            raise ValueError("Measurement must be a finite ground position (px, py)")
        return position

    def predict_to(self, frame_idx):
        if int(frame_idx) != frame_idx or frame_idx <= self.frame_idx:
            raise ValueError("EKF frames must increase strictly")
        # Repeat the proposal's one-frame Euler step, including one Q per step.
        # This also handles skipped frames without undercounting process noise.
        for _ in range(int(frame_idx) - self.frame_idx):
            jacobian = transition_jacobian(self.x)
            self.x = transition(self.x)
            self.P = jacobian @ self.P @ jacobian.T + self.Q
            self.P = (self.P + self.P.T) / 2
        self.frame_idx = int(frame_idx)

    def _bootstrap(self):
        samples = list(self.samples)
        velocities, center_frames = [], []
        # The center observation is shared: positions 1..5 and 5..9. Fit
        # position against actual frame indices in each group to reduce box
        # jitter; missing frames are never interpolated into measurements.
        for group in (samples[:5], samples[4:]):
            frames = np.array([frame for frame, _ in group], dtype=float)
            positions = np.array([position for _, position in group])
            centered_frames = frames - frames.mean()
            centered_positions = positions - positions.mean(axis=0)
            velocities.append(centered_frames @ centered_positions / (centered_frames @ centered_frames))
            center_frames.append(float(frames.mean()))
        speeds = [float(np.linalg.norm(v)) for v in velocities]
        moving = [speed >= self.min_speed for speed in speeds]
        headings = [float(np.arctan2(v[1], v[0])) if is_moving else 0.0
                    for v, is_moving in zip(velocities, moving)]
        heading_known = moving[1]
        omega_known = all(moving)
        if (self.initialized_at is not None
                and not (heading_known and not self.heading_observed)
                and not (omega_known and not self.angular_speed_observed)):
            return False
        # Headings describe their fitted segments, so use the difference of
        # the segment-center frames, rather than assuming a one-frame gap.
        omega = wrap_angle(headings[1] - headings[0]) / (center_frames[1] - center_frames[0]) if omega_known else 0.0
        self.x = np.array([*samples[-1][1], speeds[1] if moving[1] else 0.0, headings[1], omega])
        self.P = self.P0.copy()
        self.heading_observed = heading_known
        self.angular_speed_observed = omega_known
        self.initialized_at = self.frame_idx
        return True

    def update(self, position):
        position = self._position(position)
        if self.last_measurement_frame == self.frame_idx:
            raise ValueError("Duplicate observation for this frame")
        self.last_measurement_frame = self.frame_idx
        self.samples.append((self.frame_idx, position.copy()))
        if len(self.samples) < INITIALIZATION_OBSERVATIONS:
            # Position is measured; motion waits for nine actual observations.
            self.x[:2] = position
            self.P = self.P0.copy()
            return
        if self._bootstrap():
            return
        h = OBSERVATION_MATRIX
        innovation = position - h @ self.x
        s = h @ self.P @ h.T + self.R
        gain = np.linalg.solve(s, h @ self.P).T
        self.x += gain @ innovation
        residual = np.eye(5) - gain @ h
        self.P = residual @ self.P @ residual.T + gain @ self.R @ gain.T
        # Canonical nonnegative speed represents the same physical velocity.
        if self.x[2] < 0:
            self.x[2] *= -1
            self.x[3] += np.pi
            sign = np.diag([1, 1, -1, 1, 1])
            self.P = sign @ self.P @ sign
        self.x[3] = wrap_angle(self.x[3])
        self.P = (self.P + self.P.T) / 2
        if not np.isfinite(self.x).all() or not np.isfinite(self.P).all():
            raise FloatingPointError("Non-finite EKF state or covariance")

    def predict_future(self):
        """Return k+1 .. k+12 from copies, without updates or live mutations.

        Reuse the existing elapsed-frame warmup condition. This condition is a
        preparation flag, not evidence of convergence or calibrated coverage.
        Results are retained by the caller for the current frame only.
        """
        if self.initialized_at is None or self.frame_idx - self.initialized_at + 1 < self.warmup_frames:
            return []
        state, covariance = self.x.copy(), self.P.copy()
        if not np.isfinite(state).all() or not np.isfinite(covariance).all():
            raise ValueError("Non-finite current state/covariance for forecast")
        scale = max(np.max(np.abs(covariance)), np.finfo(float).tiny)
        if np.max(np.abs(covariance - covariance.T)) > 64 * np.finfo(float).eps * scale:
            raise ValueError("Asymmetric current covariance for forecast")
        eigenvalues = np.linalg.eigvalsh(covariance)
        tolerance = 64 * np.finfo(float).eps * max(np.max(np.abs(eigenvalues)), np.finfo(float).tiny)
        if eigenvalues[0] < -tolerance:
            raise ValueError(f"Non-PSD current covariance for forecast: {eigenvalues[0]}")
        predictions = []
        for offset in range(1, 13):
            jacobian = transition_jacobian(state)
            state = transition(state)
            covariance = jacobian @ covariance @ jacobian.T + self.Q
            covariance = (covariance + covariance.T) / 2
            if not np.isfinite(state).all() or not np.isfinite(covariance).all():
                raise ValueError(f"Non-finite forecast at +{offset} frames")
            eigenvalues = np.linalg.eigvalsh(covariance)
            tolerance = 64 * np.finfo(float).eps * max(np.max(np.abs(eigenvalues)), np.finfo(float).tiny)
            if eigenvalues[0] < -tolerance:
                raise ValueError(f"Non-PSD forecast covariance at +{offset}: {eigenvalues[0]}")
            predictions.append({
                "frame_idx": self.frame_idx + offset, "offset": offset,
                "predicted_state": state.tolist(),
                "predicted_covariance": covariance.tolist(),
                "predicted_position": state[:2].tolist(),
            })
        return predictions

    def result(self, track_id):
        initialized = self.initialized_at is not None
        elapsed = self.frame_idx - self.initialized_at + 1 if initialized else 0
        return {
            "frame_idx": self.frame_idx, "track_id": int(track_id),
            "estimated_px": float(self.x[0]), "estimated_py": float(self.x[1]),
            "estimated_v": float(self.x[2]) if initialized else None,
            "estimated_theta": float(self.x[3]) if self.heading_observed else None,
            "estimated_omega": float(self.x[4]) if self.angular_speed_observed else None,
            "P": self.P.tolist(), "initialized": initialized,
            "has_measurement": self.last_measurement_frame == self.frame_idx,
            "last_measurement_frame": self.last_measurement_frame,
            "initialization_observations": len(self.samples),
            "warmup_elapsed_frames": elapsed,
            "warmup_complete": initialized and elapsed >= self.warmup_frames,
        }


class TrackEKFs:
    """Own EKFs by DeepSORT ID, expire them with the original track lifecycle."""

    def __init__(self, r, q=DEFAULT_Q, p0=None, warmup_frames=9, min_speed=0.01):
        self.options = dict(r=r, q=q, p0=p0, warmup_frames=warmup_frames, min_speed=min_speed)
        # Validate settings before opening a video, even if it has no people.
        PedestrianEKF((0, 0), 0, **self.options)
        self.filters = {}
        self.last_frame = -1

    def step(self, frame_idx, observations, active_track_ids):
        if frame_idx <= self.last_frame:
            raise ValueError("Frames must increase strictly")
        self.last_frame = frame_idx
        active = set(active_track_ids)
        self.filters = {tid: ekf for tid, ekf in self.filters.items() if tid in active}
        measurements = {}
        for observation in observations:
            tid = observation["track_id"]
            if observation["frame_idx"] != frame_idx or tid not in active or tid in measurements:
                raise ValueError("Observation frame/track mismatch or duplicate")
            measurements[tid] = observation["ground_position"]
        for tid, ekf in self.filters.items():
            ekf.predict_to(frame_idx)
            if tid in measurements:
                ekf.update(measurements[tid])
        for tid, position in measurements.items():
            if tid not in self.filters:
                self.filters[tid] = PedestrianEKF(position, frame_idx, **self.options)
        return [self.filters[tid].result(tid) for tid in sorted(self.filters)]
