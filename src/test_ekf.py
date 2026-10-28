"""Run with: .venv/bin/python -B -m unittest discover -s src -p test_ekf.py"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from ekf import PedestrianEKF, TrackEKFs, transition, transition_jacobian, wrap_angle
from estimate_r import iou_matrix, unambiguous_matches
from forecast import (CHI2_95_2D, collect_forecasts, draw_forecasts,
                      ground_to_pixels, project_ellipse, uncertainty_ellipse)


class EKFTests(unittest.TestCase):
    def test_proposal_transition_and_finite_difference_jacobian(self):
        x = np.array([4.0, -2.0, 0.7, 0.4, -0.03])
        dt = 2.0
        expected = [4 + 1.4 * np.cos(.4), -2 + 1.4 * np.sin(.4), .7, .34, -.03]
        np.testing.assert_allclose(transition(x, dt), expected)
        epsilon = 1e-6
        numerical = np.column_stack([
            (transition(x + np.eye(5)[i] * epsilon, dt)
             - transition(x - np.eye(5)[i] * epsilon, dt)) / (2 * epsilon)
            for i in range(5)
        ])
        np.testing.assert_allclose(transition_jacobian(x, dt), numerical, atol=1e-8)

    def test_nine_measurements_then_separate_nine_frame_warmup(self):
        ekf = PedestrianEKF([0, 0], 0, r=[.02, .03])
        self.assertIsNone(ekf.result(1)["estimated_v"])
        for frame in range(1, 21):
            ekf.predict_to(frame)
            ekf.update([frame * .1, 0])
            state = ekf.result(1)
            self.assertEqual(state["initialized"], frame >= 8)
            self.assertEqual(state["warmup_complete"], frame >= 16)
        self.assertGreater(ekf.result(1)["warmup_elapsed_frames"], 9)
        np.testing.assert_allclose(ekf.x, [2, 0, .1, 0, 0], atol=1e-10)

    def test_gap_bootstrap_uses_actual_frame_intervals(self):
        ekf = PedestrianEKF([0, 0], 0, r=[.02, .03])
        frames = [1, 3, 4, 7, 8, 11, 12, 15]
        for frame in frames:
            ekf.predict_to(frame)
            ekf.update([.2 * frame, 0])
            self.assertEqual(ekf.result(1)["initialized"], frame == 15)
        self.assertAlmostEqual(ekf.x[2], .2)
        self.assertAlmostEqual(ekf.x[4], 0)
        before = ekf.P.copy()
        ekf.predict_to(19)
        self.assertFalse(ekf.result(1)["has_measurement"])
        self.assertEqual(ekf.result(1)["last_measurement_frame"], 15)
        self.assertGreater(np.trace(ekf.P), np.trace(before))

    def test_frame_jump_matches_repeated_one_frame_predictions(self):
        first = PedestrianEKF([0, 0], 0, r=[.02, .03])
        second = PedestrianEKF([0, 0], 0, r=[.02, .03])
        first.x[:] = second.x[:] = [1, 2, .3, .6, .04]
        first.predict_to(5)
        for frame in range(1, 6):
            second.predict_to(frame)
        np.testing.assert_allclose(first.x, second.x)
        np.testing.assert_allclose(first.P, second.P)

    def test_stationary_initialization_then_vertical_motion(self):
        ekf = PedestrianEKF([0, 0], 0, r=[.02, .03])
        for frame in range(1, 9):
            ekf.predict_to(frame)
            ekf.update([0, 0])
        self.assertEqual(ekf.result(1)["estimated_v"], 0)
        self.assertIsNone(ekf.result(1)["estimated_theta"])
        self.assertIsNone(ekf.result(1)["estimated_omega"])
        for frame in range(9, 17):
            ekf.predict_to(frame)
            ekf.update([0, (frame - 8) * .2])
        self.assertAlmostEqual(ekf.x[2], .2)
        self.assertAlmostEqual(ekf.x[3], np.pi / 2)
        self.assertAlmostEqual(ekf.x[4], 0)
        self.assertTrue(ekf.heading_observed)
        self.assertTrue(ekf.angular_speed_observed)

    def test_heading_wrap_when_bootstrapping_angular_speed(self):
        headings = np.deg2rad([179, -179])
        ekf = PedestrianEKF([0, 0], 0, r=[.02, .03])
        position = np.zeros(2)
        for frame in range(1, 9):
            heading = headings[0 if frame <= 4 else 1]
            position += [np.cos(heading), np.sin(heading)]
            ekf.predict_to(frame)
            ekf.update(position)
        self.assertAlmostEqual(ekf.x[4], np.deg2rad(2) / 4)
        self.assertAlmostEqual(wrap_angle(3 * np.pi), -np.pi)

    def test_joseph_covariance_psd_under_repeated_noisy_updates(self):
        rng = np.random.default_rng(42)
        ekf = PedestrianEKF([0, 0], 0, r=[.002, .003])
        for frame in range(1, 300):
            ekf.predict_to(frame)
            if frame % 7 != 0:
                ekf.update([frame * .08 + rng.normal(0, .04), rng.normal(0, .05)])
            self.assertTrue(np.isfinite(ekf.x).all())
            np.testing.assert_allclose(ekf.P, ekf.P.T, atol=1e-12)
            self.assertGreaterEqual(np.linalg.eigvalsh(ekf.P).min(), -1e-10)
            self.assertGreaterEqual(ekf.x[2], 0)
            self.assertTrue(-np.pi <= ekf.x[3] < np.pi)

    def test_manager_separates_tracks_predicts_misses_and_deletes_expired(self):
        manager = TrackEKFs([.02, .03])
        def observation(frame, tid, position):
            return {"frame_idx": frame, "track_id": tid, "ground_position": position}
        for frame in range(9):
            manager.step(frame, [observation(frame, 1, [frame * .1, 0]),
                                 observation(frame, 9, [50, frame * .2])], [1, 9, 25])
        self.assertEqual(set(manager.filters), {1, 9})
        states = manager.step(9, [observation(9, 9, [50, 1.8])], [1, 9])
        self.assertFalse(states[0]["has_measurement"])
        self.assertTrue(states[1]["has_measurement"])
        self.assertGreater(states[1]["estimated_px"], 40)
        manager.step(10, [], [9])
        self.assertEqual(set(manager.filters), {9})
        with self.assertRaises(ValueError):
            manager.step(10, [], [9])

    def test_invalid_covariance_and_measurements_rejected(self):
        for r in ([0, .1], [-1, .1], [np.nan, 1]):
            with self.assertRaises(ValueError):
                PedestrianEKF([0, 0], 0, r=r)
        with self.assertRaises(ValueError):
            PedestrianEKF([np.inf, 0], 0, r=[1, 1])

    def test_initialization_fits_reduce_first_frame_direction_jitter(self):
        positions = [[0, 0], [-.005, .013], [.015, .017], [.05, .015],
                     [.11, .009], [.18, .001], [.25, -.01], [.30, -.015], [.35, -.04]]
        ekf = PedestrianEKF(positions[0], 0, r=[.02, .03])
        for frame, position in enumerate(positions[1:], 1):
            ekf.predict_to(frame)
            ekf.update(position)
        self.assertEqual(ekf.initialized_at, 8)
        self.assertEqual(len(ekf.samples), 9)
        np.testing.assert_allclose(ekf.x[:2], positions[-1])
        self.assertGreater(ekf.x[2], .04)
        self.assertLess(abs(ekf.x[4]), .1)
        self.assertFalse(ekf.result(1)["warmup_complete"])


class AssociationTests(unittest.TestCase):
    def test_unique_matches_do_not_assume_numeric_ids(self):
        a, b = [0, 0, 10, 20], [30, 0, 10, 20]
        np.testing.assert_allclose(iou_matrix([a, b], [b, a]), [[0, 1], [1, 0]])
        self.assertEqual(unambiguous_matches([a, b], [b, a]), [(0, 1, 1), (1, 0, 1)])

    def test_ambiguous_or_low_overlap_matches_rejected(self):
        box = [0, 0, 10, 20]
        self.assertEqual(unambiguous_matches([box], [box, box]), [])
        self.assertEqual(unambiguous_matches([box, box], [box]), [])
        self.assertEqual(unambiguous_matches([box], [[9, 0, 10, 20]]), [])
        self.assertEqual(unambiguous_matches([], [box]), [])


class ForecastTests(unittest.TestCase):
    @staticmethod
    def ready_filter():
        ekf = PedestrianEKF([0, 0], 0, r=[.02, .03])
        for frame in range(1, 17):
            ekf.predict_to(frame)
            ekf.update([frame * .1, 0])
        return ekf

    def test_twelve_sequential_turning_steps_and_covariances(self):
        ekf = self.ready_filter()
        ekf.x[:] = [1, 2, .5, .2, .1]
        future = ekf.predict_future()
        self.assertEqual(len(future), 12)
        angles = .2 + .1 * np.arange(12)
        expected_positions = [1, 2] + np.cumsum(.5 * np.column_stack((np.cos(angles), np.sin(angles))), axis=0)
        np.testing.assert_allclose([p["predicted_position"] for p in future], expected_positions)
        # Independently reuse the established live one-frame predictor on a
        # second filter; a frozen Jacobian or extra update would disagree.
        reference = self.ready_filter()
        reference.x = ekf.x.copy()
        reference.P = ekf.P.copy()
        for offset, prediction in enumerate(future, 1):
            reference.predict_to(ekf.frame_idx + offset)
            self.assertEqual(prediction["offset"], offset)
            self.assertEqual(prediction["frame_idx"], ekf.frame_idx + offset)
            self.assertEqual(np.shape(prediction["predicted_state"]), (5,))
            self.assertEqual(np.shape(prediction["predicted_covariance"]), (5, 5))
            np.testing.assert_allclose(prediction["predicted_state"], reference.x)
            np.testing.assert_allclose(prediction["predicted_covariance"], reference.P)

    def test_future_is_detached_and_never_uses_r_or_update(self):
        ekf = self.ready_filter()
        state, covariance, q = ekf.x.copy(), ekf.P.copy(), ekf.Q.copy()
        frame, last, initialized = ekf.frame_idx, ekf.last_measurement_frame, ekf.initialized_at
        samples = [(f, z.copy()) for f, z in ekf.samples]
        ekf.R[:] = np.nan  # R cannot influence prediction-only propagation.
        with patch.object(ekf, "update", side_effect=AssertionError("Future update forbidden")):
            future = ekf.predict_future()
        future[0]["predicted_state"][0] = 99999
        future[0]["predicted_covariance"][0][0] = 99999
        np.testing.assert_array_equal(ekf.x, state)
        np.testing.assert_array_equal(ekf.P, covariance)
        np.testing.assert_array_equal(ekf.Q, q)
        self.assertEqual((ekf.frame_idx, ekf.last_measurement_frame, ekf.initialized_at), (frame, last, initialized))
        for (f0, z0), (f1, z1) in zip(samples, ekf.samples):
            self.assertEqual(f0, f1)
            np.testing.assert_array_equal(z0, z1)
        self.assertNotEqual(future[1]["predicted_state"][0], 99999)

    def test_warmup_gates_future_and_estimation_continues(self):
        ekf = PedestrianEKF([0, 0], 0, r=[.02, .03])
        self.assertEqual(ekf.predict_future(), [])
        for frame in range(1, 19):
            ekf.predict_to(frame)
            ekf.update([frame * .1, 0])
            self.assertEqual(len(ekf.predict_future()), 12 if frame >= 16 else 0)

    def test_independent_ids_and_original_display_age_gate(self):
        a, b = self.ready_filter(), self.ready_filter()
        b.x[0] += 50
        manager = SimpleNamespace(filters={1: a, 8: b})
        def track(tid, age=0, confirmed=True):
            return SimpleNamespace(track_id=tid, time_since_update=age, is_confirmed=lambda: confirmed)
        future = collect_forecasts(manager, [track(1), track(8)])
        self.assertEqual([f["track_id"] for f in future], [1, 8])
        self.assertAlmostEqual(future[1]["predictions"][0]["predicted_position"][0]
                               - future[0]["predictions"][0]["predicted_position"][0], 50)
        a.predict_to(a.frame_idx + 1)
        missed = collect_forecasts(manager, [track(1, 1)])[0]
        self.assertFalse(missed["has_measurement"])
        self.assertEqual(collect_forecasts(manager, [track(1, 2), track(8, 0, False), track(99)]), [])
        del manager.filters[1]
        self.assertEqual(collect_forecasts(manager, [track(1)]), [])

    def test_invalid_forecast_covariance_is_reported_without_replacement(self):
        ekf = self.ready_filter()
        ekf.P[0, 0] = -100
        with self.assertRaisesRegex(ValueError, "Non-PSD"):
            ekf.predict_future()
        track = SimpleNamespace(track_id=7, time_since_update=0, is_confirmed=lambda: True)
        with self.assertWarnsRegex(UserWarning, "track_id=7.*Non-PSD"):
            self.assertEqual(collect_forecasts(SimpleNamespace(filters={7: ekf}), [track]), [])
        ekf = self.ready_filter()
        ekf.P[0, 1] += .1
        with self.assertRaisesRegex(ValueError, "Asymmetric"):
            ekf.predict_future()


class EllipseTests(unittest.TestCase):
    def test_correlated_covariance_axes_and_boundary_probability_level(self):
        covariance = np.array([[2.5, 1.5], [1.5, 2.5]])
        ellipse = uncertainty_ellipse([4, -2], covariance, 64)
        np.testing.assert_allclose(ellipse["semiaxes"], np.sqrt(CHI2_95_2D * np.array([4, 1])))
        self.assertAlmostEqual(np.cos(2 * ellipse["angle_rad"]), 0)
        self.assertAlmostEqual(np.sin(2 * ellipse["angle_rad"]), 1)
        residual = ellipse["boundary"] - [4, -2]
        mahalanobis = np.einsum('ni,in->n', residual, np.linalg.solve(covariance, residual.T))
        np.testing.assert_allclose(mahalanobis, CHI2_95_2D, atol=1e-12)

    def test_only_roundoff_negative_eigenvalues_are_clamped(self):
        ellipse = uncertainty_ellipse([0, 0], np.diag([1, -np.finfo(float).eps]))
        self.assertEqual(ellipse["semiaxes"][1], 0)
        for covariance in (np.diag([1, -1e-8]), [[1, .1], [.2, 1]], [[np.nan, 0], [0, 1]]):
            with self.assertRaises(ValueError):
                uncertainty_ellipse([0, 0], covariance)

    def test_perspective_boundary_projection_uses_every_ground_point(self):
        ellipse = uncertainty_ellipse([2, 3], [[.05, .02], [.02, .03]], 48)
        inverse = np.array([[100, 0, 200], [0, 80, 100], [.02, -.03, 1]])
        actual = project_ellipse(ellipse, inverse)
        x, y = ellipse["boundary"].T
        expected = np.column_stack(((100 * x + 200) / (.02 * x - .03 * y + 1),
                                    (80 * y + 100) / (.02 * x - .03 * y + 1)))
        np.testing.assert_allclose(actual, expected)
        restored, valid, _ = ground_to_pixels(actual, np.linalg.inv(inverse))
        self.assertTrue(valid.all())
        np.testing.assert_allclose(restored, ellipse["boundary"])

    def test_projective_pole_is_rejected_not_joined_across_infinity(self):
        inverse = np.array([[1., 0, 1], [0, 1, 0], [1, 0, 0]])
        ellipse = uncertainty_ellipse([0, 0], np.eye(2))
        with self.assertRaisesRegex(ValueError, "pole"):
            project_ellipse(ellipse, inverse)
        pixels, valid, _ = ground_to_pixels([[0, 0], [1, 0]], inverse)
        self.assertFalse(valid[0])
        self.assertTrue(np.isnan(pixels[0]).all())
        self.assertTrue(valid[1])

    def test_twelve_separate_ellipses_and_points_are_drawn(self):
        ekf = ForecastTests.ready_filter()
        ekf.x[:2] = [100, 100]
        ekf.P = np.diag([.01, .01, .001, .001, .001])
        packet = {"frame_idx": ekf.frame_idx, "track_id": 5,
                  "current_position": [100, 100], "predictions": ekf.predict_future()}
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        counts = draw_forecasts(frame, [packet], np.eye(3), 48, 1)
        self.assertEqual(counts, {"tracks": 1, "predictions": 12, "ellipses": 12, "skipped_ellipses": 0})
        self.assertFalse(np.any(np.all(frame == [255, 255, 0], axis=2)))
        self.assertTrue(np.any(np.all(frame == [0, 255, 255], axis=2)))
        self.assertGreater(np.count_nonzero(frame), 0)


if __name__ == "__main__":
    unittest.main()
