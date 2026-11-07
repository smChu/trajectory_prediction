"""Verify EKF latency warmup, units and missing-forecast accounting."""

import unittest

from src.latency import PredictorLatency


class LatencyTests(unittest.TestCase):
    def test_warmup_and_nonforecast_frames_are_reported_separately(self):
        timer = PredictorLatency(("ekf",), warmup_frames=1)
        timer.record(0, {"ekf": 9.}, {"ekf": [1]}, 1)
        timer.record(1, {"ekf": .002}, {"ekf": []}, 1)
        timer.record(2, {"ekf": .004}, {"ekf": [1, 2]}, 2)
        report = timer.report({"ekf": "cpu"})
        self.assertEqual(report["measured_frames"], 2)
        self.assertEqual(report["matched_forecast_frames"], 1)
        self.assertEqual(report["methods"]["ekf"]["all_frames"]["mean_ms"], 3.)
        forecast = report["methods"]["ekf"]["matched_forecast_frames"]
        self.assertEqual(forecast["mean_ms"], 4.)
        self.assertEqual(forecast["amortized_ms_per_forecast_track"], 2.)

    def test_empty_latency_is_null_and_bad_samples_are_rejected(self):
        timer = PredictorLatency(("ekf",))
        self.assertIsNone(timer.report({"ekf": "cpu"})["methods"]["ekf"]["all_frames"]["mean_ms"])
        for seconds in [0, -1, float("nan")]:
            with self.assertRaises(ValueError):
                timer.record(0, {"ekf": seconds}, {"ekf": []}, 0)
        with self.assertRaises(ValueError):
            timer.record(1, {"ekf": .001}, {"ekf": []}, 0)


if __name__ == "__main__":
    unittest.main()
