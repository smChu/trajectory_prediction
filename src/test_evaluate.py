"""Metric correctness, exclusions and recording-level split isolation."""

from collections import deque
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from src.evaluate import (QuantitativeEvaluation, FrameProcessingSpeed,
                                  processing_speed_summary, prepare_split, save_json,
                                  split_groups, window_summary)
from src.homography import scene_from_source


class EvaluationTests(unittest.TestCase):
    def test_processing_fps_excludes_warmup_and_weights_total_elapsed_time(self):
        first = FrameProcessingSpeed(warmup_frames=2)
        for seconds in [9., 8., .1, .1]:
            first.record(seconds)
        self.assertEqual(first.report()["measured_frames"], 2)
        self.assertAlmostEqual(first.report()["processing_fps"], 10.)
        second = FrameProcessingSpeed(warmup_frames=0)
        second.record(.8)
        summary = processing_speed_summary([
            {"processing_speed": first.report()}, {"processing_speed": second.report()}])
        self.assertAlmostEqual(summary["processing_fps"], 3.)
        self.assertEqual(summary["measured_frames"], 3)

    def test_processing_fps_has_no_value_before_measured_frames(self):
        speed = FrameProcessingSpeed()
        speed.record(.5)
        self.assertIsNone(speed.report()["processing_fps"])
        self.assertIsNone(processing_speed_summary([{}])["processing_fps"])
        for value in [0, -1, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                speed.record(value)

    def test_full_recording_and_clip_names_select_only_their_documented_scene(self):
        self.assertEqual(scene_from_source("VIRAT_S_000006.mp4"), "0000")
        self.assertEqual(scene_from_source("VIRAT_S_000102.mp4"), "0001")
        self.assertEqual(scene_from_source("VIRAT_S_050300_05_001272_001330.mp4"), "0503")
        with self.assertRaises(ValueError): scene_from_source("VIRAT_S_000006_extra.mp4")

    def fixture(self, error=(3, 4), frames=20):
        gt = {f: [(7, np.array([90 + f, 80, 20, 20], dtype=float))] for f in range(frames)}
        packet = {"frame_idx": 0, "track_id": 1, "gt_id": 7, "has_measurement": True,
                  "predictions": [{"predicted_position": (np.array([100 + f, 100]) + error).tolist(),
                                   "predicted_covariance": np.eye(5).tolist()}
                                  for f in range(1, 13)]}
        samples = deque((f, np.array([100 + f, 100.])) for f in range(-8, 1))
        manager = SimpleNamespace(filters={1: SimpleNamespace(samples=samples)})
        evaluator = QuantitativeEvaluation(gt, np.eye(3), (400, 300), frames)
        return evaluator, packet, manager

    def start(self, evaluator, packet, manager):
        evaluator.step(0, [packet], {1: 7}, [], [], manager, [packet])

    def finish(self, evaluator, manager, last=12):
        for frame in range(1, last + 1):
            evaluator.step(frame, [], {}, [], [], manager, [])
        return evaluator.report({})

    def test_known_ade_fde_and_linear_baseline_without_future_or_live_mutation(self):
        evaluator, packet, manager = self.fixture()
        before = deepcopy(packet)
        samples = [(f, z.copy()) for f, z in manager.filters[1].samples]
        self.start(evaluator, packet, manager)
        report = self.finish(evaluator, manager)
        self.assertEqual(report["metrics"]["ekf"], {"ade": 5., "fde": 5.})
        self.assertEqual(report["metrics"]["linear"], {"ade": 0., "fde": 0.})
        self.assertEqual(report["ellipse_95_gt_coverage"], 0.)
        self.assertEqual(packet, before)
        for (f, z), (f1, z1) in zip(samples, manager.filters[1].samples):
            self.assertEqual(f, f1); np.testing.assert_array_equal(z, z1)

    def test_ade_uses_all_twelve_offsets_and_fde_only_last(self):
        evaluator, packet, manager = self.fixture(error=(0, 0))
        for offset, prediction in enumerate(packet["predictions"], 1):
            prediction["predicted_position"][0] += offset
        self.start(evaluator, packet, manager)
        report = self.finish(evaluator, manager)
        self.assertEqual(report["metrics"]["ekf"], {"ade": 6.5, "fde": 12.})

    def test_missing_gt_is_not_interpolated_or_partially_scored(self):
        evaluator, packet, manager = self.fixture()
        evaluator.gt[5] = {}
        self.start(evaluator, packet, manager)
        report = self.finish(evaluator, manager)
        self.assertEqual(report["status_counts"], {"missing_gt": 1})
        self.assertIsNone(report["metrics"]["ekf"]["ade"])
        self.assertIsNone(report["metrics"]["linear"]["ade"])
        self.assertEqual(report["evaluated_fraction"], 0.)

    def test_uncertain_identity_video_end_border_and_projection_exclusions(self):
        for reason in ("unmatched_at_start", "video_end", "gt_border_or_truncated", "invalid_gt_projection"):
            evaluator, packet, manager = self.fixture()
            if reason == "unmatched_at_start": packet["gt_id"] = None
            if reason == "video_end": evaluator.frame_count = 12
            if reason == "gt_border_or_truncated": evaluator.gt[5][7][0] = 0
            if reason == "invalid_gt_projection": evaluator.matrix[2] = 0
            self.start(evaluator, packet, manager)
            report = self.finish(evaluator, manager)
            self.assertEqual(report["status_counts"], {reason: 1})

    def test_early_quit_does_not_score_unprocessed_future(self):
        evaluator, packet, manager = self.fixture()
        self.start(evaluator, packet, manager)
        report = evaluator.report({})
        self.assertEqual(report["status_counts"], {"run_ended_before_horizon": 1})
        self.assertIsNone(report["metrics"]["ekf"]["fde"])

    def test_overlay_waits_for_full_horizon_and_does_not_double_count(self):
        evaluator, packet, manager = self.fixture()
        self.start(evaluator, packet, manager)
        self.assertEqual(evaluator.overlay_metrics(), {"samples": 0, "ade": None, "fde": None})
        self.finish(evaluator, manager, last=11)
        self.assertEqual(evaluator.overlay_metrics()["samples"], 0)
        evaluator.step(12, [], {}, [], [], manager, [])
        expected = {"samples": 1, "ade": 5., "fde": 5.}
        self.assertEqual(evaluator.overlay_metrics(), expected)
        self.assertEqual(evaluator.overlay_metrics(), expected)

    def test_overlay_excludes_unmatched_gt_and_clips_small_images(self):
        from src.forecast import draw_performance_metrics
        evaluator, packet, manager = self.fixture()
        packet["gt_id"] = None
        self.start(evaluator, packet, manager)
        self.finish(evaluator, manager)
        metrics = evaluator.overlay_metrics()
        self.assertEqual(metrics["samples"], 0)
        for shape in [(720, 1280, 3), (100, 40, 3), (8, 8, 3)]:
            image = np.zeros(shape, dtype=np.uint8)
            draw_performance_metrics(image, metrics)
            self.assertEqual(image.shape, shape)

    def test_display_history_never_admits_an_unmatched_window_to_metrics(self):
        evaluator, packet, manager = self.fixture()
        packet.update(gt_id=None, display_gt_id=7, display_gt_source="confirmed_history",
                      gt_path=[{"frame_idx": 0, "bottom_center_pixel": [100, 100]}])
        self.start(evaluator, packet, manager)
        report = self.finish(evaluator, manager)
        self.assertEqual(report["status_counts"], {"unmatched_at_start": 1})
        self.assertEqual(report["evaluated_windows"], 0)

    def test_covariance_coverage_and_singular_covariance_not_regularized(self):
        evaluator, packet, manager = self.fixture(error=(0, 0))
        packet["predictions"][0]["predicted_covariance"] = np.zeros((5, 5)).tolist()
        self.start(evaluator, packet, manager)
        report = self.finish(evaluator, manager)
        self.assertEqual(report["ellipse_valid_points"], 11)
        self.assertEqual(report["ellipse_95_gt_coverage"], 1.)

    def test_current_observed_and_ekf_errors_use_same_actual_matched_frames(self):
        evaluator, packet, manager = self.fixture()
        observations = [{"track_id": 1, "ground_position": [106, 108]}]
        states = [{"track_id": 1, "initialized": True, "estimated_px": 103, "estimated_py": 104}]
        evaluator.step(0, [], {1: 7}, observations, states, manager, [])
        evaluator.step(1, [], {}, observations, states, manager, [])
        report = evaluator.report({})
        self.assertEqual(report["current_position"]["observed"],
                         {"samples": 1, "mean_distance": 10., "rmse_distance": 10.})
        self.assertEqual(report["current_position"]["ekf"]["mean_distance"], 5.)

    def test_recording_groups_never_cross_roles_and_order_does_not_matter(self):
        groups = ["010200", "010201", "010202", "010203", "010204"]
        roles = split_groups(groups, "010201")
        self.assertEqual(roles["010201"], "calibration")
        self.assertEqual(set(roles.values()), {"calibration", "validation", "test"})
        self.assertEqual(roles, split_groups(groups[::-1], "010201"))
        with self.assertRaises(ValueError): split_groups(groups[:2], "010201")

    def test_all_scene_manifest_does_not_invent_holdouts_or_include_broken_gt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            videos = root / "Resources/videos_original"; videos.mkdir(parents=True)
            annotations = root / "Resources/annotations"; annotations.mkdir()
            names = ["VIRAT_S_050000_01_000000_000100",  # Only one recording.
                     "VIRAT_S_040103_01_000000_000100", "VIRAT_S_040104_01_000000_000100",
                     "VIRAT_S_040104_02_000100_000200"]
            for name in names:
                (videos / f"{name}.mp4").write_bytes(b"video")
                (annotations / f"{name}.viratdata.objects.txt").write_text("0 1 0 1 1 10 10 1\n")
            calibration = root / "calibration.json"; calibration.write_text("{}")

            def load_calibration(path, source, *unused):
                group = "050000" if "0500" in source.name else "040103"
                return np.array([1., 1.]), {"source": f"VIRAT_S_{group}_01_000000_000100.mp4"}

            def load_annotation(source, frames):
                if source.stem.endswith("02_000100_000200"):
                    raise ValueError("bad annotation columns")
                return {}, {"sha256": "verified"}

            cap = SimpleNamespace(get=lambda property: 100, release=lambda: None)
            with patch("src.evaluate.ROOT", root), patch("src.evaluate.default_calibration_path", return_value=calibration), \
                    patch("src.evaluate.load_r_calibration", side_effect=load_calibration), \
                    patch("src.evaluate.load_gt", side_effect=load_annotation), \
                    patch("src.evaluate.cv2.VideoCapture", return_value=cap):
                manifest = prepare_split(root / "split.json", all_scenes=True)
            self.assertEqual(len(manifest["videos"]), 3)
            self.assertEqual({e["split"] for e in manifest["videos"] if e["scene"] == "0500"}, {"calibration"})
            self.assertEqual({e["split"] for e in manifest["videos"] if e["group"] == "040104"}, {"validation"})
            self.assertEqual(set(manifest["scene_constraints"]), {"0401", "0500"})
            self.assertEqual(len(manifest["excluded"]), 1)

    def test_empty_metrics_are_null_and_reports_refuse_overwrite(self):
        self.assertIsNone(window_summary([])["metrics"]["ekf"]["ade"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            save_json(path, window_summary([]))
            self.assertIn('null', path.read_text())
            with self.assertRaises(FileExistsError): save_json(path, {})


if __name__ == "__main__":
    unittest.main()
