import json
import tempfile
import time
import unittest
import warnings
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sklearn.exceptions import ConvergenceWarning

import protocol.locked_runner as runner
import protocol.target_bounds as target_bounds


def feature(marker, module="AAA"):
    return {"signed_registration_lead_time_days": marker, "module": module,
            "log1p_total_clicks": 0.1, "active_days": 1, "recency_days_at_day28": 2,
            "no_activity": 0, "log1p_clicks_days_0_6": 0.1, "log1p_clicks_days_7_13": 0,
            "log1p_clicks_days_14_20": 0, "log1p_clicks_days_21_27": 0,
            "log1p_clicks_days_28_28": 0}


def small_bundle():
    source = [(f"s{i:03}", "AAA", "2013J") for i in range(60)]
    known = [(f"k{i:05}", "AAA", "2014J") for i in range(6771)]
    unknown = [(f"u{i:03}", "BBB", "2014J") for i in range(58)]
    return {"source_labels": {k: int(i % 2 == 0) for i, k in enumerate(source)},
            "source_features": {k: feature(i) for i, k in enumerate(source)},
            "known_target_labels": {k: i % 2 for i, k in enumerate(known)},
            "known_target_features": {k: feature(9000 + i) for i, k in enumerate(known)},
            "uncertain_target_features": {k: feature(10000 + i, "BBB") for i, k in enumerate(unknown)},
            "_known": known, "_unknown": unknown}


@contextmanager
def mock_run_context(config, bundle):
    derived = runner.ROOT / "data/derived"
    derived.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=derived) as temp:
        run_dir = Path(temp).resolve()
        run_id = run_dir.name
        ref = "mock-parent-authorized-test-only"
        hashes = {"protocol_sha256": "mock-protocol", "data_sha256": "mock-data",
                  "split_sha256": "mock-split", "code_sha256": "mock-code"}
        claim_path = runner._claim_path(run_id)
        runner._write_claim_exclusive(claim_path, {"run_id": run_id, "parent_authorization_reference": ref,
            "output_path": str(run_dir), "hashes": hashes, "overall_status": "running"})
        (run_dir / "authorization_receipt.json").write_text(json.dumps({"test_only": True}))
        ctx = runner.ExecutionContext(run_id, ref, run_dir, claim_path, hashes,
                                      data_bundle=bundle)
        try:
            yield run_dir, ctx
        finally:
            claim_path.unlink(missing_ok=True)


@contextmanager
def allow_mock_context_only():
    # The test-only substitution never ships outside this test module.
    with patch.object(runner, "_validate_execution_context", return_value=True):
        yield


class FakeEstimator:
    def __init__(self, params, store, kind="base"):
        self.params, self.store, self.kind = params, store, kind
        self.fit_calls = 0
        self.decision_batches = []

    def fit(self, x, y):
        self.fit_calls += 1
        self.store.append((self.kind, np.asarray(x).copy(), np.asarray(y).copy()))
        self.coef_ = np.zeros((1, 1))
        self.intercept_ = np.zeros(1)
        return self

    def decision_function(self, x):
        x = np.asarray(x, dtype=object)
        self.decision_batches.append(len(x))
        return x[:, 0] / 100.0


class FakeCalibrator(FakeEstimator):
    def __init__(self, params, store):
        super().__init__(params, store, "calibrator")

    def predict_proba(self, x):
        self.prediction_batches = getattr(self, "prediction_batches", []) + [len(x)]
        n = len(x)
        return np.column_stack([np.full(n, .6), np.full(n, .4)])


class FakeBadCalibrator(FakeCalibrator):
    def predict_proba(self, x):
        n = len(x)
        return np.column_stack([np.zeros(n), np.full(n, np.nan)])


class AuthorizationGateTest(unittest.TestCase):
    def test_requires_explicit_parent_auth_and_all_exact_hashes(self):
        config = {"document_status": "LOCKED", "protocol_lock": True, "execution_authorization": True}
        hashes = dict(protocol_hash="p", data_hash="d", split_hash="s", code_hash="c")
        manifest = {"review_status": "APPROVED",
                    "protocol_sha256": "p", "data_sha256": "d", "split_sha256": "s", "code_sha256": "c",
                    "reviewer": "reviewer", "parent_authorization_reference": "approved-session-123"}
        kwargs = dict(parent_authorization_reference="approved-session-123")
        with self.assertRaises(runner.GateError):
            runner.verify_lock(config=config, manifest=manifest, explicit_cli_authorization=False, **hashes, **kwargs)
        for field, value in (("review_status", "DRAFT"),
                             ("protocol_sha256", "wrong"), ("data_sha256", "wrong"),
                             ("split_sha256", "wrong"), ("code_sha256", "wrong"),
                             ("parent_authorization_reference", "wrong")):
            with self.subTest(field=field):
                bad = manifest | {field: value}
                with self.assertRaises(runner.GateError):
                    runner.verify_lock(config=config, manifest=bad, explicit_cli_authorization=True, **hashes, **kwargs)
        with self.assertRaises(runner.GateError):
            runner.verify_lock(config=config | {"document_status": "DRAFT_FOR_REVIEW"}, manifest=manifest,
                               explicit_cli_authorization=True, **hashes, **kwargs)
        self.assertTrue(runner.verify_lock(config=config, manifest=manifest,
                                           explicit_cli_authorization=True, **hashes, **kwargs))

    def test_normal_invocation_is_explicitly_non_fitting(self):
        report = runner.dry_run_summary(Path("protocol/candidate_protocol.json"), Path("audit/data_manifest.json"))
        self.assertEqual(report["status"], "DRY_RUN_NO_FIT")
        self.assertFalse(report["fit_called"])
        self.assertFalse(report["prediction_called"])

    def test_runtime_pins_and_execution_caps_are_checked_without_fitting(self):
        config = json.loads(Path("protocol/candidate_protocol.json").read_text())
        actual = runner.verify_runtime_and_limits(config)
        self.assertEqual(actual["python"], config["runtime"]["python"])
        for field, bad in (("cpu_threads_max", 3), ("overall_execution_timeout_seconds", 1201),
                           ("base_estimator_fit_timeout_seconds", 301), ("device", "GPU")):
            with self.subTest(field=field):
                copy_config = json.loads(json.dumps(config))
                copy_config["execution_limits"][field] = bad
                with self.assertRaises(runner.GateError):
                    runner.verify_runtime_and_limits(copy_config)
        changed_params = json.loads(json.dumps(config))
        changed_params["predictors"][1]["parameters"]["C"] = 2.0
        with self.assertRaisesRegex(runner.GateError, "fixed base-model parameters"):
            runner.verify_runtime_and_limits(changed_params)

    def test_pipeline_has_fixed_parameters_and_fit_partition_transformers(self):
        config = json.loads(Path("protocol/candidate_protocol.json").read_text())
        registration = runner._make_pipeline(config["predictors"][1])
        activity = runner._make_pipeline(config["predictors"][2])
        pipeline = activity
        self.assertEqual(pipeline.named_steps["model"].get_params()["max_iter"], 1000)
        transformers = pipeline.named_steps["features"].transformers
        self.assertIn("StandardScaler", repr(transformers[0][1]))
        self.assertIn("OneHotEncoder", repr(transformers[1][1]))
        self.assertEqual(transformers[1][1].handle_unknown, "error")
        self.assertEqual(registration.named_steps["features"].transformers[0][2], [0])
        self.assertEqual(len(transformers[0][2]), 10)

    def test_fit_budget_enforces_per_fit_and_total_caps_with_mock_clock(self):
        self.assertEqual(runner.fit_budget(started_at=10, now=11, total_cap=1200, per_fit_cap=300), 300)
        self.assertEqual(runner.fit_budget(started_at=10, now=1110, total_cap=1200, per_fit_cap=300), 100)
        with self.assertRaises(TimeoutError):
            runner.fit_budget(started_at=10, now=1210, total_cap=1200, per_fit_cap=300)

    def test_private_artifact_path_must_be_git_ignored(self):
        self.assertTrue(runner.verify_private_output(runner.ROOT / "data/derived/candidate"))
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(runner.GateError):
            runner.verify_private_output(Path(tmp) / "predictions.csv")

    def test_public_runner_records_rejected_gate_without_touching_data_or_fitting(self):
        (runner.ROOT / "data/derived").mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=runner.ROOT / "data/derived") as tmp:
            called = []
            with self.assertRaises(runner.GateError):
                runner.run_authorized_from_data(data_dir=Path(tmp) / "missing-data",
                                                data_manifest_path=Path(tmp) / "missing-manifest.json",
                                                config=json.loads(Path("protocol/candidate_protocol.json").read_text()),
                                                manifest={}, run_id=Path(tmp).name,
                                                parent_authorization_reference="", explicit_cli_authorization=False)
            self.assertEqual(called, [])
            self.assertFalse((Path(tmp) / "run_manifest.json").exists())

    def test_existing_claim_replay_fails_before_loading_or_fitting(self):
        (runner.ROOT / "data/derived").mkdir(parents=True, exist_ok=True)
        run_id = "replay-test-unique"
        claim_path = runner._claim_path(run_id)
        claim_path.unlink(missing_ok=True)
        runner._write_claim_exclusive(claim_path, {"run_id": run_id, "overall_status": "failed"})
        calls = []
        try:
            with self.assertRaisesRegex(runner.GateError, "replay is prohibited"):
                runner.run_authorized_from_data(data_dir=Path("missing"), data_manifest_path=Path("missing.json"),
                    config={}, manifest={}, run_id=run_id, parent_authorization_reference="parent-task:abc",
                    explicit_cli_authorization=True)
            self.assertEqual(calls, [])
        finally:
            claim_path.unlink(missing_ok=True)


class MockOnlyExecutionTest(unittest.TestCase):
    def _config(self):
        config = json.loads(Path("protocol/candidate_protocol.json").read_text())
        config["split"]["minimum_calibration_positive_records"] = 2
        config["study"]["shared_modules"] = ["AAA"]
        return config

    def test_mock_pipeline_fits_once_without_target_leakage_and_writes_atomically(self):
        bundle = small_bundle()
        calls = []
        models, calibrators = [], []

        def model_factory(params):
            obj = FakeEstimator(params, calls)
            models.append(obj)
            return obj

        def calibrator_factory(params):
            obj = FakeCalibrator(params, calls)
            calibrators.append(obj)
            return obj

        def fake_fit(estimator, x, y, *, timeout_seconds, log_path, execution_context, config):
            self.assertLessEqual(timeout_seconds, 300)
            self.assertGreater(timeout_seconds, 0)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("mock fit only\n")
            estimator.fit(x, y)
            return estimator

        config = self._config()
        with mock_run_context(config, bundle) as (out, context):
            with allow_mock_context_only(), patch.object(target_bounds, "KNOWN_TARGET_KEYSET_SHA256",
                              target_bounds._fingerprint(set(bundle["_known"]))), \
                 patch.object(target_bounds, "UNCERTAIN_TARGET_KEYSET_SHA256",
                              target_bounds._fingerprint(set(bundle["_unknown"]))):
                result = runner._run_experiment_after_gate(data_bundle=bundle, config=config, run_dir=out,
                                                           execution_context=context,
                                                           fit_executor=fake_fit, estimator_factory=model_factory,
                                                           calibrator_factory=calibrator_factory, clock=lambda: 100.0)
            self.assertEqual(len(models), 2)
            self.assertTrue(all(m.fit_calls == 1 for m in models))
            self.assertEqual([m.params["name"] for m in models],
                             ["registration_logistic", "registration_plus_vle_logistic"])
            self.assertEqual(len(calibrators), 2)
            self.assertTrue(all(c.fit_calls == 1 for c in calibrators))
            self.assertTrue(all(m.decision_batches == [12, 6771, 58] for m in models))
            self.assertTrue(all(c.prediction_batches == [6771, 58] for c in calibrators))
            self.assertEqual(len(calls), 4)
            _, expected_fit, expected_cal = runner.split_fingerprint(bundle["source_labels"], bundle["source_features"],
                                                                     self._config()["split"])
            expected_fit_y = np.asarray([bundle["source_labels"][k] for k in expected_fit])
            expected_cal_y = np.asarray([bundle["source_labels"][k] for k in expected_cal])
            for kind, x, y in calls:
                if kind == "base":
                    self.assertLess(float(np.asarray(x, dtype=object)[:, 0].astype(float).max()), 60)
                    np.testing.assert_array_equal(y, expected_fit_y)
                if kind == "calibrator":
                    np.testing.assert_array_equal(y, expected_cal_y)
                    self.assertLess(float(np.asarray(x, dtype=float).max()), .6)
            self.assertEqual(result["primary_paired_brier_improvement_bounds"]["known_records"], 6771)
            self.assertEqual(result["primary_paired_brier_improvement_bounds"]["uncertain_records"], 58)
            self.assertEqual(json.loads((out / "stage_status.json").read_text())["overall_status"], "complete")
            self.assertTrue(runner.verify_complete_run(out))
            self.assertEqual(json.loads((out / "run_manifest.json").read_text())["overall_status"], "complete")
            self.assertTrue((out / "private_predictions.csv").is_file())
            self.assertTrue((out / "aggregate_metrics.json").is_file())
            self.assertTrue((out / "frozen_models/registration_logistic.pkl").is_file())
            self.assertTrue((out / "frozen_models/registration_plus_vle_logistic.pkl").is_file())
            self.assertFalse(list(out.rglob("*.part")))

    def test_core_fit_helper_rejects_bypass_stale_hash_and_output_escape(self):
        bundle = small_bundle()
        config = self._config()
        calls = []
        with mock_run_context(config, bundle) as (run_dir, context):
            with self.assertRaisesRegex(runner.GateError, "validated execution context"):
                runner._run_experiment_after_gate(data_bundle=bundle, config=config, run_dir=run_dir,
                                                  execution_context=None, fit_executor=lambda *a, **k: calls.append(True))
            with tempfile.TemporaryDirectory() as escape:
                with self.assertRaisesRegex(runner.GateError, "claimed run ID"):
                    runner._run_experiment_after_gate(data_bundle=bundle, config=config, run_dir=Path(escape) / "out",
                                                      execution_context=context, fit_executor=lambda *a, **k: calls.append(True))
            stale = replace(context, hashes=context.hashes | {"code_sha256": "stale-hash"})
            with self.assertRaisesRegex(runner.GateError, "claim receipt mismatch"):
                runner._validate_execution_context(stale, config, run_dir)
            with self.assertRaises(runner.GateError):
                runner.fit_with_deadline(FakeEstimator({}, []), np.zeros((2, 1)), np.array([0, 1]),
                                         timeout_seconds=1, log_path=run_dir / "logs/bypass.json",
                                         execution_context=None, config=config)
            self.assertEqual(calls, [])

    def test_injected_timeout_fails_stage_and_preserves_log(self):
        bundle = small_bundle()
        def model_factory(params):
            return FakeEstimator(params, [])
        def timeout_fit(estimator, x, y, *, timeout_seconds, log_path, execution_context, config):
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("injected timeout evidence")
            raise runner.FitFailure("injected timeout", log_path)
        config = self._config()
        with mock_run_context(config, bundle) as (out, context):
            with allow_mock_context_only(), patch.object(target_bounds, "KNOWN_TARGET_KEYSET_SHA256",
                              target_bounds._fingerprint(set(bundle["_known"]))), \
                 patch.object(target_bounds, "UNCERTAIN_TARGET_KEYSET_SHA256",
                              target_bounds._fingerprint(set(bundle["_unknown"]))):
                with self.assertRaises(runner.FitFailure):
                    runner._run_experiment_after_gate(data_bundle=bundle, config=config, run_dir=out,
                                                      execution_context=context,
                                                      fit_executor=timeout_fit, estimator_factory=model_factory,
                                                      calibrator_factory=lambda p: FakeCalibrator(p, []), clock=lambda: 20.0)
            state = json.loads((out / "stage_status.json").read_text())
            self.assertEqual(state["overall_status"], "failed")
            self.assertTrue((out / "logs/registration_logistic_base_fit.json").is_file())
            self.assertFalse(runner.verify_complete_run(out))
            self.assertFalse(json.loads((out / "run_manifest.json").read_text())["outputs_usable"])

    def test_invalid_target_probabilities_fail_closed(self):
        class Model:
            def decision_function(self, x):
                return np.zeros(len(x))
        class Cal:
            def predict_proba(self, x):
                return np.column_stack([np.zeros(len(x)), np.full(len(x), np.nan)])
        with self.assertRaises(ValueError):
            runner._probabilities(Model(), Cal(), np.zeros((2, 1)))
        class BadScore(Model):
            def decision_function(self, x):
                return np.full(len(x), np.inf)
        with self.assertRaisesRegex(ValueError, "non-finite decision score"):
            runner._probabilities(BadScore(), Cal(), np.zeros((2, 1)))

    def test_invalid_probability_cannot_produce_complete_run(self):
        bundle = small_bundle()
        def mock_fit(estimator, x, y, *, timeout_seconds, log_path, execution_context, config):
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("mock estimator/calibrator fit only")
            estimator.fit(x, y)
            return estimator
        config = self._config()
        with mock_run_context(config, bundle) as (out, context):
            with allow_mock_context_only(), patch.object(target_bounds, "KNOWN_TARGET_KEYSET_SHA256",
                              target_bounds._fingerprint(set(bundle["_known"]))), \
                 patch.object(target_bounds, "UNCERTAIN_TARGET_KEYSET_SHA256",
                              target_bounds._fingerprint(set(bundle["_unknown"]))):
                with self.assertRaisesRegex(RuntimeError, "probability is missing or invalid"):
                    runner._run_experiment_after_gate(data_bundle=bundle, config=config, run_dir=out,
                        execution_context=context,
                        fit_executor=mock_fit, estimator_factory=lambda p: FakeEstimator(p, []),
                        calibrator_factory=lambda p: FakeBadCalibrator(p, []), clock=lambda: 30.0)
            state = json.loads((out / "stage_status.json").read_text())
            self.assertEqual(state["overall_status"], "failed")
            self.assertFalse((out / "aggregate_metrics.json").exists())
            self.assertTrue((out / "logs/runner_failure.json").is_file())
            self.assertFalse(runner.verify_complete_run(out))

    def test_mid_write_failure_marks_all_partial_outputs_unusable(self):
        bundle = small_bundle()
        config = self._config()
        def mock_fit(estimator, x, y, *, timeout_seconds, log_path, execution_context, config):
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("mock only")
            estimator.fit(x, y)
            return estimator
        with mock_run_context(config, bundle) as (out, context):
            real_atomic = runner._atomic_write
            def fail_prediction_write(path, payload):
                if Path(path).name == "private_predictions.csv":
                    raise OSError("injected failure during result write")
                return real_atomic(path, payload)
            with allow_mock_context_only(), patch.object(target_bounds, "KNOWN_TARGET_KEYSET_SHA256",
                              target_bounds._fingerprint(set(bundle["_known"]))), \
                 patch.object(target_bounds, "UNCERTAIN_TARGET_KEYSET_SHA256",
                              target_bounds._fingerprint(set(bundle["_unknown"]))), \
                 patch.object(runner, "_atomic_write", side_effect=fail_prediction_write):
                with self.assertRaisesRegex(RuntimeError, "injected failure during result write"):
                    runner._run_experiment_after_gate(data_bundle=bundle, config=config, run_dir=out,
                        execution_context=context, fit_executor=mock_fit, estimator_factory=lambda p: FakeEstimator(p, []),
                        calibrator_factory=lambda p: FakeCalibrator(p, []), clock=lambda: 30.0)
            manifest = json.loads((out / "run_manifest.json").read_text())
            self.assertEqual(manifest["overall_status"], "failed")
            self.assertFalse(manifest["outputs_usable"])
            self.assertTrue((out / "aggregate_metrics.json").is_file())  # partial file is explicitly unusable
            self.assertFalse(runner.verify_complete_run(out))
            self.assertEqual(json.loads(context.claim_path.read_text())["overall_status"], "failed")

    def test_actual_fit_worker_kills_timeout_and_rejects_convergence_warning(self):
        config = json.loads(Path("protocol/candidate_protocol.json").read_text())
        with allow_mock_context_only(), mock_run_context(config, {}) as (run_dir, context):
            success_log = run_dir / "logs/success.json"
            successful = runner.fit_with_deadline(FakeEstimator({}, []), np.zeros((2, 1)), np.array([0, 1]),
                                                  timeout_seconds=2, log_path=success_log,
                                                  execution_context=context, config=config)
            self.assertEqual(successful.fit_calls, 1)
            self.assertEqual(json.loads(success_log.read_text())["status"], "complete")
        class Slow:
            def fit(self, x, y):
                time.sleep(.3)
        class Warn:
            def fit(self, x, y):
                warnings.warn("simulated non-convergence", ConvergenceWarning)
                self.coef_ = np.zeros((1, 1))
        class Nonfinite:
            def fit(self, x, y):
                self.coef_ = np.array([[np.inf]])
        with allow_mock_context_only(), mock_run_context(config, {}) as (run_dir, context):
            timeout_log = run_dir / "logs/timeout.json"
            with self.assertRaises(runner.FitFailure):
                runner.fit_with_deadline(Slow(), np.zeros((2, 1)), np.array([0, 1]),
                                         timeout_seconds=.02, log_path=timeout_log,
                                         execution_context=context, config=config)
            self.assertTrue(timeout_log.is_file())
            warning_log = run_dir / "logs/warning.json"
            with self.assertRaisesRegex(runner.FitFailure, "convergence_warning"):
                runner.fit_with_deadline(Warn(), np.zeros((2, 1)), np.array([0, 1]),
                                         timeout_seconds=2, log_path=warning_log,
                                         execution_context=context, config=config)
            self.assertIn("simulated non-convergence", warning_log.read_text())
            coefficient_log = run_dir / "logs/nonfinite.json"
            with self.assertRaisesRegex(runner.FitFailure, "non_finite_coefficients"):
                runner.fit_with_deadline(Nonfinite(), np.zeros((2, 1)), np.array([0, 1]),
                                         timeout_seconds=2, log_path=coefficient_log,
                                         execution_context=context, config=config)
            self.assertIn("non_finite_coefficients", coefficient_log.read_text())


if __name__ == "__main__":
    unittest.main()
