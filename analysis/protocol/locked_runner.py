"""Hash-gated, bounded execution runner for the reviewed candidate protocol.

The default CLI is a non-fitting dry run. Real execution requires a locked
protocol, an authenticated parent-task authorization reference, matching
protocol/data/split/code fingerprints, and the explicit --execute-authorized switch.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import math
import multiprocessing as mp
import os
import pickle
import signal
import time
import traceback
import warnings
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import platform

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from audit_pipeline import sha256
from protocol.brier_bounds import _probability
from protocol.day28_features import build_target_preflight
from protocol.target_bounds import primary_target_bounds

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_FILES = ("protocol/candidate_protocol.json", "protocol/draft_fitting_protocol.md",
                  "protocol/partial_brier_bounds.md", "protocol/requirements-modeling.txt")
CODE_FILES = ("protocol/locked_runner.py", "protocol/day28_features.py", "protocol/target_bounds.py",
              "protocol/brier_bounds.py", "protocol/execution_contract.py",
              "audit_pipeline.py", "audit_label_availability.py", ".gitignore")
FEATURE_COLUMNS = ("signed_registration_lead_time_days", "module", "log1p_total_clicks", "active_days",
                   "recency_days_at_day28", "no_activity", "log1p_clicks_days_0_6", "log1p_clicks_days_7_13",
                   "log1p_clicks_days_14_20", "log1p_clicks_days_21_27", "log1p_clicks_days_28_28")
NUMERIC_COLUMNS = ("signed_registration_lead_time_days", "log1p_total_clicks", "active_days",
                   "recency_days_at_day28", "no_activity", "log1p_clicks_days_0_6", "log1p_clicks_days_7_13",
                   "log1p_clicks_days_14_20", "log1p_clicks_days_21_27", "log1p_clicks_days_28_28")


class GateError(RuntimeError):
    pass


class FitFailure(RuntimeError):
    def __init__(self, message, log_path):
        super().__init__(f"{message}; preserved log: {log_path}")
        self.log_path = str(log_path)


@dataclass(frozen=True)
class ExecutionContext:
    run_id: str
    parent_authorization_reference: str
    output_dir: Path
    claim_path: Path
    hashes: dict
    data_dir: Path | None = None
    data_manifest_path: Path | None = None
    data_bundle: dict | None = None


def _valid_run_id(run_id):
    import re
    return isinstance(run_id, str) and bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", run_id)) and run_id not in {".", ".."}


def _claim_path(run_id, root=ROOT):
    return root / "data/derived/.claims" / f"{run_id}.json"


def _write_claim_exclusive(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    lock_path = path.with_suffix(".claiming")
    lock_fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(lock_fd)
    temp_path = path.with_suffix(".part")
    try:
        fd = os.open(temp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, path)
        lock_path.unlink()
    except BaseException:
        # A stale lock/part remains a durable no-replay signal after interrupted claim creation.
        raise


def _update_claim(context: ExecutionContext, status: str):
    claim = json.loads(context.claim_path.read_text())
    claim["overall_status"] = status
    _atomic_json(context.claim_path, claim)


def _canonical_hash(entries):
    payload = "".join(f"{name}\t{digest}\n" for name, digest in entries).encode()
    return hashlib.sha256(payload).hexdigest()


def verify_private_output(path: Path, root: Path = ROOT):
    path = path.resolve()
    derived = (root / "data/derived").resolve()
    patterns = {line.strip() for line in (root / ".gitignore").read_text().splitlines() if line.strip() and not line.lstrip().startswith("#")}
    if not path.is_relative_to(derived) or not ({"data/derived", "data/derived/"} & patterns):
        raise GateError("model artifacts and row-level predictions must remain in Git-ignored data/derived")
    return path


def protocol_fingerprint(root: Path = ROOT) -> str:
    return _canonical_hash((name, sha256(root / name)) for name in PROTOCOL_FILES)


def code_fingerprint(root: Path = ROOT) -> str:
    return _canonical_hash((name, sha256(root / name)) for name in CODE_FILES)


def verify_data_files(data_dir: Path, manifest_path: Path) -> str:
    manifest = json.loads(manifest_path.read_text())
    rows = []
    archive = manifest["archive"]
    archive_path = ROOT / archive["path"]
    if not archive_path.is_file() or sha256(archive_path) != archive["sha256"]:
        raise GateError("raw archive is missing or its hash differs from audit/data_manifest.json")
    rows.append((archive["path"], archive["sha256"]))
    for table, metadata in sorted(manifest["tables"].items()):
        path = data_dir / f"{table}.csv"
        if not path.is_file() or path.stat().st_size != metadata["bytes"] or sha256(path) != metadata["sha256"]:
            raise GateError(f"raw table hash/size mismatch: {table}")
        rows.append((f"{table}.csv", metadata["sha256"]))
    return _canonical_hash(rows)


def verify_runtime_and_limits(config):
    expected = config["runtime"]
    actual = {"python": platform.python_version(), "scikit_learn": importlib.metadata.version("scikit-learn"),
              "numpy": importlib.metadata.version("numpy"), "scipy": importlib.metadata.version("scipy"),
              "joblib": importlib.metadata.version("joblib"), "threadpoolctl": importlib.metadata.version("threadpoolctl")}
    for name, value in actual.items():
        if expected.get(name) != value:
            raise GateError(f"runtime version differs from reviewed pin: {name}")
    limits = config["execution_limits"]
    fixed = {"cpu_threads_max": 2, "overall_execution_timeout_seconds": 1200,
             "base_estimator_fit_timeout_seconds": 300, "device": "CPU only", "paid_resources": False,
             "estimator_parameters_unchanged": True, "base_estimator_max_iter": 1000}
    for name, value in fixed.items():
        if limits.get(name) != value:
            raise GateError(f"execution contract differs from reviewed bound: {name}")
    base_parameters = {"C": 1.0, "penalty": "l2", "solver": "lbfgs", "class_weight": None,
                       "fit_intercept": True, "max_iter": 1000, "tol": 0.0001, "random_state": 42}
    calibration_parameters = {"penalty": None, "solver": "lbfgs", "fit_intercept": True,
                              "max_iter": 1000, "tol": 0.0001, "random_state": 42}
    for predictor in config["predictors"]:
        if predictor["name"] != "source_prevalence_constant" and predictor.get("parameters") != base_parameters:
            raise GateError(f"fixed base-model parameters changed: {predictor['name']}")
    if config["calibration"].get("calibrator_parameters") != calibration_parameters:
        raise GateError("fixed calibration parameters changed")
    return actual


def split_fingerprint(source_labels, source_features, split_config):
    keys = sorted(source_labels)
    y = np.asarray([source_labels[k] for k in keys], dtype=np.int8)
    groups = np.asarray([k[0] for k in keys])
    x_dummy = np.zeros((len(keys), 1), dtype=np.int8)
    splitter = StratifiedGroupKFold(n_splits=split_config["n_splits"], shuffle=split_config["shuffle"],
                                    random_state=split_config["random_state"])
    fit_ix, calibration_ix = next(splitter.split(x_dummy, y, groups))
    if split_config["fixed_fold_index"] != 0:
        raise GateError("only prespecified fold zero is allowed")
    fit_keys = sorted(keys[i] for i in fit_ix)
    calibration_keys = sorted(keys[i] for i in calibration_ix)
    if set(fit_keys) & set(calibration_keys):
        raise GateError("source student overlap in split")
    split_hash = _canonical_hash([("fit", hashlib.sha256("".join("\t".join(k)+"\n" for k in fit_keys).encode()).hexdigest()),
                                  ("calibration", hashlib.sha256("".join("\t".join(k)+"\n" for k in calibration_keys).encode()).hexdigest())])
    return split_hash, fit_keys, calibration_keys


def _validate_execution_context(context, config, run_dir):
    if not isinstance(context, ExecutionContext):
        raise GateError("a validated execution context is required for every fit helper")
    if not _valid_run_id(context.run_id) or not isinstance(context.parent_authorization_reference, str) or not context.parent_authorization_reference.strip():
        raise GateError("execution context lacks a valid run ID or parent authorization reference")
    expected_output = verify_private_output(ROOT / "data/derived" / context.run_id)
    if Path(run_dir).resolve() != expected_output or context.output_dir.resolve() != expected_output:
        raise GateError("execution context output path does not match its claimed run ID")
    expected_claim = _claim_path(context.run_id).resolve()
    if (context.claim_path.resolve() != expected_claim or not expected_claim.is_file()
            or expected_claim.with_suffix(".claiming").exists()):
        raise GateError("execution context claim receipt is absent or mismatched")
    claim = json.loads(expected_claim.read_text())
    for field, value in (("run_id", context.run_id),
                         ("parent_authorization_reference", context.parent_authorization_reference),
                         ("output_path", str(expected_output)), ("hashes", context.hashes)):
        if claim.get(field) != value:
            raise GateError(f"execution claim receipt mismatch: {field}")
    if claim.get("overall_status") not in {"claimed", "running"}:
        raise GateError("execution claim is no longer active")
    if config.get("document_status") != "LOCKED" or not config.get("protocol_lock") or not config.get("execution_authorization"):
        raise GateError("protocol lock/authorization no longer matches execution context")
    if config != json.loads((ROOT / "protocol/candidate_protocol.json").read_text()):
        raise GateError("in-memory configuration differs from the hashed protocol file")
    verify_runtime_and_limits(config)
    if protocol_fingerprint() != context.hashes["protocol_sha256"]:
        raise GateError("protocol hash changed after authorization")
    if code_fingerprint() != context.hashes["code_sha256"]:
        raise GateError("code hash changed after authorization")
    if verify_data_files(context.data_dir, context.data_manifest_path) != context.hashes["data_sha256"]:
        raise GateError("data hash changed after authorization")
    if split_fingerprint(context.data_bundle["source_labels"], context.data_bundle["source_features"],
                         config["split"])[0] != context.hashes["split_sha256"]:
        raise GateError("source split hash changed after authorization")
    return True


def verify_lock(*, config, manifest, explicit_cli_authorization, protocol_hash,
                data_hash, split_hash, code_hash, parent_authorization_reference):
    if not explicit_cli_authorization:
        raise GateError("fitting requires the explicit --execute-authorized switch")
    if config.get("document_status") != "LOCKED" or not config.get("protocol_lock"):
        raise GateError("candidate protocol is not finally locked")
    if not config.get("execution_authorization"):
        raise GateError("candidate protocol does not authorize execution")
    required = {"review_status": "APPROVED",
                "protocol_sha256": protocol_hash, "data_sha256": data_hash,
                "split_sha256": split_hash, "code_sha256": code_hash}
    for field, expected in required.items():
        if manifest.get(field) != expected:
            raise GateError(f"authorization manifest mismatch: {field}")
    if not manifest.get("reviewer") or not parent_authorization_reference:
        raise GateError("reviewer and explicit parent task authorization reference are required")
    if manifest.get("parent_authorization_reference") != parent_authorization_reference:
        raise GateError("audit receipt parent authorization reference does not match the authenticated task instruction")
    return True


def fit_budget(*, started_at, now, total_cap, per_fit_cap):
    remaining = started_at + total_cap - now
    if remaining <= 0:
        raise TimeoutError("overall execution deadline reached")
    return min(per_fit_cap, remaining)


def _atomic_write(path: Path, payload: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.part")
    try:
        with tmp.open("wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _atomic_json(path: Path, data):
    _atomic_write(path, (json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())


def _artifact_hashes(run_dir: Path):
    return {str(path.relative_to(run_dir)): sha256(path)
            for path in sorted(run_dir.rglob("*")) if path.is_file() and path.name != "run_manifest.json"}


def verify_complete_run(run_dir: Path) -> bool:
    """Artifacts are usable only with a complete marker and matching file hashes."""
    run_dir = Path(run_dir)
    try:
        manifest = json.loads((run_dir / "run_manifest.json").read_text())
        stages = json.loads((run_dir / "stage_status.json").read_text())
        if manifest.get("overall_status") != "complete" or manifest.get("outputs_usable") is not True:
            return False
        if stages.get("overall_status") != "complete":
            return False
        expected = manifest.get("artifact_sha256", {})
        return bool(expected) and expected == _artifact_hashes(run_dir)
    except (OSError, ValueError, TypeError):
        return False


class StageLedger:
    def __init__(self, run_dir: Path, clock=time.monotonic):
        self.path = run_dir / "stage_status.json"
        self.clock = clock
        self.started = clock()
        self.stages = {}
        self.run_dir = run_dir
        self.write("incomplete")

    def set(self, name, status, **evidence):
        self.stages[name] = {"status": status, "at_elapsed_seconds": round(self.clock() - self.started, 3), **evidence}
        self.write("incomplete")

    def write(self, status):
        _atomic_json(self.path, {"overall_status": status, "stages": self.stages})

    def fail(self, stage, error, log_path=None):
        self.set(stage, "failed", error=str(error), log_path=str(log_path) if log_path else None)
        self.write("failed")


def _child_fit(send, estimator, x, y, log_path, execution_context, config, run_dir):
    messages = []
    try:
        _validate_execution_context(execution_context, config, run_dir)
        if not Path(log_path).resolve().is_relative_to(execution_context.output_dir.resolve()):
            raise GateError("fit logs must remain under the claimed private output directory")
        with threadpool_limits(limits=2):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                estimator.fit(x, y)
            messages.extend(f"{w.category.__name__}: {w.message}" for w in caught)
            convergence = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
            if convergence:
                send.send({"ok": False, "reason": "convergence_warning", "log": "\n".join(messages)})
                return
            coefficient_owner = estimator.named_steps["model"] if hasattr(estimator, "named_steps") else estimator
            coefficients = []
            for attr in ("coef_", "intercept_"):
                if hasattr(coefficient_owner, attr):
                    coefficients.extend(np.asarray(getattr(coefficient_owner, attr), dtype=float).ravel().tolist())
            if coefficients and not np.isfinite(coefficients).all():
                send.send({"ok": False, "reason": "non_finite_coefficients", "log": "\n".join(messages)})
                return
            send.send({"ok": True, "estimator": estimator, "log": "\n".join(messages)})
    except BaseException as exc:
        send.send({"ok": False, "reason": f"{type(exc).__name__}: {exc}",
                   "log": "\n".join(messages + [traceback.format_exc()])})
    finally:
        send.close()


def fit_with_deadline(estimator, x, y, *, timeout_seconds, log_path, execution_context, config):
    """Run one fit in a killable child process; convergence and timeout fail closed."""
    if not isinstance(execution_context, ExecutionContext):
        raise GateError("a validated execution context is required for every fit helper")
    _validate_execution_context(execution_context, config, execution_context.output_dir)
    if not Path(log_path).resolve().is_relative_to(execution_context.output_dir.resolve()):
        raise GateError("fit logs must remain under the claimed private output directory")
    timeout_seconds = min(float(timeout_seconds), config["execution_limits"]["base_estimator_fit_timeout_seconds"])
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise TimeoutError("fit deadline is non-positive or invalid")
    parent, child = mp.Pipe(duplex=False)
    process = mp.get_context("fork").Process(target=_child_fit,
        args=(child, estimator, x, y, str(log_path), execution_context, config, execution_context.output_dir))
    started = time.monotonic()
    process.start()
    child.close()
    try:
        remaining = max(0.0, timeout_seconds - (time.monotonic() - started))
        process.join(remaining)
    except BaseException:
        if process.is_alive():
            process.terminate()
            process.join()
        parent.close()
        raise
    if process.is_alive():
        process.terminate()
        process.join()
        log = {"status": "failed", "reason": "timeout", "timeout_seconds": timeout_seconds}
        _atomic_json(Path(log_path), log)
        raise FitFailure("base/calibration fit timed out", log_path)
    if not parent.poll():
        log = {"status": "failed", "reason": f"worker_exit_{process.exitcode}",
               "elapsed_seconds": time.monotonic() - started}
        _atomic_json(Path(log_path), log)
        raise FitFailure("fit worker exited without a result", log_path)
    result = parent.recv()
    parent.close()
    _atomic_json(Path(log_path), {"status": "complete" if result["ok"] else "failed",
                                  "elapsed_seconds": time.monotonic() - started,
                                  "reason": result.get("reason"), "log": result.get("log", "")})
    if not result["ok"]:
        raise FitFailure(result.get("reason", "fit failed"), log_path)
    return result["estimator"]


@contextmanager
def overall_deadline(seconds):
    if not hasattr(signal, "setitimer") or not hasattr(signal, "ITIMER_REAL"):
        raise GateError("strict overall deadline requires POSIX interval timers")
    def timeout_handler(_signum, _frame):
        raise TimeoutError(f"overall execution exceeded {seconds} seconds")
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    prior = signal.getitimer(signal.ITIMER_REAL)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *prior)
        signal.signal(signal.SIGALRM, old_handler)


def _matrix(keys, feature_map):
    rows = []
    for key in keys:
        if key not in feature_map:
            raise ValueError("missing canonical feature key")
        row = feature_map[key]
        rows.append([row[name] for name in FEATURE_COLUMNS])
    return np.asarray(rows, dtype=object)


def _make_pipeline(predictor):
    params = predictor["parameters"]
    numeric_features = [name for name in predictor["inputs"] if name != "module_one_hot"]
    if not numeric_features or any(name not in NUMERIC_COLUMNS for name in numeric_features):
        raise GateError("predictor feature list is not supported by the fixed matrix schema")
    numeric_indices = [FEATURE_COLUMNS.index(c) for c in numeric_features]
    transformer = ColumnTransformer([
        ("numeric", StandardScaler(), numeric_indices),
        ("module", OneHotEncoder(handle_unknown="error", sparse_output=False), [FEATURE_COLUMNS.index("module")]),
    ], remainder="drop")
    return Pipeline([("features", transformer), ("model", LogisticRegression(**params))])


def _probabilities(model, calibrator, x):
    raw = np.asarray(model.decision_function(x), dtype=float).reshape(-1, 1)
    if not np.isfinite(raw).all():
        raise ValueError("non-finite decision score; probability unavailable")
    probs = np.asarray(calibrator.predict_proba(raw), dtype=float)[:, 1]
    for value in probs:
        _probability(value)
    return probs, raw.ravel()


def _metrics(labels, calibrated, raw, modules, keys):
    y = np.asarray(labels, dtype=np.int8)
    p = np.asarray(calibrated, dtype=float)
    rp = np.asarray(raw, dtype=float)
    if len(set(y.tolist())) != 2:
        raise ValueError("complete-case metrics require both classes")
    bins = []
    for i in range(10):
        mask = ((p >= i / 10) & (p < (i + 1) / 10)) if i < 9 else ((p >= .9) & (p <= 1))
        bins.append({"bin": i, "count": int(mask.sum()),
                     "mean_probability": None if not mask.any() else float(p[mask].mean()),
                     "observed_prevalence": None if not mask.any() else float(y[mask].mean())})
    module_results = {}
    for module in sorted(set(modules)):
        mask = np.asarray([m == module for m in modules])
        module_results[module] = {"records": int(mask.sum()), "positives": int(y[mask].sum()),
                                  "prevalence": float(y[mask].mean()),
                                  "brier": float(brier_score_loss(y[mask], p[mask]))}
    return {"average_precision": float(average_precision_score(y, p)),
            "prevalence_baseline": float(y.mean()), "auroc": float(roc_auc_score(y, p)),
            "brier": float(brier_score_loss(y, p)), "raw_brier": float(brier_score_loss(y, rp)),
            "reliability_10_bins": bins, "module_descriptive_only": module_results,
            "scope": "known-label complete cases only; conditional on observed labels and selected known-eligible records"}


def _run_experiment_after_gate(*, data_bundle, config, run_dir: Path, fit_executor=fit_with_deadline,
                               execution_context, estimator_factory=_make_pipeline, calibrator_factory=None, clock=time.monotonic,
                               absolute_deadline=None):
    """Internal experiment core. CLI reaches this only after all gates pass."""
    _validate_execution_context(execution_context, config, run_dir)
    limits = config["execution_limits"]
    total_seconds = limits["overall_execution_timeout_seconds"]
    per_fit = limits["base_estimator_fit_timeout_seconds"]
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    unexpected = [p for p in run_dir.iterdir() if p.name not in {"authorization_receipt.json", "run_manifest.json"}]
    if unexpected:
        raise GateError("run directory must be new and empty; preserve earlier runs")
    run_manifest_path = run_dir / "run_manifest.json"
    if not run_manifest_path.exists():
        _atomic_json(run_manifest_path, {"overall_status": "incomplete", "outputs_usable": False,
                                         "run_id": execution_context.run_id, "hashes": execution_context.hashes})
    else:
        initial_state = json.loads(run_manifest_path.read_text())
        if initial_state.get("overall_status") != "incomplete" or initial_state.get("outputs_usable") is not False:
            raise GateError("run manifest is not in a fresh incomplete state")
    ledger = StageLedger(run_dir, clock)
    stage = "setup"
    try:
        source_y = data_bundle["source_labels"]
        source_features = data_bundle["source_features"]
        known_y = data_bundle["known_target_labels"]
        uncertain_features = data_bundle["uncertain_target_features"]
        split_hash, fit_keys, calibration_keys = split_fingerprint(source_y, source_features, config["split"])
        y_fit = np.asarray([source_y[k] for k in fit_keys], dtype=np.int8)
        y_cal = np.asarray([source_y[k] for k in calibration_keys], dtype=np.int8)
        if set(y_fit) != {0, 1} or set(y_cal) != {0, 1} or int(y_cal.sum()) < config["split"]["minimum_calibration_positive_records"]:
            raise ValueError("prespecified fixed source split does not satisfy class/calibration minimum")
        if not set(fit_keys).isdisjoint(calibration_keys):
            raise ValueError("source split key overlap")
        if {k[0] for k in fit_keys} & {k[0] for k in calibration_keys}:
            raise ValueError("source student overlap between fit and calibration partitions")
        ledger.set("split", "passed", fit_records=len(fit_keys), calibration_records=len(calibration_keys), split_sha256=split_hash)
        x_fit = _matrix(fit_keys, source_features)
        x_cal = _matrix(calibration_keys, source_features)
        known_keys = sorted(known_y)
        uncertain_keys = sorted(uncertain_features)
        x_known = _matrix(known_keys, data_bundle["known_target_features"])
        x_uncertain = _matrix(uncertain_keys, uncertain_features)
        deadline = absolute_deadline if absolute_deadline is not None else clock() + total_seconds
        if not set(config["study"]["shared_modules"]).issubset({source_features[k]["module"] for k in fit_keys}):
            raise ValueError("a shared module category is absent from source fit; stop without changing split")
        calibrated = {}
        raw_prob = {}
        for predictor in config["predictors"]:
            name = predictor["name"]
            if name == "source_prevalence_constant":
                continue
            stage = f"base_fit_{name}"
            model = estimator_factory(predictor)
            log_path = run_dir / "logs" / f"{name}_base_fit.json"
            _validate_execution_context(execution_context, config, run_dir)
            budget = fit_budget(started_at=deadline-total_seconds, now=clock(), total_cap=total_seconds, per_fit_cap=per_fit)
            model = fit_executor(model, x_fit, y_fit, timeout_seconds=budget, log_path=log_path,
                                 execution_context=execution_context, config=config)
            _validate_execution_context(execution_context, config, run_dir)
            stage = f"base_coefficients_{name}"
            final_model = model.named_steps["model"] if hasattr(model, "named_steps") else model
            for attr in ("coef_", "intercept_"):
                if hasattr(final_model, attr) and not np.isfinite(np.asarray(getattr(final_model, attr), dtype=float)).all():
                    raise FitFailure("non-finite coefficients", log_path)
            stage = f"calibration_fit_{name}"
            cal_scores = np.asarray(model.decision_function(x_cal), dtype=float).reshape(-1, 1)
            if calibrator_factory is None:
                cal = LogisticRegression(**config["calibration"]["calibrator_parameters"])
            else:
                cal = calibrator_factory(config["calibration"]["calibrator_parameters"])
            _validate_execution_context(execution_context, config, run_dir)
            budget = fit_budget(started_at=deadline-total_seconds, now=clock(), total_cap=total_seconds, per_fit_cap=per_fit)
            cal = fit_executor(cal, cal_scores, y_cal, timeout_seconds=budget,
                               log_path=run_dir / "logs" / f"{name}_calibration_fit.json",
                               execution_context=execution_context, config=config)
            _validate_execution_context(execution_context, config, run_dir)
            if hasattr(cal, "coef_") and not np.isfinite(np.asarray(cal.coef_, dtype=float)).all():
                raise FitFailure("non-finite calibrator coefficients", run_dir / "logs" / f"{name}_calibration_fit.json")
            if hasattr(cal, "intercept_") and not np.isfinite(np.asarray(cal.intercept_, dtype=float)).all():
                raise FitFailure("non-finite calibrator intercept", run_dir / "logs" / f"{name}_calibration_fit.json")
            model_artifacts = run_dir / "frozen_models"
            model_artifacts.mkdir(parents=True, exist_ok=True)
            _atomic_write(model_artifacts / f"{name}.pkl", pickle.dumps({"base_model": model, "calibrator": cal}, protocol=5))
            ledger.set(f"base_and_calibrator_{name}", "passed", base_fit_count=1,
                       calibration_fit_count=1, base_model_refit=False)
            stage = f"target_scores_{name}"
            p_known, raw_known = _probabilities(model, cal, x_known)
            p_uncertain, raw_uncertain = _probabilities(model, cal, x_uncertain)
            if len(p_known) != len(known_keys) or len(p_uncertain) != len(uncertain_keys):
                raise ValueError("invalid target prediction coverage")
            calibrated[name] = {"known": p_known, "uncertain": p_uncertain}
            raw_prob[name] = {"known": 1 / (1 + np.exp(-np.clip(raw_known, -700, 700))),
                              "uncertain": 1 / (1 + np.exp(-np.clip(raw_uncertain, -700, 700)))}

        reg_name, activity_name = "registration_logistic", "registration_plus_vle_logistic"
        stage = "primary_bounds"
        bounds = primary_target_bounds(
            known_unit_keys=known_keys, uncertain_unit_keys=uncertain_keys,
            known_registration_scores=list(zip(known_keys, calibrated[reg_name]["known"])),
            known_activity_scores=list(zip(known_keys, calibrated[activity_name]["known"])),
            known_labels=list(zip(known_keys, [known_y[k] for k in known_keys])),
            uncertain_registration_scores=list(zip(uncertain_keys, calibrated[reg_name]["uncertain"])),
            uncertain_activity_scores=list(zip(uncertain_keys, calibrated[activity_name]["uncertain"])))
        ledger.set("primary_bounds", "passed", known_records=bounds.known_records,
                   uncertain_records=bounds.uncertain_records,
                   lower=str(bounds.lower), upper=str(bounds.upper), interpretation="finite-cohort identification bounds only")

        metrics = {}
        for name in (reg_name, activity_name):
            metrics[name] = _metrics([known_y[k] for k in known_keys], calibrated[name]["known"],
                                     raw_prob[name]["known"], [k[1] for k in known_keys], known_keys)
        prevalence = float(y_fit.mean())
        complete_labels = [known_y[k] for k in known_keys]
        constant = np.full(len(known_keys), prevalence, dtype=float)
        metrics["source_prevalence_constant"] = _metrics(complete_labels, constant, constant,
                                                         [k[1] for k in known_keys], known_keys)
        report = {"scope": "retrospective temporal validation; complete-case diagnostics and finite-cohort bounds",
                  "source_prevalence_reference": prevalence,
                  "primary_paired_brier_improvement_bounds": {"lower": str(bounds.lower), "upper": str(bounds.upper),
                      "positive_favors_activity": True, "known_records": bounds.known_records,
                      "uncertain_records": bounds.uncertain_records, "fixed_predictors_only": True},
                  "complete_case_metrics": metrics,
                  "confidence_region_for_partial_bounds": "none; finite-cohort identification endpoints only",
                  "limitations": ["bounds are conditional on accepted labels and fixed fitted/calibrated probabilities",
                      "no fitting/calibration uncertainty", "no source-selection uncertainty", "no future-population inference",
                      "no administrative-recording-delay correction", "AP/AUROC/reliability are not partially identified",
                      "module summaries are descriptive only; no module-specific superiority claims"]}
        predictions = []
        for name in (reg_name, activity_name):
            for kind, keys in (("known", known_keys), ("uncertain", uncertain_keys)):
                for i, key in enumerate(keys):
                    predictions.append({"id_student": key[0], "code_module": key[1], "code_presentation": key[2],
                                        "target_kind": kind, "predictor": name,
                                        "calibrated_probability": float(calibrated[name][kind][i]),
                                        "raw_probability": float(raw_prob[name][kind][i]),
                                        "known_label": known_y[key] if kind == "known" else ""})
        stage = "artifacts"
        _atomic_json(run_dir / "aggregate_metrics.json", report)
        from io import StringIO
        sio = StringIO()
        writer = csv.DictWriter(sio, fieldnames=list(predictions[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(predictions)
        _atomic_write(run_dir / "private_predictions.csv", sio.getvalue().encode())
        ledger.set("results", "passed", aggregate_metrics="aggregate_metrics.json",
                   private_predictions="private_predictions.csv", row_level_data_git_ignored=True)
        ledger.write("complete")
        _atomic_json(run_manifest_path, {"overall_status": "complete", "outputs_usable": True,
                     "run_id": execution_context.run_id, "hashes": execution_context.hashes,
                     "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                     "artifact_sha256": _artifact_hashes(run_dir)})
        if not verify_complete_run(run_dir):
            raise RuntimeError("completion marker verification failed; outputs remain unusable")
        _update_claim(execution_context, "complete")
        return report
    except BaseException as exc:
        failure_log = run_dir / "logs" / "runner_failure.json"
        _atomic_json(failure_log, {"stage": stage, "error": str(exc),
                                   "traceback": traceback.format_exc(),
                                   "model_fitting_or_prediction_attempted": stage not in {"setup", "split"}})
        ledger.fail(stage, exc, getattr(exc, "log_path", failure_log))
        _atomic_json(run_manifest_path, {"overall_status": "failed", "outputs_usable": False,
                     "run_id": execution_context.run_id, "hashes": execution_context.hashes,
                     "error": str(exc), "traceback": traceback.format_exc()})
        _update_claim(execution_context, "failed")
        if isinstance(exc, (TimeoutError, FitFailure)):
            raise
        raise RuntimeError(f"runner failed at {stage}: {exc}") from exc


def run_authorized_from_data(*, data_dir, config, manifest, run_id, parent_authorization_reference,
                             explicit_cli_authorization, data_manifest_path):
    """Only supported actual-fit entry; local manifest is an audit receipt, not authorization."""
    if not explicit_cli_authorization:
        raise GateError("fitting requires explicit parent task authorization and CLI opt-in")
    if not _valid_run_id(run_id):
        raise GateError("a valid one-time run ID is required")
    if not isinstance(parent_authorization_reference, str) or not parent_authorization_reference.strip():
        raise GateError("an explicit authenticated parent task authorization reference is required")
    run_dir = verify_private_output(ROOT / "data/derived" / run_id)
    claim_path = _claim_path(run_id)
    if claim_path.exists() or claim_path.with_suffix(".claiming").exists() or run_dir.exists():
        raise GateError("run ID is already claimed or output exists; replay is prohibited")
    claim_context = None
    output_created = False
    try:
        if config.get("document_status") != "LOCKED" or not config.get("protocol_lock") or not config.get("execution_authorization"):
            raise GateError("candidate protocol remains draft, unlocked, or unauthorized")
        if config != json.loads((ROOT / "protocol/candidate_protocol.json").read_text()):
            raise GateError("execution config differs from the reviewed protocol file")
        if manifest.get("review_status") != "APPROVED" or not manifest.get("reviewer"):
            raise GateError("review approval and reviewer are required")
        runtime = verify_runtime_and_limits(config)
        total = config["execution_limits"]["overall_execution_timeout_seconds"]
        deadline_started = time.monotonic()
        with overall_deadline(total), threadpool_limits(limits=config["execution_limits"]["cpu_threads_max"]):
            data_hash = verify_data_files(Path(data_dir), Path(data_manifest_path))
            preflight, target_features, source_features, target_labels, source_labels = build_target_preflight(Path(data_dir))
            if preflight["shared_modules"] != config["study"]["shared_modules"]:
                raise GateError("actual shared modules differ from the reviewed protocol")
            split_hash, _, _ = split_fingerprint(source_labels, source_features, config["split"])
            checks = {"protocol_hash": protocol_fingerprint(), "data_hash": data_hash,
                      "split_hash": split_hash, "code_hash": code_fingerprint()}
            verify_lock(config=config, manifest=manifest, explicit_cli_authorization=True,
                        parent_authorization_reference=parent_authorization_reference, **checks)
            bundle = {"source_labels": source_labels, "source_features": source_features,
                      "known_target_labels": target_labels, "known_target_features": target_features,
                      "uncertain_target_features": {k: target_features[k] for k in set(target_features) - set(target_labels)}}
            hashes = {"protocol_sha256": checks["protocol_hash"], "data_sha256": checks["data_hash"],
                      "split_sha256": checks["split_hash"], "code_sha256": checks["code_hash"]}
            claim = {"run_id": run_id, "parent_authorization_reference": parent_authorization_reference,
                     "output_path": str(run_dir), "hashes": hashes, "overall_status": "claimed",
                     "claimed_at_utc": datetime.now(timezone.utc).isoformat()}
            _write_claim_exclusive(claim_path, claim)
            claim_context = ExecutionContext(run_id=run_id,
                parent_authorization_reference=parent_authorization_reference, output_dir=run_dir,
                claim_path=claim_path, hashes=hashes, data_dir=Path(data_dir).resolve(),
                data_manifest_path=Path(data_manifest_path).resolve(), data_bundle=bundle)
            run_dir.mkdir(parents=False, exist_ok=False)
            output_created = True
            _atomic_json(run_dir / "authorization_receipt.json", {"status": "authorized_not_completed",
                "local_protocol_receipt_only": True, "parent_authorization_reference": parent_authorization_reference,
                "reviewer": manifest["reviewer"], "hashes": hashes, "runtime": runtime, "run_id": run_id})
            _atomic_json(run_dir / "run_manifest.json", {"overall_status": "incomplete", "outputs_usable": False,
                         "run_id": run_id, "hashes": hashes})
            _update_claim(claim_context, "running")
            report = _run_experiment_after_gate(data_bundle=bundle, config=config, run_dir=run_dir,
                execution_context=claim_context, absolute_deadline=deadline_started + total)
            return report
    except BaseException as exc:
        if claim_context is not None:
            _update_claim(claim_context, "failed")
            if output_created:
                _atomic_json(run_dir / "run_manifest.json", {"overall_status": "failed", "outputs_usable": False,
                             "run_id": run_id, "hashes": claim_context.hashes, "error": str(exc),
                             "traceback": traceback.format_exc()})
        raise


def dry_run_summary(config_path: Path, data_manifest: Path):
    config = json.loads(config_path.read_text())
    return {"status": "DRY_RUN_NO_FIT", "document_status": config["document_status"],
            "protocol_lock": config["protocol_lock"], "execution_authorization": config["execution_authorization"],
            "limits": config["execution_limits"], "data_manifest_present": data_manifest.is_file(),
            "fit_called": False, "prediction_called": False,
            "next_gate": "final reviewed lock hashes and parent authorization are required"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/extracted"))
    parser.add_argument("--config", type=Path, default=Path("protocol/candidate_protocol.json"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--parent-authorization-reference")
    parser.add_argument("--execute-authorized", action="store_true")
    args = parser.parse_args()
    if not args.execute_authorized:
        print(json.dumps(dry_run_summary(args.config, ROOT / "audit/data_manifest.json"), indent=2))
        return
    if args.manifest is None or not args.run_id or not args.parent_authorization_reference:
        raise SystemExit("execution requires --manifest, --run-id, and a parent authorization reference from the authenticated task")
    config = json.loads(args.config.read_text())
    manifest = json.loads(args.manifest.read_text())
    report = run_authorized_from_data(data_dir=args.data_dir, config=config, manifest=manifest,
                                      run_id=args.run_id,
                                      parent_authorization_reference=args.parent_authorization_reference,
                                      explicit_cli_authorization=True,
                                      data_manifest_path=ROOT / "audit/data_manifest.json")
    print(json.dumps({"status": "complete", "output": str(ROOT / "data/derived" / args.run_id),
                      "primary_bounds": report["primary_paired_brier_improvement_bounds"]}, indent=2))


if __name__ == "__main__":
    main()
