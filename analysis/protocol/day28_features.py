"""Non-fitting, key-preserving day-28 feature preflight for the audited cohort.

Only aggregate coverage is written. No row-level feature matrix or student key is
persisted. The implementation intentionally contains no sklearn fit/predict call.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

from audit_label_availability import joint_record, parse_day

KEY_FIELDS = ("id_student", "code_module", "code_presentation")
BUCKETS = ((0, 6), (7, 13), (14, 20), (21, 27), (28, 28))
EXPECTED_ACCEPTED = 6771
EXPECTED_UNCERTAIN = 58


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        yield from csv.DictReader(f)


def unit_key(row: dict[str, str]) -> tuple[str, str, str]:
    return tuple(row[name] for name in KEY_FIELDS)


def _unique_index(rows, label: str):
    result = {}
    for row in rows:
        key = unit_key(row)
        if key in result:
            raise ValueError(f"duplicate {label} canonical unit key")
        result[key] = row
    return result


def key_set_fingerprint(keys) -> str:
    payload = "".join("\t".join(key) + "\n" for key in sorted(keys)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def aggregate_day28(events):
    """Pure feature transform; events are (integer course-day, nonnegative clicks)."""
    clicks = [0] * len(BUCKETS)
    active_days = set()
    total = 0
    for day, count in events:
        day, count = int(day), int(count)
        if count < 0:
            raise ValueError("negative click count")
        if not 0 <= day <= 28:
            continue
        total += count
        active_days.add(day)
        for i, (lo, hi) in enumerate(BUCKETS):
            if lo <= day <= hi:
                clicks[i] += count
                break
    no_activity = not active_days
    return {
        "log1p_total_clicks": math.log1p(total),
        "active_days": len(active_days),
        "recency_days_at_day28": 0 if no_activity else 28 - max(active_days),
        "no_activity": int(no_activity),
        **{f"log1p_clicks_days_{lo}_{hi}": math.log1p(value)
           for (lo, hi), value in zip(BUCKETS, clicks)},
    }


def signed_registration_lead_time(date_registration: int) -> int:
    return -int(date_registration)


def build_target_preflight(data_dir: Path):
    courses = {}
    for row in read_csv(data_dir / "courses.csv"):
        courses[(row["code_module"], row["code_presentation"])] = int(row["module_presentation_length"])
    common = sorted({m for m, p in courses if p == "2013J"} & {m for m, p in courses if p == "2014J"})
    train_ids = {r["id_student"] for name in ("studentInfo.csv", "studentRegistration.csv")
                 for r in read_csv(data_dir / name) if r["code_presentation"] == "2013J"}
    info = _unique_index((r for r in read_csv(data_dir / "studentInfo.csv") if r["code_presentation"] == "2014J"), "studentInfo")
    regs = _unique_index((r for r in read_csv(data_dir / "studentRegistration.csv")
                          if r["code_presentation"] == "2014J"), "studentRegistration")
    source_info = _unique_index((r for r in read_csv(data_dir / "studentInfo.csv") if r["code_presentation"] == "2013J"), "source studentInfo")
    source_regs = _unique_index((r for r in read_csv(data_dir / "studentRegistration.csv")
                                 if r["code_presentation"] == "2013J"), "source studentRegistration")

    accepted_labels = {}
    source_labels = {}
    uncertain = set()
    eligible_target_registrations = []
    for key, reg in regs.items():
        student, module, presentation = key
        if presentation != "2014J" or module not in common or student in train_ids:
            continue
        status = info.get(key, {}).get("final_result", "")
        duration = courses.get((module, presentation))
        q = joint_record(reg.get("date_registration"), reg.get("date_unregistration"), status, duration)
        eligibility = q["day28_eligibility"]
        outcome = q["horizon_outcome_given_eligibility"]
        rday, rcondition = parse_day(reg.get("date_registration"))
        if eligibility == "known_eligible_at_day_28" and outcome in {"positive_day_29_56", "negative_after_day_56", "negative_no_withdrawal_date_nonwithdrawn_status"}:
            accepted_labels[key] = int(outcome == "positive_day_29_56")
            eligible_target_registrations.append((key, -rday, module))
        elif eligibility == "unknown_withdrawn_date_missing" and rcondition == "known" and rday <= 28:
            uncertain.add(key)

    for key, reg in source_regs.items():
        student, module, presentation = key
        if presentation != "2013J" or module not in common:
            continue
        status = source_info.get(key, {}).get("final_result", "")
        q = joint_record(reg.get("date_registration"), reg.get("date_unregistration"), status,
                         courses.get((module, presentation)))
        if q["day28_eligibility"] == "known_eligible_at_day_28" and q["horizon_outcome_given_eligibility"] in {
            "positive_day_29_56", "negative_after_day_56", "negative_no_withdrawal_date_nonwithdrawn_status"
        }:
            source_labels[key] = int(q["horizon_outcome_given_eligibility"] == "positive_day_29_56")
    if len(source_labels) != 7676:
        raise ValueError(f"source modeling population mismatch: {len(source_labels)}")

    if len(accepted_labels) != EXPECTED_ACCEPTED or len(uncertain) != EXPECTED_UNCERTAIN:
        raise ValueError(f"target population mismatch: accepted={len(accepted_labels)}, uncertain={len(uncertain)}")
    target_keys = set(accepted_labels) | uncertain
    all_keys = target_keys | set(source_labels)
    if len(target_keys) != len(accepted_labels) + len(uncertain):
        raise ValueError("accepted and uncertain keys overlap")
    if target_keys & set(source_labels):
        raise ValueError("source and target canonical keys overlap")

    # Initialize every selected key before streaming VLE, preserving zero-activity units.
    events = defaultdict(list)
    later_rows_by_uncertain = defaultdict(int)
    for row in read_csv(data_dir / "studentVle.csv"):
        key = unit_key(row)
        if key not in all_keys:
            continue
        day = int(row["date"])
        clicks = int(row["sum_click"])
        if 0 <= day <= 28:
            events[key].append((day, clicks))
        elif day >= 29 and key in uncertain:
            later_rows_by_uncertain[key] += 1

    matrices = {}
    source_matrices = {}
    for key in sorted(all_keys):
        registration = regs[key] if key[2] == "2014J" else source_regs[key]
        reg_day, condition = parse_day(registration.get("date_registration"))
        if condition != "known" or reg_day > 28:
            raise ValueError("selected unit has no valid at-risk registration feature")
        feature_row = {"signed_registration_lead_time_days": signed_registration_lead_time(reg_day),
                       "module": key[1], **aggregate_day28(events[key])}
        (matrices if key[2] == "2014J" else source_matrices)[key] = feature_row
    if set(matrices) != target_keys or set(source_matrices) != set(source_labels):
        raise ValueError("feature key coverage mismatch")
    accepted_zero = sum(matrices[k]["no_activity"] for k in accepted_labels)
    uncertain_zero = sum(matrices[k]["no_activity"] for k in uncertain)
    later_rows = sum(later_rows_by_uncertain.values())
    later_units = sum(v > 0 for v in later_rows_by_uncertain.values())
    if (accepted_zero, uncertain_zero, later_rows, later_units) != (206, 29, 759, 7):
        raise ValueError("observed day-28 feature coverage disagrees with independent raw-data checks")

    return ({"status": "NON_FITTING_PREFLIGHT_PASS", "shared_modules": common,
             "accepted_known_target_units": len(accepted_labels), "uncertain_target_units": len(uncertain),
             "feature_matrix_units": len(matrices), "source_feature_matrix_units": len(source_matrices),
             "accepted_zero_activity_units": accepted_zero,
             "uncertain_zero_activity_units": uncertain_zero, "total_zero_activity_units": accepted_zero + uncertain_zero,
             "accepted_key_set_sha256": key_set_fingerprint(accepted_labels),
             "uncertain_key_set_sha256": key_set_fingerprint(uncertain),
             "uncertain_units_with_later_day_rows": later_units, "uncertain_later_day_rows_ignored": later_rows,
             "activity_days_inclusive": [0, 28], "weekly_bins_inclusive": [[0, 6], [7, 13], [14, 20], [21, 27], [28, 28]],
             "feature_names": ["signed_registration_lead_time_days", "module", "log1p_total_clicks", "active_days",
                               "recency_days_at_day28", "no_activity", "log1p_clicks_days_0_6", "log1p_clicks_days_7_13",
                               "log1p_clicks_days_14_20", "log1p_clicks_days_21_27", "log1p_clicks_days_28_28"],
             "preserves_all_canonical_keys": True, "duplicate_or_lost_keys": 0,
             "learned_transformations_fit": False, "estimator_or_calibrator_fit_or_prediction": False},
            matrices, source_matrices, accepted_labels, source_labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/extracted"))
    parser.add_argument("--output", type=Path, default=Path("protocol/feature_preflight.json"))
    args = parser.parse_args()
    report, *_ = build_target_preflight(args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
