"""Aggregate joint registration/date/status quality before cohort exclusions.

This is an extension to the first audit. It reads raw OULAD registration rows,
emits aggregate counts only, and does not alter the original audit artifacts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

COHORTS = ("2013J", "2014J")
STATUS_ENUMS = {"withdrawn": "Withdrawn", "pass": "Pass", "fail": "Fail", "distinction": "Distinction"}
SOURCE = "https://archive.ics.uci.edu/static/public/349/open+university+learning+analytics+dataset.zip"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_day(value: str | None) -> tuple[int | None, str]:
    if value is None or value.strip() in ("", "?"):
        return None, "missing"
    try:
        return int(value), "known"
    except (ValueError, TypeError):
        return None, "invalid"


def joint_record(registration_raw: str | None, unregistration_raw: str | None,
                 final_status_raw: str | None, duration: int | None) -> dict[str, str]:
    """Describe all label/eligibility dimensions without short-circuiting exclusions."""
    reg, reg_condition = parse_day(registration_raw)
    unreg, unreg_condition = parse_day(unregistration_raw)
    unreg_band = ("missing" if unreg_condition == "missing" else
                  "invalid" if unreg_condition == "invalid" else
                  "<=28" if unreg <= 28 else
                  "day_29_56" if unreg <= 56 else "after_day_56")
    raw_status = (final_status_raw or "").strip()
    normalized = raw_status.casefold()
    status_condition = STATUS_ENUMS.get(normalized, "missing" if not normalized else "other")

    if unreg_condition == "missing":
        relation = "missing_date_withdrawn_status" if status_condition == "Withdrawn" else (
            "missing_date_nonwithdrawn_status" if status_condition in {"Pass", "Fail", "Distinction"} else "missing_date_status_unknown")
    elif unreg_condition == "invalid":
        relation = "invalid_date_status_withdrawn" if status_condition == "Withdrawn" else (
            "invalid_date_status_nonwithdrawn" if status_condition in {"Pass", "Fail", "Distinction"} else "invalid_date_status_unknown")
    elif status_condition == "Withdrawn":
        relation = "known_date_withdrawn_status"
    elif status_condition in {"Pass", "Fail", "Distinction"}:
        relation = "known_date_nonwithdrawn_status_contradiction"
    else:
        relation = "known_date_status_unknown"

    # Establish risk-set membership from known dates/status documentation, keeping unknowns explicit.
    if reg_condition == "known" and unreg_condition == "known" and unreg < reg:
        eligibility = "unknown_impossible_date_order"
    elif duration is None:
        eligibility = "unknown_presentation_length"
    elif duration <= 56:
        eligibility = "known_ineligible_short_presentation"
    elif unreg_band == "<=28":
        eligibility = "known_ineligible_withdrawn_by_day_28"
    elif reg_condition != "known":
        eligibility = "unknown_registration_date"
    elif reg > 28:
        eligibility = "known_ineligible_registered_after_day_28"
    elif unreg_band in {"day_29_56", "after_day_56"}:
        eligibility = "known_eligible_at_day_28"
    elif unreg_band == "missing" and status_condition in {"Pass", "Fail", "Distinction"}:
        eligibility = "known_eligible_at_day_28"
    elif unreg_band == "missing" and status_condition == "Withdrawn":
        eligibility = "unknown_withdrawn_date_missing"
    else:
        eligibility = "unknown_event_or_status_date"

    if eligibility.startswith("known_ineligible"):
        outcome = "not_at_risk_at_day_28"
    elif eligibility != "known_eligible_at_day_28":
        # Retain explicit event timing when available even if missing registration
        # prevents deciding whether the record belongs to the day-28 risk set.
        if unreg_band == "day_29_56" and status_condition == "Withdrawn":
            outcome = "event_day_29_56_known_but_risk_set_unknown"
        elif unreg_band == "after_day_56" and status_condition == "Withdrawn":
            outcome = "event_after_day_56_known_but_risk_set_unknown"
        elif unreg_band == "missing" and status_condition == "Withdrawn":
            outcome = "unknown_withdrawal_time_and_risk_set_membership"
        elif unreg_band == "invalid":
            outcome = "unknown_invalid_withdrawal_date_and_risk_set_membership"
        elif unreg_band == "<=28" and status_condition == "Withdrawn":
            outcome = "pre_landmark_event_known_but_risk_set_unknown"
        else:
            outcome = "outcome_unknown_risk_set_membership"
    elif unreg_band == "day_29_56":
        outcome = "positive_day_29_56" if status_condition == "Withdrawn" else "unknown_date_status_contradiction"
    elif unreg_band == "after_day_56":
        outcome = "negative_after_day_56" if status_condition == "Withdrawn" else "unknown_date_status_contradiction"
    elif unreg_band == "missing":
        outcome = "negative_no_withdrawal_date_nonwithdrawn_status" if status_condition in {"Pass", "Fail", "Distinction"} else "unknown_withdrawal_time_or_status"
    else:
        outcome = "unknown_invalid_withdrawal_date_or_status"

    return {"registration_date_condition": reg_condition,
            "withdrawal_time_band": unreg_band,
            "final_status_condition": status_condition,
            "date_status_condition": relation,
            "day28_eligibility": eligibility,
            "horizon_outcome_given_eligibility": outcome}


def make_tables(data_dir: Path) -> tuple[list[dict], list[dict]]:
    courses = {}
    with (data_dir / "courses.csv").open(newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try:
                courses[(r["code_module"], r["code_presentation"])] = int(r["module_presentation_length"])
            except (ValueError, TypeError):
                courses[(r["code_module"], r["code_presentation"])] = None
    info = {}
    with (data_dir / "studentInfo.csv").open(newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            info[(r["id_student"], r["code_module"], r["code_presentation"])] = r.get("final_result", "")

    joint = Counter()
    summary = Counter()
    with (data_dir / "studentRegistration.csv").open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            cohort = row["code_presentation"]
            if cohort not in COHORTS:
                continue
            module = row["code_module"]
            raw_status = info.get((row["id_student"], module, cohort), "")
            quality = joint_record(row.get("date_registration"), row.get("date_unregistration"), raw_status,
                                   courses.get((module, cohort)))
            joint[(cohort, module, *quality.values())] += 1
            summary[(cohort, module, "raw_records")] += 1
            summary[(cohort, module, "registration_missing_records")] += quality["registration_date_condition"] == "missing"
            summary[(cohort, module, f"day28_eligibility:{quality['day28_eligibility']}")] += 1
            summary[(cohort, module, f"horizon_outcome:{quality['horizon_outcome_given_eligibility']}")] += 1
            summary[(cohort, module, f"date_status:{quality['date_status_condition']}")] += 1

    joint_rows = [{"cohort":k[0],"module":k[1],"registration_date_condition":k[2],"withdrawal_time_band":k[3],
                   "final_status_condition":k[4],"date_status_condition":k[5],"day28_eligibility":k[6],
                   "horizon_outcome_given_eligibility":k[7],"records":v} for k,v in sorted(joint.items())]
    summary_rows = [{"cohort":k[0],"module":k[1],"measure":k[2],"records":v} for k,v in sorted(summary.items())]
    return joint_rows, summary_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("audit"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    joint, summary = make_tables(args.data_dir)
    write_csv(args.out_dir / "label_availability_joint.csv", joint)
    write_csv(args.out_dir / "label_availability_summary.csv", summary)
    manifest = {"source_url": SOURCE, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "software": {"python": platform.python_version(), "script_sha256": sha256(Path(__file__))},
                "scope": "Every source registration row in 2013J and 2014J, before overlap removal, common-module filtering, eligibility filtering, or mutually exclusive exclusions; aggregate output only.",
                "inputs": {name: {"bytes": (args.data_dir / name).stat().st_size,
                                   "sha256": sha256(args.data_dir / name)}
                           for name in ("courses.csv", "studentInfo.csv", "studentRegistration.csv")},
                "outputs": {name: {"bytes": (args.out_dir / name).stat().st_size,
                                    "sha256": sha256(args.out_dir / name)}
                            for name in ("label_availability_joint.csv", "label_availability_summary.csv")}}
    (args.out_dir / "label_availability_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps({"joint_rows": len(joint), "summary_rows": len(summary), "outputs": ["label_availability_joint.csv", "label_availability_summary.csv", "label_availability_manifest.json"]}, indent=2))


if __name__ == "__main__":
    main()
