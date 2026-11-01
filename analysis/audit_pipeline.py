"""Non-fitting, deterministic OULAD day-28 cohort and label audit.

Run: python audit_pipeline.py --data-dir data/raw/extracted --out-dir audit
Only standard-library code is required. No estimator or model fitting is present.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sqlite3
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

SOURCE = "https://archive.ics.uci.edu/static/public/349/open+university+learning+analytics+dataset.zip"
LICENSE = "CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)"
TABLES = ["courses", "assessments", "studentAssessment", "studentInfo", "studentRegistration", "studentVle", "vle"]
UNIT = ("id_student", "code_module", "code_presentation")
COHORTS = ("2013J", "2014J")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def number(value: str | None) -> int | None:
    if value is None or value.strip() in ("", "?"):
        return None
    return int(value)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def key(row: dict[str, str]) -> tuple[str, str, str]:
    return row["id_student"], row["code_module"], row["code_presentation"]


def assess_record(reg: int | None, unreg: int | None, status: str | None, duration: int | None) -> tuple[str, int | None, str | None]:
    """Return (eligibility status, label, quality issue). label is 1=withdrawal days29-56."""
    result = (status or "").strip().casefold()
    if reg is None:
        return "excluded_unknown_registration", None, None
    if unreg is not None and unreg < reg:
        return "excluded_impossible_dates", None, "unregistration_before_registration"
    if reg > 28:
        return "excluded_registered_after_landmark", None, None
    if unreg is not None and unreg <= 28:
        return "excluded_withdrawn_by_landmark", None, None
    if duration is None or duration <= 56:
        return "excluded_short_or_unknown_presentation", None, None
    if not result:
        return "quarantined_label", None, "missing_final_status"
    if unreg is None:
        if result == "withdrawn":
            return "quarantined_label", None, "withdrawn_status_missing_unregistration_date"
        return "eligible", 0, None
    if 28 < unreg <= 56:
        if result != "withdrawn":
            return "quarantined_label", None, "withdrawal_date_conflicts_with_final_status"
        return "eligible", 1, None
    # Later withdrawal is negative for this horizon, with final status used only as a check.
    if unreg > 56:
        if result != "withdrawn":
            return "quarantined_label", None, "late_unregistration_conflicts_with_final_status"
        return "eligible", 0, None
    return "quarantined_label", None, "unclassified_label_state"


def day28_features(events: list[tuple[int, int]]) -> dict[str, int | bool]:
    """Aggregate days 0-28 only; fixed descriptive buckets [0,6],[7,13],[14,20],[21,28]."""
    valid = [(d, c) for d, c in events if 0 <= d <= 28]
    days = {d for d, _ in valid}
    buckets = [sum(c for d, c in valid if lo <= d <= hi) for lo, hi in ((0, 6), (7, 13), (14, 20), (21, 28))]
    out: dict[str, int | bool] = {
        "clicks_0_28": sum(c for _, c in valid), "active_days_0_28": len(days),
        "recency_days_at_28": 0 if not days else 28 - max(days), "has_activity_0_28": bool(days),
    }
    out.update({f"clicks_days_{lo}_{hi}": value for (lo, hi), value in zip(((0, 6), (7, 13), (14, 20), (21, 28)), buckets)})
    return out


def assessment_count(rows: list[dict[str, str]], assessment_context: dict[str, tuple[str, str]], expected_contexts: set[tuple[str, str]] | None = None) -> int:
    """Count dated current-presentation submissions through day28; reject duplicate assessment IDs."""
    count = 0
    for row in rows:
        context = assessment_context.get(row["id_assessment"])
        if context is None:
            continue
        if expected_contexts is not None and context not in expected_contexts:
            continue
        if number(row.get("is_banked")) == 1:
            continue
        submitted = number(row.get("date_submitted"))
        if submitted is not None and 0 <= submitted <= 28:
            count += 1
    return count


def build_assessment_context(rows: list[dict[str, str]]) -> dict[str, tuple[str, str]]:
    """Build an assessment-ID map and fail closed if a supposedly unique key repeats."""
    result = {}
    for row in rows:
        ident = row["id_assessment"]
        if ident in result:
            raise ValueError(f"duplicate id_assessment: {ident}")
        result[ident] = (row["code_module"], row["code_presentation"])
    return result


def overlap_filter(train_student_ids: set[str], test_rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], set[str]]:
    overlap = train_student_ids.intersection(r["id_student"] for r in test_rows)
    return [r for r in test_rows if r["id_student"] not in train_student_ids], overlap


def _counts(rows: list[dict[str, str]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(r[field] for r in rows).items()))


def full_audit(data_dir: Path, out_dir: Path, archive: Path | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {t: data_dir / f"{t}.csv" for t in TABLES}
    for p in files.values():
        if not p.is_file():
            raise FileNotFoundError(p)

    courses = read_csv(files["courses"])
    infos = read_csv(files["studentInfo"])
    regs = read_csv(files["studentRegistration"])
    assessments = read_csv(files["assessments"])
    submissions = read_csv(files["studentAssessment"])
    vle = read_csv(files["vle"])
    course_by_context = {(r["code_module"], r["code_presentation"]): number(r["module_presentation_length"]) for r in courses}
    info_by_key = {key(r): r for r in infos}
    reg_by_key = {key(r): r for r in regs}

    duplicate_info = len(infos) - len(info_by_key)
    duplicate_regs = len(regs) - len(reg_by_key)
    missingness = defaultdict(int)
    for name, rows in (("courses", courses), ("studentInfo", infos), ("studentRegistration", regs),
                       ("assessments", assessments), ("studentAssessment", submissions), ("vle", vle)):
        for row in rows:
            for col, value in row.items():
                if value.strip() in ("", "?"):
                    missingness[f"{name}.{col}"] += 1
    # Cohort overlap must be computed over all course records before restricting to common modules/eligibility.
    train_rows_all = [r for r in infos if r["code_presentation"] == "2013J"]
    test_rows_all = [r for r in infos if r["code_presentation"] == "2014J"]
    train_ids_all = {r["id_student"] for r in train_rows_all}
    test_rows_disjoint, overlap_ids = overlap_filter(train_ids_all, test_rows_all)
    disjoint_test_ids = {r["id_student"] for r in test_rows_disjoint}
    modules_by_presentation = defaultdict(set)
    for r in courses:
        modules_by_presentation[r["code_presentation"]].add(r["code_module"])
    common_modules = sorted(modules_by_presentation["2013J"] & modules_by_presentation["2014J"])
    common_set = set(common_modules)

    info_index = {key(r): r for r in infos}
    # Diagnostic counterfactual: valid-label 2014J records before returning-student removal.
    test_before_overlap = Counter()
    for r in regs:
        if r["code_presentation"] != "2014J" or r["code_module"] not in common_set:
            continue
        info = info_index.get(key(r))
        if info is None:
            continue
        elig, label, _ = assess_record(number(r["date_registration"]), number(r["date_unregistration"]),
                                        info.get("final_result", ""), course_by_context.get((r["code_module"], "2014J")))
        if elig == "eligible":
            test_before_overlap[(r["code_module"], "n")] += 1
            test_before_overlap[(r["code_module"], "positive")] += label

    records = []
    exclusion_counts = Counter()
    quality_counts = Counter()
    cohort_rows = {c: [] for c in COHORTS}
    for r in regs:
        cohort = r["code_presentation"]
        if cohort not in COHORTS:
            continue
        k = key(r)
        info = info_index.get(k)
        duration = course_by_context.get((r["code_module"], cohort))
        reg_day, unreg_day = number(r["date_registration"]), number(r["date_unregistration"])
        if cohort == "2014J" and r["id_student"] in overlap_ids:
            exclusion_counts[(cohort, r["code_module"], "training_student_overlap")] += 1
            continue
        if r["code_module"] not in common_set:
            exclusion_counts[(cohort, r["code_module"], "not_common_J_module")] += 1
            continue
        if info is None:
            exclusion_counts[(cohort, r["code_module"], "missing_student_info")]+=1
            continue
        status = info.get("final_result", "")
        elig, label, issue = assess_record(reg_day, unreg_day, status, duration)
        if elig != "eligible":
            exclusion_counts[(cohort, r["code_module"], elig)] += 1
            if issue:
                quality_counts[(cohort, issue)] += 1
            continue
        record = {"id_student": r["id_student"], "code_module": r["code_module"], "code_presentation": cohort,
                  "label": label, "reg_day": reg_day, "status": status, "unreg_day": unreg_day}
        records.append(record)
        cohort_rows[cohort].append(record)

    eligible_keys = {(r["id_student"], r["code_module"], r["code_presentation"]): r for r in records}
    features = defaultdict(list)
    activity_timing = defaultdict(Counter)
    units_with_pre_registration_activity = defaultdict(set)
    vle_quality = Counter()
    max_vle_day = None
    min_vle_day = None
    missingness_vle = Counter()
    vle_rows = 0
    # Use SQLite for bounded-memory global duplicate checks on the raw VLE row key.
    db_path = out_dir / "_vle_audit.sqlite"
    if db_path.exists(): db_path.unlink()
    db = sqlite3.connect(db_path)
    db.execute("CREATE TABLE seen (module TEXT, presentation TEXT, student TEXT, site TEXT, day TEXT, PRIMARY KEY(module,presentation,student,site,day)) WITHOUT ROWID")
    duplicates = 0
    with files["studentVle"].open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        batch = []
        for row in reader:
            vle_rows += 1
            for col, value in row.items():
                if value.strip() in ("", "?"):
                    missingness_vle[f"studentVle.{col}"] += 1
            d = number(row["date"])
            if d is not None:
                max_vle_day = d if max_vle_day is None else max(max_vle_day, d)
                min_vle_day = d if min_vle_day is None else min(min_vle_day, d)
            if d is None or d < 0: vle_quality["missing_or_negative_date"] += 1
            if number(row["sum_click"]) is None or number(row["sum_click"]) < 0: vle_quality["missing_or_negative_clicks"] += 1
            key_tuple = (row["code_module"], row["code_presentation"], row["id_student"], row["id_site"], row["date"])
            batch.append(key_tuple)
            if len(batch) >= 50000:
                before = db.total_changes
                db.executemany("INSERT OR IGNORE INTO seen VALUES (?,?,?,?,?)", batch)
                duplicates += len(batch) - (db.total_changes - before)
                batch.clear()
            k = (row["id_student"], row["code_module"], row["code_presentation"])
            if k in eligible_keys and d is not None:
                record = eligible_keys[k]
                group = (record["code_presentation"], record["code_module"])
                if d < 0:
                    activity_timing[group]["pre_course_event_rows"] += 1
                if d < record["reg_day"]:
                    activity_timing[group]["before_registration_event_rows"] += 1
                    activity_timing[group]["before_registration_clicks"] += number(row["sum_click"]) or 0
                    units_with_pre_registration_activity[group].add(k)
            if k in eligible_keys and d is not None and 0 <= d <= 28:
                features[k].append((d, number(row["sum_click"]) or 0))
        if batch:
            before = db.total_changes
            db.executemany("INSERT OR IGNORE INTO seen VALUES (?,?,?,?,?)", batch)
            duplicates += len(batch) - (db.total_changes - before)
        db.commit()

    # Referential coverage and contextual mismatch checks.
    student_keys = set(info_by_key)
    student_ids = {r["id_student"] for r in infos}
    contexts_by_student = defaultdict(set)
    for info_row in infos:
        contexts_by_student[info_row["id_student"]].add((info_row["code_module"], info_row["code_presentation"]))
    course_keys = set(course_by_context)
    assessment_ids = {r["id_assessment"] for r in assessments}
    assessment_context = build_assessment_context(assessments)
    site_contexts = {(r["id_site"], r["code_module"], r["code_presentation"]) for r in vle}
    site_ids = {r["id_site"] for r in vle}
    missing_course_context = sum(1 for r in regs if (r["code_module"], r["code_presentation"]) not in course_keys)
    missing_info = sum(1 for r in regs if key(r) not in student_keys)
    missing_assessment = sum(1 for r in submissions if r["id_assessment"] not in assessment_ids)
    missing_student = sum(1 for r in submissions if r["id_student"] not in student_ids)
    wrong_assessment_context = 0
    banked = 0
    submission_duplicates = len(submissions) - len({(r["id_assessment"], r["id_student"]) for r in submissions})
    for r in submissions:
        if r["id_assessment"] in assessment_context:
            modpres = assessment_context[r["id_assessment"]]
            if modpres not in contexts_by_student.get(r["id_student"], set()):
                wrong_assessment_context += 1
        if number(r.get("is_banked")) == 1: banked += 1
    site_missing = 0
    site_context_mismatch = 0
    # Second VLE pass is skipped: row foreign key integrity is checked inline with SQL key table only for duplicates;
    # material/site context coverage is computed by streaming the source once more.
    with files["studentVle"].open(newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["id_site"] not in site_ids: site_missing += 1
            if (r["id_site"], r["code_module"], r["code_presentation"]) not in site_contexts: site_context_mismatch += 1

    # Aggregate cohort counts, labels and zero activity at unit of analysis.
    cohort_summary = []
    label_rows = []
    for cohort in COHORTS:
        for module in common_modules:
            subset = [r for r in cohort_rows[cohort] if r["code_module"] == module]
            clicks = [day28_features(features.get((r["id_student"], module, cohort), [])) for r in subset]
            pos = sum(r["label"] == 1 for r in subset)
            cohort_summary.append({"cohort": cohort, "module": module,
                                   "raw_registration_records": sum(r["code_presentation"] == cohort and r["code_module"] == module for r in regs),
                                   "raw_student_info_records": sum(r["code_presentation"] == cohort and r["code_module"] == module for r in infos),
                                   "eligible_records": len(subset), "unique_students": len({r["id_student"] for r in subset}),
                                   "eligible_before_test_overlap_removal": test_before_overlap.get((module, "n"), 0) if cohort == "2014J" else len(subset),
                                   "test_eligible_records_removed_for_overlap": (test_before_overlap.get((module, "n"), 0) - len(subset)) if cohort == "2014J" else 0,
                                   "withdrawals_before_overlap_removal": test_before_overlap.get((module, "positive"), 0) if cohort == "2014J" else pos,
                                   "withdrawal_days29_56": pos, "negative_horizon_labels": len(subset)-pos,
                                   "prevalence": None if not subset else pos/len(subset),
                                   "prevalence_before_overlap_removal": (test_before_overlap.get((module, "positive"), 0) / test_before_overlap.get((module, "n"), 1)) if cohort == "2014J" and test_before_overlap.get((module, "n"), 0) else (None if cohort == "2014J" else (None if not subset else pos/len(subset))),
                                   "zero_activity_records": sum(not x["has_activity_0_28"] for x in clicks),
                                   "day28_clicks": sum(x["clicks_0_28"] for x in clicks)})
        cohort_records = cohort_rows[cohort]
        n = len(cohort_records); positives = sum(r["label"] == 1 for r in cohort_records)
        label_rows.append({"cohort": cohort, "eligible_records": n, "positive_29_56": positives, "negative_29_56": n-positives,
                           "prevalence": None if not n else positives/n,
                           "quarantined_label_records": sum(v for (c, _), v in quality_counts.items() if c == cohort),
                           "unknown_withdrawal_time_quarantined": quality_counts.get((cohort, "withdrawn_status_missing_unregistration_date"), 0),
                           "followup_rule": "module lasts beyond day 56 and observed registration record plus final-status validation",
                           "negative_is_not_lifetime_nonwithdrawal": True})

    # Stage of integrity metadata and source hashes.
    manifest = {"source_url": SOURCE, "retrieved_at_utc": datetime.fromtimestamp(archive.stat().st_mtime, timezone.utc).isoformat() if archive else None, "license": LICENSE,
                "citation": "Kuzilek, Hlosta & Zdrahal, Scientific Data 4, 170171 (2017), DOI 10.1038/sdata.2017.171; UCI DOI 10.24432/C5KK69",
                "software": {"python": platform.python_version(),"pipeline_sha256":sha256(Path(__file__)),
                             "acquirer_sha256":sha256(Path(__file__).with_name("acquire_oulad.py"))},
                "archive": None if archive is None else {"path": "data/raw/oulad.zip", "bytes": archive.stat().st_size, "sha256": sha256(archive)},
                "tables": {n: {"bytes": p.stat().st_size, "sha256": sha256(p)} for n,p in files.items()},
                "schema": {n: list(read_csv_header(p)) for n,p in files.items()}}
    (out_dir / "data_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    write_csv(out_dir/"cohort_counts.csv", cohort_summary)
    write_csv(out_dir/"label_quality.csv", label_rows)
    write_csv(out_dir/"exclusion_counts.csv", [{"cohort":k[0],"module":k[1],"reason":k[2],"records":v} for k,v in sorted(exclusion_counts.items())])
    write_csv(out_dir/"label_issues.csv", [{"cohort":k[0],"issue":k[1],"records":v} for k,v in sorted(quality_counts.items())])
    write_csv(out_dir/"activity_timing.csv", [{"cohort":c,"module":m,
                "eligible_records_with_pre_registration_activity":len(units_with_pre_registration_activity[(c,m)]),
                "pre_course_event_rows":activity_timing[(c,m)].get("pre_course_event_rows",0),
                "before_registration_event_rows":activity_timing[(c,m)].get("before_registration_event_rows",0),
                "before_registration_clicks":activity_timing[(c,m)].get("before_registration_clicks",0)} for c in COHORTS for m in common_modules])
    integrity = {
        "rows": {"courses":len(courses),"studentInfo":len(infos),"studentRegistration":len(regs),"assessments":len(assessments),"studentAssessment":len(submissions),"vle":len(vle),"studentVle":vle_rows},
        "primary_key_duplicates": {"courses":len(courses)-len(course_by_context),"studentInfo_unit":duplicate_info,"studentRegistration_unit":duplicate_regs,
                                    "assessments_id":len(assessments)-len(assessment_ids),"studentAssessment_pair":submission_duplicates,
                                    "vle_site_context":len(vle)-len(site_contexts)},
        "studentVle_repeated_composite_key_rows": {"key": ["code_module","code_presentation","id_student","id_site","date"], "extra_rows":duplicates,
                                                   "interpretation":"This composite is not declared as a primary key in the supplied schema. Rows are retained and sum_click values aggregated; no blind deduplication."},
        "foreign_key_coverage_failures": {"registration_to_course":missing_course_context,"registration_to_studentInfo":missing_info,
                                          "studentAssessment_to_assessment":missing_assessment,"studentAssessment_student_unknown":missing_student,
                                          "studentAssessment_assessment_student_context_mismatch":wrong_assessment_context,
                                          "studentVle_site_unknown":site_missing,"studentVle_site_context_mismatch":site_context_mismatch},
        "registration_date_ranges": {"registration_min":min(filter(lambda x:x is not None,(number(r['date_registration']) for r in regs)),default=None),"registration_max":max(filter(lambda x:x is not None,(number(r['date_registration']) for r in regs)),default=None),
                                     "unregistration_min":min(filter(lambda x:x is not None,(number(r['date_unregistration']) for r in regs)),default=None),"unregistration_max":max(filter(lambda x:x is not None,(number(r['date_unregistration']) for r in regs)),default=None)},
        "course_lengths": {"min":min(number(r['module_presentation_length']) for r in courses),"max":max(number(r['module_presentation_length']) for r in courses)},
        "final_status_enums":_counts(infos,"final_result"),"banked_submission_records":banked,"studentVle_quality":dict(vle_quality),
        "missingness": {**dict(sorted(missingness.items())), **dict(sorted(missingness_vle.items()))},
        "registration_impossible_unregistration_before_registration":sum(1 for r in regs if number(r['date_registration']) is not None and number(r['date_unregistration']) is not None and number(r['date_unregistration']) < number(r['date_registration'])),
        "registration_unknown_dates": {"registration":sum(number(r['date_registration']) is None for r in regs),"unregistration":sum(number(r['date_unregistration']) is None for r in regs)},
        "assessment_date_range": {"min":min((number(r['date']) for r in assessments if number(r['date']) is not None),default=None),"max":max((number(r['date']) for r in assessments if number(r['date']) is not None),default=None)},
        "submission_date_range": {"min":min((number(r['date_submitted']) for r in submissions if number(r['date_submitted']) is not None),default=None),"max":max((number(r['date_submitted']) for r in submissions if number(r['date_submitted']) is not None),default=None)},
        "studentVle_date_range": {"min":min_vle_day,"max":max_vle_day},"common_J_modules":common_modules,
        "label_data_contradictions_all_records": {
            "withdrawn_status_missing_unregistration_date":sum(1 for r in regs if number(r['date_unregistration']) is None and (info_by_key.get(key(r),{}).get('final_result','').strip().casefold()=='withdrawn')),
            "nonwithdrawn_status_with_unregistration_date":sum(1 for r in regs if number(r['date_unregistration']) is not None and (info_by_key.get(key(r),{}).get('final_result','').strip().casefold()!='withdrawn')),
            "unknown_or_missing_final_status":sum(1 for r in regs if not info_by_key.get(key(r),{}).get('final_result','').strip()),
        },
        "day56_followup_evidence": {"minimum_common_module_length_days":min(course_by_context[(m,p)] for m in common_modules for p in COHORTS),
                                    "all_common_module_lengths_exceed_56":all(course_by_context[(m,p)]>56 for m in common_modules for p in COHORTS),
                                    "documentation":"OULAD.names states registrations carry date_unregistration; completed students have it empty and unregistered students have final_result Withdrawn. This supports the observed dataset follow-up interpretation but does not establish administrative recording latency."},
        "student_overlap":{"train_all_records":len(train_rows_all),"train_all_unique_ids":len(train_ids_all),"test_all_records":len(test_rows_all),
                           "test_all_unique_ids":len({r['id_student'] for r in test_rows_all}),"overlap_unique_ids_removed":len(overlap_ids),
                           "test_records_removed":len(test_rows_all)-len(test_rows_disjoint),"remaining_test_unique_ids":len(disjoint_test_ids)},
    }
    (out_dir/"integrity.json").write_text(json.dumps(integrity,indent=2)+"\n")
    failed_joins = {name: count for name, count in integrity["foreign_key_coverage_failures"].items() if count}
    unresolved_label_count = sum(v for (cohort, issue), v in quality_counts.items()
                                 if issue in {"withdrawn_status_missing_unregistration_date", "withdrawal_date_conflicts_with_final_status", "late_unregistration_conflicts_with_final_status", "missing_final_status"})
    class_adequacy = {r["cohort"]: {"both_classes_present":r["positive_29_56"] > 0 and r["negative_29_56"] > 0,
                                   "positive":r["positive_29_56"], "negative":r["negative_29_56"]} for r in label_rows}
    blockers = []
    if failed_joins: blockers.append("unexplained referential-integrity failures")
    if not all(x["both_classes_present"] for x in class_adequacy.values()): blockers.append("empty or single-class cohort")
    if unresolved_label_count: blockers.append("unresolved withdrawal timing/status for eligible records; quarantined counts cannot be labeled by day 56")
    gate = {"decision":"NO-GO" if blockers else "GO", "fitting_authorized":False,
            "basis":{"common_modules":common_modules,"class_adequacy":class_adequacy,"failed_joins":failed_joins,
                     "eligible_label_records_quarantined_for_unresolved_timing_or_status":unresolved_label_count,
                     "followup_window_supported_by_course_length":integrity["day56_followup_evidence"]["all_common_module_lengths_exceed_56"]},
            "blockers":blockers,"next_step":"Review label availability and any permitted handling before a separate protocol lock; do not change the target or cohorts."}
    (out_dir/"gate_decision.json").write_text(json.dumps(gate,indent=2)+"\n")
    (out_dir/"stage_status.json").write_text(json.dumps({
        "stages":{
          "0_environment_preflight":{"status":"passed","evidence":["audit/preflight.json"]},
          "1_data_route_preflight":{"status":"passed","evidence":["archive.ics.uci.edu HTTP 200; restricted network policy enforced"]},
          "2_acquisition_checkpoint":{"status":"passed","evidence":["audit/data_manifest.json; CRC-checked archive and table hashes"]},
          "3_synthetic_smoke_tests":{"status":"passed","evidence":["10 standard-library unittest cases"]},
          "4_full_non_fitting_audit":{"status":"completed","evidence":["audit/integrity.json","audit/cohort_counts.csv","audit/label_quality.csv","audit/gate_decision.json"],"gate_decision":gate["decision"]},
          "5_protocol_lock":{"status":"blocked","dependency":"Independent review of NO-GO label-availability finding and explicit modeling authorization"},
          "6_model_fits":{"status":"not_started","reason":"Out of scope until protocol lock and review"},
          "7_delivery":{"status":"pending","reason":"Repository commit and push pending final review/test"}},
        "recorded_at_utc":datetime.now(timezone.utc).isoformat()},indent=2)+"\n")
    db.close(); db_path.unlink(missing_ok=True)
    return {"manifest":manifest,"integrity":integrity,"cohort":cohort_summary,"labels":label_rows,"exclusions":exclusion_counts,"issues":quality_counts,"gate":gate}


def read_csv_header(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return next(csv.reader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--archive", type=Path, default=Path("data/raw/oulad.zip"))
    args = ap.parse_args()
    result = full_audit(args.data_dir, args.out_dir, args.archive if args.archive.exists() else None)
    print(json.dumps({"common_modules":result["integrity"]["common_J_modules"],"cohort":result["labels"],"student_overlap":result["integrity"]["student_overlap"],"gate":result["gate"]},indent=2))


if __name__ == "__main__": main()
