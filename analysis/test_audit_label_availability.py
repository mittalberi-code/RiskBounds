import csv
import tempfile
import unittest
from pathlib import Path

from audit_label_availability import joint_record, make_tables


class JointLabelAvailabilityTest(unittest.TestCase):
    def test_missing_registration_does_not_hide_multiple_quality_dimensions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (root / "courses.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["code_module", "code_presentation", "module_presentation_length"])
                writer.writeheader(); writer.writerow({"code_module":"AAA", "code_presentation":"2014J", "module_presentation_length":"269"})
            with (root / "studentInfo.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["id_student", "code_module", "code_presentation", "final_result"])
                writer.writeheader(); writer.writerow({"id_student":"x", "code_module":"AAA", "code_presentation":"2014J", "final_result":"Withdrawn"})
            with (root / "studentRegistration.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["id_student", "code_module", "code_presentation", "date_registration", "date_unregistration"])
                writer.writeheader(); writer.writerow({"id_student":"x", "code_module":"AAA", "code_presentation":"2014J", "date_registration":"?", "date_unregistration":"?"})
            joint, summary = make_tables(root)
        self.assertEqual(len(joint), 1)
        row = joint[0]
        self.assertEqual(row["records"], 1)
        self.assertEqual(row["registration_date_condition"], "missing")
        self.assertEqual(row["withdrawal_time_band"], "missing")
        self.assertEqual(row["final_status_condition"], "Withdrawn")
        self.assertEqual(row["date_status_condition"], "missing_date_withdrawn_status")
        self.assertEqual(row["day28_eligibility"], "unknown_registration_date")
        self.assertEqual(row["horizon_outcome_given_eligibility"], "unknown_withdrawal_time_and_risk_set_membership")
        self.assertIn({"cohort":"2014J", "module":"AAA", "measure":"registration_missing_records", "records":1}, summary)

    def test_event_band_is_distinguished_from_unknown_risk_set(self):
        row = joint_record("?", "29", "Withdrawn", 269)
        self.assertEqual(row["withdrawal_time_band"], "day_29_56")
        self.assertEqual(row["day28_eligibility"], "unknown_registration_date")
        self.assertEqual(row["horizon_outcome_given_eligibility"], "event_day_29_56_known_but_risk_set_unknown")

    def test_known_prelandmark_unregistration_is_ineligible_even_if_registration_is_missing(self):
        row = joint_record("?", "12", "Withdrawn", 269)
        self.assertEqual(row["day28_eligibility"], "known_ineligible_withdrawn_by_day_28")
        self.assertEqual(row["horizon_outcome_given_eligibility"], "not_at_risk_at_day_28")

    def test_known_late_date_conflict_is_not_unknown_event_time(self):
        row = joint_record("0", "57", "Pass", 269)
        self.assertEqual(row["withdrawal_time_band"], "after_day_56")
        self.assertEqual(row["date_status_condition"], "known_date_nonwithdrawn_status_contradiction")
        self.assertEqual(row["day28_eligibility"], "known_eligible_at_day_28")
        self.assertEqual(row["horizon_outcome_given_eligibility"], "unknown_date_status_contradiction")

    def test_missing_withdrawal_time_with_withdrawn_status_is_unknown(self):
        row = joint_record("0", "?", "Withdrawn", 269)
        self.assertEqual(row["day28_eligibility"], "unknown_withdrawn_date_missing")
        self.assertEqual(row["horizon_outcome_given_eligibility"], "unknown_withdrawal_time_and_risk_set_membership")

    def test_documented_completed_record_is_feasible_negative(self):
        row = joint_record("0", "?", "Pass", 269)
        self.assertEqual(row["day28_eligibility"], "known_eligible_at_day_28")
        self.assertEqual(row["horizon_outcome_given_eligibility"], "negative_no_withdrawal_date_nonwithdrawn_status")


if __name__ == "__main__":
    unittest.main()
