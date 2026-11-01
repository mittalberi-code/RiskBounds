import unittest
import tempfile
import zipfile
from pathlib import Path

from acquire_oulad import EXPECTED, extract, validate_members
from audit_pipeline import assess_record, assessment_count, build_assessment_context, day28_features, overlap_filter


class LandmarkRulesTest(unittest.TestCase):
    def test_archive_member_validation(self):
        validate_members(sorted(EXPECTED))
        with self.assertRaises(ValueError):
            validate_members(sorted(EXPECTED | {"../../escape.csv"}))

    def test_verified_archive_extraction_is_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "fixture.zip"
            with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
                for name in sorted(EXPECTED):
                    zf.writestr(name, "fixture\n")
            destination = root / "extracted"
            extract(archive, destination)
            self.assertEqual({p.name for p in destination.iterdir()}, EXPECTED)

    def test_landmark_and_label_boundaries(self):
        self.assertEqual(assess_record(0, 28, "Withdrawn", 100)[0], "excluded_withdrawn_by_landmark")
        for day, expected in [(29, 1), (56, 1), (57, 0)]:
            status = "Withdrawn"
            self.assertEqual(assess_record(0, day, status, 100), ("eligible", expected, None))
        self.assertEqual(assess_record(0, 28, "Withdrawn", 100)[0], "excluded_withdrawn_by_landmark")

    def test_unknown_registration_and_impossible_dates(self):
        self.assertEqual(assess_record(None, None, "Pass", 100)[0], "excluded_unknown_registration")
        self.assertEqual(assess_record(10, 9, "Withdrawn", 100)[2], "unregistration_before_registration")

    def test_label_contradictions_are_quarantined(self):
        self.assertEqual(assess_record(0, None, "Withdrawn", 100)[0], "quarantined_label")
        self.assertEqual(assess_record(0, 29, "Pass", 100)[0], "quarantined_label")

    def test_short_course_cannot_establish_followup(self):
        self.assertEqual(assess_record(0, None, "Pass", 56)[0], "excluded_short_or_unknown_presentation")

    def test_late_vle_events_do_not_change_day28_features_and_zero_retained(self):
        early = day28_features([(0, 3), (28, 2)])
        with_late = day28_features([(0, 3), (28, 2), (29, 1000), (57, 9)])
        self.assertEqual(early, with_late)
        zero = day28_features([])
        self.assertEqual(zero["clicks_0_28"], 0)
        self.assertFalse(zero["has_activity_0_28"])
        self.assertEqual(zero["recency_days_at_28"], 0)

    def test_banked_and_late_assessment_submissions_excluded(self):
        ctx = {"a": ("AAA", "2014J"), "b": ("BBB", "2014J")}
        rows = [
            {"id_assessment": "a", "is_banked": "0", "date_submitted": "28"},
            {"id_assessment": "a", "is_banked": "0", "date_submitted": "29"},
            {"id_assessment": "b", "is_banked": "1", "date_submitted": "10"},
        ]
        self.assertEqual(assessment_count(rows, ctx, {("AAA", "2014J")}), 1)

    def test_duplicate_assessment_join_key_fails_closed(self):
        rows = [
            {"id_assessment": "a", "code_module": "AAA", "code_presentation": "2014J"},
            {"id_assessment": "a", "code_module": "AAA", "code_presentation": "2014J"},
        ]
        with self.assertRaisesRegex(ValueError, "duplicate id_assessment"):
            build_assessment_context(rows)

    def test_overlap_removal_by_student_before_record_filtering(self):
        train = {"1", "3"}
        rows = [{"id_student": "1", "code_module": "AAA"}, {"id_student": "2", "code_module": "BBB"}, {"id_student": "3", "code_module": "CCC"}]
        retained, overlap = overlap_filter(train, rows)
        self.assertEqual({r["id_student"] for r in retained}, {"2"})
        self.assertEqual(overlap, {"1", "3"})
        self.assertTrue({r["id_student"] for r in retained}.isdisjoint(train))


if __name__ == "__main__":
    unittest.main()
