import unittest
import numpy as np

from protocol.execution_contract import validate_fit_record


class ExecutionContractTest(unittest.TestCase):
    def test_valid_record_has_no_retry(self):
        result = validate_fit_record(elapsed_seconds=12, overall_elapsed_seconds=120,
                                     timed_out=False, convergence_warning=False,
                                     coefficients_finite=True, probabilities_valid=True,
                                     log_path="fit.log")
        self.assertEqual(result["status"], "candidate_fit_record_valid")
        self.assertFalse(result["retry"])

    def test_failure_modes_fail_closed_and_keep_log_reference(self):
        common = dict(elapsed_seconds=10, overall_elapsed_seconds=20, timed_out=False,
                      convergence_warning=False, coefficients_finite=True,
                      probabilities_valid=True, log_path="preserved-fit.log")
        for override, message in (({"timed_out": True}, "timeout"),
                                  ({"convergence_warning": True}, "convergence_warning"),
                                  ({"coefficients_finite": False}, "non_finite_coefficients"),
                                  ({"probabilities_valid": False}, "invalid_probability"),
                                  ({"elapsed_seconds": 301}, "per_fit_timeout_exceeded"),
                                  ({"overall_elapsed_seconds": 1201}, "overall_timeout_exceeded")):
            with self.subTest(message=message):
                with self.assertRaisesRegex(RuntimeError, message) as err:
                    validate_fit_record(**(common | override))
                self.assertIn("preserved-fit.log", str(err.exception))

    def test_invalid_elapsed_values_fail_closed(self):
        common = dict(elapsed_seconds=1, overall_elapsed_seconds=2, timed_out=False,
                      convergence_warning=False, coefficients_finite=True,
                      probabilities_valid=True, log_path="preserved-fit.log")
        for field, bad in (("elapsed_seconds", float("nan")), ("elapsed_seconds", float("inf")),
                           ("elapsed_seconds", -0.01), ("overall_elapsed_seconds", np.float64("nan")),
                           ("overall_elapsed_seconds", np.float32(-1))):
            with self.subTest(field=field, bad=bad):
                args = common | {field: bad}
                with self.assertRaisesRegex(RuntimeError, f"invalid_{field}"):
                    validate_fit_record(**args)


if __name__ == "__main__":
    unittest.main()
