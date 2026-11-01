"""Adapter-only tests with a mock runner: no analysis import, data read or fit."""
import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

sys.dont_write_bytecode = True
SPEC = importlib.util.spec_from_file_location('public_adapter', Path(__file__).resolve().parents[1]/'reproduce.py')
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)

class AdapterTest(unittest.TestCase):
    def test_default_never_imports_runner(self):
        with patch.object(adapter, '_load_runner', side_effect=AssertionError('must not import')), contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(adapter.main([]), 0)
        self.assertIn('STATIC_DRY_RUN', out.getvalue())

    def test_changed_archived_hash_fails_before_import(self):
        with patch.object(adapter, 'fingerprint', return_value='mismatch'), patch.object(adapter, '_load_runner') as load:
            with self.assertRaises(SystemExit): adapter.main([])
            load.assert_not_called()

    def test_execution_requires_valid_new_run_id(self):
        for args in [[], ['--run-id','../escape']]:
            with patch.object(adapter, '_load_runner') as load, contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit): adapter.main(['--execute-local-reproduction',*args])
                load.assert_not_called()

    def test_explicit_flag_translates_only_local_opt_in(self):
        runner = Mock()
        runner.run_authorized_from_data.return_value = {'primary_paired_brier_improvement_bounds':{'mock_only':True}}
        with patch.object(adapter, '_load_runner', return_value=runner), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(adapter.main(['--execute-local-reproduction','--run-id','synthetic-adapter-test']), 0)
        args = runner.run_authorized_from_data.call_args.kwargs
        self.assertEqual(args['parent_authorization_reference'], 'LOCAL_REPRODUCTION_OPT_IN:synthetic-adapter-test')
        self.assertIn('No external reviewer claimed', args['manifest']['reviewer'])
        for key,value in adapter.EXPECTED.items(): self.assertEqual(args['manifest'][key],value)
        self.assertTrue(args['explicit_cli_authorization'])

if __name__ == '__main__': unittest.main()
