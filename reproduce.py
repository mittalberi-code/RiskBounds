#!/usr/bin/env python3
"""Portable opt-in adapter; the archived scientific implementation is unchanged.

Default: static dry run, with no analysis imports, data access or fitting.
Opt-in: a NEW local reproduction, never the original historical execution.
The legacy runner's 'APPROVED' field means local execution consent here, not
external scientific review. Its reviewer field explicitly records that fact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent
ANALYSIS = ROOT / 'analysis'
EXPECTED = {
    'protocol_sha256': '9131bb16347914642d97e5e5e49856d9411b5c4f00ddb7ce427af32f16549a07',
    'data_sha256': 'ced1ca15ed3491366758fd40a33b73bd1dd189b68e2423ced13aba2c5d7f71d7',
    'split_sha256': '74609d952ccddaf94593f3dad29d6369878c4e0f1904d42b3533d59cfc5ecc4a',
    'code_sha256': '4972c30eb8783595c83ee96285ce341abe0f6527d3dd71b55ddb5411fc865ba6',
}
PROTOCOL_FILES = ('protocol/candidate_protocol.json', 'protocol/draft_fitting_protocol.md',
                  'protocol/partial_brier_bounds.md', 'protocol/requirements-modeling.txt')
CODE_FILES = ('protocol/locked_runner.py', 'protocol/day28_features.py', 'protocol/target_bounds.py',
              'protocol/brier_bounds.py', 'protocol/execution_contract.py',
              'audit_pipeline.py', 'audit_label_availability.py', '.gitignore')

def fingerprint(names):
    entries = ''.join(f'{name}\t{hashlib.sha256((ANALYSIS/name).read_bytes()).hexdigest()}\n' for name in names)
    return hashlib.sha256(entries.encode()).hexdigest()

def _load_runner():
    # Loaded only after the explicit execution flag and static fingerprint gate.
    sys.path.insert(0, str(ANALYSIS))
    from protocol import locked_runner
    return locked_runner

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-local-reproduction', action='store_true',
                        help='Explicitly request a NEW local run with the fixed scientific specification.')
    parser.add_argument('--run-id', help='Unique local output identifier; existing runs are never overwritten.')
    parser.add_argument('--data-dir', type=Path, default=ANALYSIS/'data/raw/extracted')
    args = parser.parse_args(argv)
    if fingerprint(PROTOCOL_FILES) != EXPECTED['protocol_sha256'] or fingerprint(CODE_FILES) != EXPECTED['code_sha256']:
        raise SystemExit('Archived scientific source/specification fingerprint mismatch; refusing to continue.')
    config = json.loads((ANALYSIS/'protocol/candidate_protocol.json').read_text())
    if not args.execute_local_reproduction:
        print(json.dumps({'status':'STATIC_DRY_RUN', 'analysis_imported':False, 'data_accessed':False,
                          'models_fit':False, 'scientific_fingerprints_match':True,
                          'required_runtime':config['runtime'],
                          'next_step':'Read ANALYSIS.md; explicit --execute-local-reproduction and a new --run-id are required.'}, indent=2))
        return 0
    if not args.run_id or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,79}', args.run_id):
        parser.error('--run-id must be a new identifier of 1–80 letters, digits, dot, underscore or hyphen, starting with a letter or digit.')
    reference = 'LOCAL_REPRODUCTION_OPT_IN:' + args.run_id
    # Translate explicit local execution consent into legacy gate fields.
    # This is not a claim of historical authorization or independent review.
    local_manifest = {
        **EXPECTED,
        'review_status':'APPROVED',
        'reviewer':'No external reviewer claimed; explicit local reproduction opt-in.',
        'parent_authorization_reference':reference,
        'approval_scope':'Local execution consent only; not historical or independent scientific approval.',
    }
    runner = _load_runner()
    print('Starting a NEW local reproduction. No external scientific approval is asserted. Private outputs remain local.')
    result = runner.run_authorized_from_data(
        data_dir=args.data_dir.resolve(), config=config, manifest=local_manifest,
        run_id=args.run_id, parent_authorization_reference=reference,
        explicit_cli_authorization=True,
        data_manifest_path=ANALYSIS/'audit/data_manifest.json')
    print(json.dumps({'status':'LOCAL_REPRODUCTION_COMPLETE',
                      'run_id':args.run_id,
                      'output':str(ANALYSIS/'data/derived'/args.run_id),
                      'primary_bounds':result['primary_paired_brier_improvement_bounds'],
                      'independent_scientific_review_claimed':False}, indent=2))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
