#!/usr/bin/env python3
"""Static package verification: never imports analysis or executes model code."""
import ast
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_FILES = ('protocol/candidate_protocol.json', 'protocol/draft_fitting_protocol.md',
                  'protocol/partial_brier_bounds.md', 'protocol/requirements-modeling.txt')
CODE_FILES = ('protocol/locked_runner.py', 'protocol/day28_features.py', 'protocol/target_bounds.py',
              'protocol/brier_bounds.py', 'protocol/execution_contract.py',
              'audit_pipeline.py', 'audit_label_availability.py', '.gitignore')

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def composite(names):
    return hashlib.sha256(''.join(f'{name}\t{digest(ROOT / "analysis" / name)}\n' for name in names).encode()).hexdigest()

def main():
    manifest = json.loads((ROOT / 'MANIFEST.json').read_text())
    listed = set()
    for row in manifest['files']:
        name = row['path']
        rel = PurePosixPath(name)
        if not name or rel.is_absolute() or '..' in rel.parts or '\\' in name or name in listed:
            raise SystemExit(f'Unsafe or duplicate manifest path: {name!r}')
        listed.add(name)
    observed = set()
    for path in ROOT.rglob('*'):
        rel = path.relative_to(ROOT)
        if rel.parts[0] == '.git':
            continue  # Git metadata is never a published package member.
        if path.is_symlink():
            raise SystemExit(f'Symlink is not allowed in publication staging: {rel}')
        if path.is_file():
            observed.add(rel.as_posix())
    unexpected = observed - listed - {'MANIFEST.json'}
    if unexpected:
        raise SystemExit(f'Unlisted files in publication staging: {sorted(unexpected)}')
    for row in manifest['files']:
        path = ROOT / row['path']
        if not path.is_file() or path.is_symlink():
            raise SystemExit(f'Missing/non-regular allowlisted file: {row["path"]}')
        if path.stat().st_size != row['bytes'] or digest(path) != row['sha256']:
            raise SystemExit(f'Integrity mismatch: {row["path"]}')
        if path.suffix == '.py':
            ast.parse(path.read_text(), filename=row['path'])
    assert composite(PROTOCOL_FILES) == '9131bb16347914642d97e5e5e49856d9411b5c4f00ddb7ce427af32f16549a07'
    assert composite(CODE_FILES) == '4972c30eb8783595c83ee96285ce341abe0f6527d3dd71b55ddb5411fc865ba6'
    schemas = {}
    for path in sorted((ROOT / 'figures/inputs').glob('*.csv')):
        with path.open(newline='', encoding='utf-8') as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames
            rows = list(reader)
        assert not ({'id_student', 'student_id', 'known_label', 'threshold', 'raw_probability', 'calibrated_probability'} & set(fields)), path.name
        schemas[path.name] = {'columns': fields, 'rows': len(rows)}
    curve = schemas['roc_pr_curve_points.csv']['columns']
    assert curve == ['curve','predictor','probability','x_fpr','y_tpr','x_recall','y_precision']
    print(json.dumps({'status':'PASS', 'allowlisted_files_verified':len(manifest['files']),
        'protocol_fingerprint_matches':True, 'code_fingerprint_matches':True,
        'python_syntax_checked':True, 'analysis_imported_or_executed':False,
        'raw_data_downloaded':False, 'models_fit':False, 'aggregate_schemas':schemas}, indent=2))

if __name__ == '__main__':
    main()
