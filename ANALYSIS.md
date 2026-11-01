# Reading and reproducing the analysis

## Preserved specification

`analysis/protocol/candidate_protocol.json` has `document_status: LOCKED`. Its four protocol components and eight implementation components reproduce the archived composite fingerprints reported in the supplement. File names and historical draft wording are preserved deliberately; they are not current instructions to revise the study after seeing results.

The source fit/calibration split is the first fold from `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`, grouped by student. It contains 6,141 fitting records with 174 events and 1,535 calibration records with 44 events. The registration model and its activity extension share the source split, L2 logistic specification and preprocessing rules. Each separate unpenalized source sigmoid uses its base model's `decision_function`, not a target-fitted probability recalibration.

The three unresolved source records remain excluded. The three-state target construction retains all 58 uncertain records. The preserved protocol's phrase “independent Cartesian-product assumption” means every combination of feasible record states is admissible, without an external aggregate constraint; it does not establish probabilistic independence. The primary endpoints condition on the fitted models and accepted labels. Source-selection, refitting, sampling and future-population uncertainty are not estimated. Sparse reliability bins and module results remain descriptive.

## Runtime and non-fitting inspection

The recorded analysis runtime was Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, scikit-learn 1.8.0, joblib 1.5.3 and threadpoolctl 3.6.0. Pins are preserved in `analysis/protocol/requirements-modeling.txt`. Its historical “draft” comment is unchanged. Installation and execution require a compatible environment; packaging did not independently reinstall all dependencies or repeat model fitting.

From `analysis/`, after installing the listed dependencies:

```sh
python -m protocol.locked_runner
```

This is a dry-run report and does not train or predict. The data are not needed for this dry run. Avoid invoking individual internal fit functions outside the runner.

The audit and feature programs are also available for a researcher who separately obtains the verified data. Outputs should remain local and Git-ignored:

```sh
python audit_pipeline.py --data-dir data/raw/extracted --out-dir data/derived/nonfitting-audit --archive data/raw/oulad.zip
python audit_label_availability.py --data-dir data/raw/extracted --out-dir data/derived/nonfitting-audit
python -m protocol.day28_features --data-dir data/raw/extracted --output data/derived/feature_preflight.json
```

The first audit predates the partial-identification solution and will report unresolved labels as its historical gate. Do not interpret that preliminary gate as a failed reconstruction of the later bounds. The locked feature implementation is `protocol/day28_features.py`; the earlier audit utility's four descriptive click buckets are not the five fixed model inputs.

## Portable local reproduction adapter

The original runner's private parent/reviewer workflow is a historical execution control, not a scientific obligation imposed on outside researchers. Its source and hashes remain unchanged. The separate `reproduce.py` adapter uses the same frozen configuration, preprocessing, source split, sigmoid specification, four fitting stages, bounds and runtime limits. It adds a public entry point that requires explicit local execution opt-in, without asking researchers to invent private approval references or claim independent review.

From the repository root, the default is a static dry run; it does not import the analysis, read raw data, fit or predict:

```sh
python reproduce.py
python tools/test_reproduction_adapter.py
```

The adapter tests use a mock runner only. After separately obtaining the verified data in `analysis/data/raw/`, installing the exact analysis runtime and reading the scientific assumptions, a researcher can request a **new local reproduction**:

```sh
python reproduce.py --execute-local-reproduction --run-id MY_NEW_LOCAL_RUN
```

This command **does fit models**. It was not executed while preparing this repository. It validates the original protocol and code fingerprints before importing the runner; the unchanged runner then enforces exact data/split/runtime fingerprints, output-path protection, resource limits and no-overwrite guards. The adapter translates the explicit flag into legacy receipt fields labelled `LOCAL_REPRODUCTION_OPT_IN`; its reviewer field expressly says that no external reviewer is claimed. This local consent is not historical authorization or independent scientific approval.

Private outputs remain under ignored `analysis/data/derived/`. Do not publish individual predictions, fitted models, feature matrices or logs. The new run has a new identifier and does not overwrite the recorded results. Byte-for-byte numerical reproduction after new fitting has **not** been verified for this adapter; it is an inspected, mock-tested entry point, not an executed validation of a new run. Environment/platform differences can affect numerical outputs and must be reported rather than hidden by changing the specification.

## Test and verification boundaries

The active-pipeline synthetic/mock test sources are supplied without scientific changes. The draft-only source-split checker and its corresponding test are omitted from this minimal public package; their historical status is documented below. The study recorded 49 passing pre-lock tests and 48 selected post-lock tests, not a complete post-lock suite rerun. The historical `test_source_split` invoked a preflight that rejects locked configurations; that known status is reported rather than rewriting the locked protocol or representing the omitted test as passed. The realised source split was separately reconciled after locking.

Previous verification reconciled aggregate metrics and exact rational bound endpoints. A later check regenerated 27,316 probabilities from saved models and reconstructed inputs; it was not independent model refitting or independent reconstruction of the original outcome/eligibility labels. Those private models, inputs and probabilities are not redistributed here. Packaging verification is limited to static source/manifest/privacy checks and, where stated in the release receipt, aggregate-only figure regeneration. Full historical end-to-end reproduction is not newly claimed.
