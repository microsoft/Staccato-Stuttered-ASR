# Offline paired-transcript evaluation

Compare saved baseline and refined transcripts without CUDA, model downloads,
API calls, or the full inference environment. The evaluator uses the project's
existing JiWER dependency; it does not change model inference.

## Run a synthetic smoke test

From the repository root, with [uv](https://docs.astral.sh/uv/) installed:

```shell
uv run --script src/evaluation/compare_transcripts.py examples/synthetic_transcripts.jsonl --reference-kind intended
```

The script declares its lightweight dependency inline, so this command does not
install the project's GPU dependencies. Alternatively, in an isolated Python
3.12+ environment:

```shell
python -m pip install jiwer==4.0.0
python src/evaluation/compare_transcripts.py examples/synthetic_transcripts.jsonl --reference-kind intended
```

**Every transcript, latency, and cost in the included fixture is invented.**
It checks the reporting workflow; it is not a benchmark, a model execution,
evidence of clinical effectiveness, or a Staccato performance claim. The fixture
deliberately includes a refinement regression and hallucinated speech on silence.
Expected corpus WER is 0.125 for the baseline and 0.375 for refinement.

## Input contract

UTF-8 JSONL, with one paired example per non-blank line:

```json
{"id":"clip-001","reference":"I would like tea","baseline":"I I would like tea","refined":"I would like tea"}
```

Required string fields:

- `id`: unique, non-empty identifier, with no surrounding whitespace.
- `reference`: independently annotated target transcript.
- `baseline`: baseline system output.
- `refined`: refined system output for the **same audio clip**.

Empty transcript strings are valid, including silence references. Missing
transcripts are not silently replaced with empty strings. Duplicate IDs,
duplicate JSON keys, unknown fields, invalid JSON, and wrong types are errors
reported with the input filename and line number.

Optional numeric fields:

- `baseline_latency_ms`, `refined_latency_ms`
- `baseline_cost_usd`, `refined_cost_usd`

Supply both baseline and refined measurements for a metric on any measured row,
or omit both. Values must be finite and non-negative; zero is a measured value,
not a synonym for missing. Explicit `null` values are rejected.

Latency and cost must cover comparable **end-to-end** work. Refined values should
include baseline transcription plus refinement, not only the second stage.
Document model versions, hardware, warm-up, batch size, API pricing date,
dataset version, split, and reference-annotation protocol alongside real runs.
Those facts cannot be established from saved transcripts alone.

## Choose the reference task explicitly

`--reference-kind` is required:

- `intended`: human-annotated intended-message transcripts, excluding involuntary
  disfluencies under a documented annotation protocol.
- `verbatim`: human-annotated speech including the disfluencies being evaluated.

Do not silently remove repetitions from references, use a system's own output as
ground truth, or compare scores across these two tasks as if they were equivalent.
Run separate evaluations when both reference types exist.

Normalization defaults to `whitespace`: Unicode whitespace is collapsed and
leading/trailing whitespace removed. Case, punctuation, and repetitions are
preserved. `--normalization lowercase` additionally lowercases all transcripts.
No language-specific tokenization is applied: word-based scores are not an
appropriate standalone metric for unsegmented languages.

## Report contract

The command prints JSON to stdout. To save a new file, add:

```shell
--baseline-label whisper-large-v3 --refined-label staccato --output report.json
```

Existing output files are never overwritten, including the input file. Failures
exit with status 2 and a diagnostic on stderr, not a success-shaped empty report.

Schema version 1 includes:

- SHA-256 of exact input bytes, JiWER version, normalization, reference task,
  system labels, and example count.
- Corpus word error rate (WER) and substitution, deletion, insertion, and hit
  counts for each system. **WER is total errors / total reference words**, not
  an average of per-clip WER. It can exceed 1.
- Per-example counts and changes, plus improved/unchanged/regressed totals.
  Changes are **refined minus baseline**: negative means fewer errors.
- Optional measurement count, sum, and mean per system. Count communicates
  coverage; totals cover only measured pairs, not the whole corpus unless
  coverage is complete. Missing measurements produce `null`, not zero.

Silence-reference insertions count toward corpus errors. WER is `null` when the
relevant reference word count is zero; raw error counts remain available. This
avoids presenting JiWER's empty-reference insertion count as an error *rate*.

The report omits transcript text, but **IDs and file hashes can still be
sensitive**. Review them before sharing. WER insertions are not a complete
hallucination or safety metric, and lower WER alone does not establish that
meaning, speaker intent, or accessibility has been preserved.

## Validate changes

In the lightweight environment above:

```shell
python -m unittest discover -s tests -v
```

For Ruff, use the repository's existing lint/format conventions, scoped to the
changed Python files. To reproduce the additional CI checks:

```shell
python -m pip install ruff==0.17.3 pyright==1.1.414
python -m ruff check src/evaluation/compare_transcripts.py tests/test_compare_transcripts.py
python -m ruff format --check src/evaluation/compare_transcripts.py tests/test_compare_transcripts.py
python -m pyright src/evaluation/compare_transcripts.py tests/test_compare_transcripts.py
```

CI runs these checks, the tests, and the synthetic CLI on Linux and Windows
without loading any speech models.

Real evaluation requires appropriately licensed data and independently checked
references. Do not commit restricted speech recordings, personal transcripts,
API credentials, or data whose redistribution terms are unclear.
