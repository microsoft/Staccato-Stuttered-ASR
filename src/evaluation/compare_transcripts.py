# /// script
# requires-python = ">=3.12"
# dependencies = ["jiwer==4.0.0"]
# ///
"""Compare paired ASR transcripts without loading models or sending audio."""

import argparse
import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Literal

import jiwer

type ReferenceKind = Literal["intended", "verbatim"]
type Normalization = Literal["whitespace", "lowercase"]

TEXT_FIELDS = ("id", "reference", "baseline", "refined")
MEASUREMENT_FIELDS = (
    "baseline_latency_ms",
    "refined_latency_ms",
    "baseline_cost_usd",
    "refined_cost_usd",
)


@dataclass(frozen=True)
class TranscriptPair:
    id: str
    reference: str
    baseline: str
    refined: str
    baseline_latency_ms: float | None = None
    refined_latency_ms: float | None = None
    baseline_cost_usd: float | None = None
    refined_cost_usd: float | None = None


@dataclass(frozen=True)
class ErrorCounts:
    reference_words: int
    hits: int
    substitutions: int
    deletions: int
    insertions: int
    errors: int
    wer: float | None


@dataclass(frozen=True)
class Measurement:
    count: int
    total: float | None
    mean: float | None


@dataclass(frozen=True)
class SystemResult:
    label: str
    quality: ErrorCounts
    latency_ms: Measurement
    cost_usd: Measurement


@dataclass(frozen=True)
class ExampleResult:
    id: str
    baseline: ErrorCounts
    refined: ErrorCounts
    error_change: int


@dataclass(frozen=True)
class Comparison:
    wer_change: float | None
    error_change: int
    improved: int
    unchanged: int
    regressed: int


@dataclass(frozen=True)
class EvaluationReport:
    schema_version: int
    input_sha256: str
    jiwer_version: str
    reference_kind: ReferenceKind
    normalization: Normalization
    example_count: int
    baseline: SystemResult
    refined: SystemResult
    comparison: Comparison
    examples: list[ExampleResult]


def _unique_object(items: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _measurement(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a finite, non-negative number")
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError(f"{field} is outside the supported numeric range") from error
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{field} must be a finite, non-negative number")
    return number


def load_pairs(path: Path) -> tuple[list[TranscriptPair], str]:
    """Load strict JSONL and fingerprint the exact input bytes."""
    content = path.read_bytes()
    pairs: list[TranscriptPair] = []
    seen: set[str] = set()
    allowed = set(TEXT_FIELDS + MEASUREMENT_FIELDS)
    for line_number, line in enumerate(content.decode("utf-8-sig").split("\n"), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line, object_pairs_hook=_unique_object)
            if not isinstance(row, dict):
                raise TypeError("each line must be a JSON object")
            unknown = set(row) - allowed
            if unknown:
                raise ValueError(f"unknown fields: {', '.join(sorted(unknown))}")
            for field in TEXT_FIELDS:
                if field not in row or not isinstance(row[field], str):
                    raise ValueError(f"{field} is required and must be a string")
            identifier: str = row["id"]
            if not identifier.strip() or identifier != identifier.strip():
                raise ValueError("id must be non-empty with no surrounding whitespace")
            if identifier in seen:
                raise ValueError(f"duplicate id {identifier!r}")
            measurements = {
                field: _measurement(row[field], field)
                for field in MEASUREMENT_FIELDS
                if field in row
            }
            for suffix in ("latency_ms", "cost_usd"):
                if (f"baseline_{suffix}" in row) != (f"refined_{suffix}" in row):
                    raise ValueError(
                        f"provide both baseline_{suffix} and refined_{suffix}"
                    )
            pair = TranscriptPair(
                id=identifier,
                reference=row["reference"],
                baseline=row["baseline"],
                refined=row["refined"],
                **measurements,
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"{path.name}:{line_number}: {error}") from error
        seen.add(identifier)
        pairs.append(pair)
    if not pairs:
        raise ValueError(f"{path.name}: no transcript pairs found")
    return pairs, hashlib.sha256(content).hexdigest()


def _normalize(text: str, normalization: Normalization) -> str:
    normalized = " ".join(text.split())
    return normalized.lower() if normalization == "lowercase" else normalized


def _score(references: list[str], hypotheses: list[str]) -> ErrorCounts:
    output = jiwer.process_words(references, hypotheses)
    reference_words = output.hits + output.substitutions + output.deletions
    errors = output.substitutions + output.deletions + output.insertions
    return ErrorCounts(
        reference_words=reference_words,
        hits=output.hits,
        substitutions=output.substitutions,
        deletions=output.deletions,
        insertions=output.insertions,
        errors=errors,
        wer=errors / reference_words if reference_words else None,
    )


def _summarize(values: Sequence[float | None]) -> Measurement:
    present = [value for value in values if value is not None]
    if not present:
        return Measurement(count=0, total=None, mean=None)
    total = math.fsum(present)
    return Measurement(count=len(present), total=total, mean=total / len(present))


def compare(
    pairs: list[TranscriptPair],
    input_sha256: str,
    *,
    reference_kind: ReferenceKind,
    normalization: Normalization = "whitespace",
    baseline_label: str = "baseline",
    refined_label: str = "refined",
) -> EvaluationReport:
    if not pairs:
        raise ValueError("at least one transcript pair is required")
    references = [_normalize(pair.reference, normalization) for pair in pairs]
    baselines = [_normalize(pair.baseline, normalization) for pair in pairs]
    refinements = [_normalize(pair.refined, normalization) for pair in pairs]
    baseline_quality = _score(references, baselines)
    refined_quality = _score(references, refinements)
    examples: list[ExampleResult] = []
    for pair, reference, baseline, refined in zip(
        pairs, references, baselines, refinements, strict=True
    ):
        baseline_score = _score([reference], [baseline])
        refined_score = _score([reference], [refined])
        examples.append(
            ExampleResult(
                id=pair.id,
                baseline=baseline_score,
                refined=refined_score,
                error_change=refined_score.errors - baseline_score.errors,
            )
        )
    wer_change = (
        refined_quality.wer - baseline_quality.wer
        if baseline_quality.wer is not None and refined_quality.wer is not None
        else None
    )
    return EvaluationReport(
        schema_version=1,
        input_sha256=input_sha256,
        jiwer_version=version("jiwer"),
        reference_kind=reference_kind,
        normalization=normalization,
        example_count=len(pairs),
        baseline=SystemResult(
            label=baseline_label,
            quality=baseline_quality,
            latency_ms=_summarize([pair.baseline_latency_ms for pair in pairs]),
            cost_usd=_summarize([pair.baseline_cost_usd for pair in pairs]),
        ),
        refined=SystemResult(
            label=refined_label,
            quality=refined_quality,
            latency_ms=_summarize([pair.refined_latency_ms for pair in pairs]),
            cost_usd=_summarize([pair.refined_cost_usd for pair in pairs]),
        ),
        comparison=Comparison(
            wer_change=wer_change,
            error_change=refined_quality.errors - baseline_quality.errors,
            improved=sum(example.error_change < 0 for example in examples),
            unchanged=sum(example.error_change == 0 for example in examples),
            regressed=sum(example.error_change > 0 for example in examples),
        ),
        examples=examples,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Paired transcript JSONL")
    parser.add_argument(
        "--reference-kind", required=True, choices=("intended", "verbatim")
    )
    parser.add_argument(
        "--normalization", choices=("whitespace", "lowercase"), default="whitespace"
    )
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--refined-label", default="refined")
    parser.add_argument(
        "--output", type=Path, help="New report file; never overwritten"
    )
    args = parser.parse_args(argv)
    try:
        pairs, fingerprint = load_pairs(args.input)
        report = compare(
            pairs,
            fingerprint,
            reference_kind=args.reference_kind,
            normalization=args.normalization,
            baseline_label=args.baseline_label,
            refined_label=args.refined_label,
        )
        serialized = json.dumps(asdict(report), indent=2, allow_nan=False) + "\n"
        if args.output is None:
            print(serialized, end="")
        else:
            with args.output.open("x", encoding="utf-8", newline="\n") as output:
                output.write(serialized)
    except (OSError, ValueError, OverflowError) as error:
        parser.exit(2, f"{parser.prog}: error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
