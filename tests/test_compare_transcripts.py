import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from src.evaluation.compare_transcripts import (
    EvaluationReport,
    compare,
    load_pairs,
    main,
)

ROOT = Path(__file__).resolve().parents[1]


class TranscriptEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "pairs.jsonl"

    def write_rows(self, rows: list[dict[str, object]]) -> None:
        self.path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

    def row(self, **updates: object) -> dict[str, object]:
        return {
            "id": "sample",
            "reference": "one two",
            "baseline": "one two",
            "refined": "one two",
            **updates,
        }

    def evaluate(self, rows: list[dict[str, object]]) -> EvaluationReport:
        self.write_rows(rows)
        pairs, fingerprint = load_pairs(self.path)
        return compare(pairs, fingerprint, reference_kind="intended")

    def test_perfect_transcripts(self) -> None:
        report = self.evaluate([self.row()])
        self.assertEqual(report.baseline.quality.wer, 0)
        self.assertEqual(report.refined.quality.hits, 2)
        self.assertEqual(report.comparison.unchanged, 1)

    def test_insertions_deletions_and_substitutions(self) -> None:
        report = self.evaluate(
            [
                self.row(id="insertion", baseline="one two three"),
                self.row(id="deletion", baseline="one"),
                self.row(id="substitution", baseline="one three"),
            ]
        )
        quality = report.baseline.quality
        self.assertEqual(
            (quality.insertions, quality.deletions, quality.substitutions), (1, 1, 1)
        )
        self.assertEqual(quality.reference_words, 6)
        self.assertEqual(quality.wer, 0.5)
        self.assertEqual(report.comparison.improved, 3)
        self.assertEqual(report.comparison.wer_change, -0.5)

    def test_corpus_wer_is_not_average_utterance_wer(self) -> None:
        report = self.evaluate(
            [
                self.row(id="short", reference="one", baseline="", refined=""),
                self.row(
                    id="long",
                    reference="one two three four five six seven eight nine",
                    baseline="one two three four five six seven eight nine",
                    refined="one two three four five six seven eight nine",
                ),
            ]
        )
        self.assertEqual(report.baseline.quality.wer, 0.1)

    def test_regression_is_not_hidden_by_aggregate_improvement(self) -> None:
        report = self.evaluate(
            [
                self.row(id="better", baseline="", refined="one two"),
                self.row(id="worse", refined="one"),
            ]
        )
        self.assertEqual(report.comparison.improved, 1)
        self.assertEqual(report.comparison.regressed, 1)
        self.assertEqual(report.examples[1].error_change, 1)

    def test_silence_insertions_count_toward_corpus_errors(self) -> None:
        report = self.evaluate(
            [
                self.row(),
                self.row(
                    id="silence", reference="", baseline="", refined="extra words"
                ),
            ]
        )
        self.assertEqual(report.refined.quality.wer, 1)
        self.assertIsNone(report.examples[1].refined.wer)
        self.assertEqual(report.examples[1].refined.insertions, 2)
        self.assertEqual(report.comparison.regressed, 1)

    def test_all_silence_has_null_wer_and_preserves_errors(self) -> None:
        report = self.evaluate(
            [self.row(reference="", baseline="", refined="unexpected words")]
        )
        self.assertIsNone(report.baseline.quality.wer)
        self.assertIsNone(report.refined.quality.wer)
        self.assertIsNone(report.comparison.wer_change)
        self.assertEqual(report.refined.quality.insertions, 2)
        json.dumps(asdict(report), allow_nan=False)

    def test_wer_can_exceed_one(self) -> None:
        report = self.evaluate(
            [self.row(reference="one", refined="one two three four")]
        )
        self.assertEqual(report.refined.quality.wer, 3)

    def test_whitespace_normalization_handles_tabs_and_unicode(self) -> None:
        report = self.evaluate(
            [self.row(reference="one\u2003two", baseline="\t one  two \t")]
        )
        self.assertEqual(report.baseline.quality.wer, 0)

    def test_default_preserves_case_and_punctuation(self) -> None:
        report = self.evaluate(
            [self.row(reference="Hello world", baseline="hello world!")]
        )
        self.assertEqual(report.baseline.quality.substitutions, 2)

    def test_lowercase_normalization_is_explicit(self) -> None:
        self.write_rows(
            [self.row(reference="Hello", baseline="hello", refined="HELLO")]
        )
        pairs, fingerprint = load_pairs(self.path)
        report = compare(
            pairs, fingerprint, reference_kind="verbatim", normalization="lowercase"
        )
        self.assertEqual(report.baseline.quality.wer, 0)
        self.assertEqual(report.refined.quality.wer, 0)
        self.assertEqual(report.reference_kind, "verbatim")

    def test_measurement_coverage_is_not_assumed_complete(self) -> None:
        report = self.evaluate(
            [
                self.row(
                    id="measured",
                    baseline_latency_ms=100,
                    refined_latency_ms=250,
                    baseline_cost_usd=0,
                    refined_cost_usd=0.1,
                ),
                self.row(id="unmeasured"),
            ]
        )
        self.assertEqual(report.example_count, 2)
        self.assertEqual(report.baseline.latency_ms.count, 1)
        self.assertEqual(report.refined.latency_ms.mean, 250)
        self.assertEqual(report.baseline.cost_usd.total, 0)
        self.assertEqual(report.refined.cost_usd.total, 0.1)

    def test_missing_measurements_are_null_not_zero(self) -> None:
        report = self.evaluate([self.row()])
        self.assertEqual(report.baseline.cost_usd.count, 0)
        self.assertIsNone(report.baseline.cost_usd.total)
        self.assertIsNone(report.baseline.latency_ms.mean)

    def test_report_contains_no_transcript_text(self) -> None:
        report = self.evaluate([self.row(reference="private transcript content")])
        self.assertNotIn("private transcript content", json.dumps(asdict(report)))

    def test_fingerprint_matches_exact_bytes(self) -> None:
        self.write_rows([self.row()])
        _, fingerprint = load_pairs(self.path)
        self.assertEqual(
            fingerprint, hashlib.sha256(self.path.read_bytes()).hexdigest()
        )

    def test_utf8_bom_and_blank_lines(self) -> None:
        self.path.write_text(
            "\ufeff\n" + json.dumps(self.row()) + "\n\n", encoding="utf-8"
        )
        pairs, _ = load_pairs(self.path)
        self.assertEqual(len(pairs), 1)

    def test_unicode_line_separators_inside_json_strings_are_not_rows(self) -> None:
        self.path.write_text(
            json.dumps(self.row(reference="one\u2028two"), ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        pairs, fingerprint = load_pairs(self.path)
        report = compare(pairs, fingerprint, reference_kind="intended")
        self.assertEqual(report.baseline.quality.wer, 0)

    def test_empty_files_are_rejected(self) -> None:
        for content in ("", " \n\n"):
            with self.subTest(content=content):
                self.path.write_text(content, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "no transcript pairs"):
                    load_pairs(self.path)

    def test_invalid_json_has_line_number(self) -> None:
        self.path.write_text(json.dumps(self.row()) + "\n{bad}\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "pairs.jsonl:2:"):
            load_pairs(self.path)

    def test_duplicate_json_keys_are_rejected(self) -> None:
        self.path.write_text('{"id":"one","id":"two"}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            load_pairs(self.path)

    def test_non_object_rows_are_rejected(self) -> None:
        for value in ([], None, "text", 4):
            with self.subTest(value=value):
                self.path.write_text(json.dumps(value), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "JSON object"):
                    load_pairs(self.path)

    def test_required_fields_must_be_strings(self) -> None:
        for field in ("id", "reference", "baseline", "refined"):
            for value in (None, [], {}, 1, True):
                with self.subTest(field=field, value=value):
                    self.write_rows([self.row(**{field: value})])
                    with self.assertRaisesRegex(ValueError, f"{field} is required"):
                        load_pairs(self.path)
            row = self.row()
            del row[field]
            self.write_rows([row])
            with self.assertRaisesRegex(ValueError, f"{field} is required"):
                load_pairs(self.path)

    def test_invalid_or_duplicate_ids_are_rejected(self) -> None:
        for identifier in ("", " ", " padded", "padded "):
            with self.subTest(identifier=identifier):
                self.write_rows([self.row(id=identifier)])
                with self.assertRaisesRegex(ValueError, "id must be non-empty"):
                    load_pairs(self.path)
        self.write_rows([self.row(), self.row()])
        with self.assertRaisesRegex(ValueError, "duplicate id"):
            load_pairs(self.path)

    def test_unknown_fields_are_rejected(self) -> None:
        self.write_rows([self.row(baseline_latenc_ms=5)])
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            load_pairs(self.path)

    def test_invalid_measurements_are_rejected(self) -> None:
        for value in (
            -1,
            True,
            None,
            "1",
            float("nan"),
            float("inf"),
            -float("inf"),
            10**400,
        ):
            with self.subTest(value=value):
                self.write_rows(
                    [self.row(baseline_latency_ms=value, refined_latency_ms=1)]
                )
                with self.assertRaises(ValueError):
                    load_pairs(self.path)

    def test_measurements_must_be_paired(self) -> None:
        for field in (
            "baseline_latency_ms",
            "refined_latency_ms",
            "baseline_cost_usd",
            "refined_cost_usd",
        ):
            with self.subTest(field=field):
                self.write_rows([self.row(**{field: 1})])
                with self.assertRaisesRegex(ValueError, "provide both"):
                    load_pairs(self.path)

    def test_empty_programmatic_comparison_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            compare([], "unused", reference_kind="intended")

    def test_cli_writes_valid_json_to_stdout(self) -> None:
        self.write_rows([self.row()])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = main(
                [
                    str(self.path),
                    "--reference-kind",
                    "intended",
                    "--baseline-label",
                    "Whisper",
                    "--refined-label",
                    "Staccato",
                ]
            )
        report = json.loads(output.getvalue())
        self.assertEqual(status, 0)
        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["baseline"]["label"], "Whisper")
        self.assertEqual(report["refined"]["label"], "Staccato")
        self.assertEqual(report["jiwer_version"], "4.0.0")

    def test_cli_output_never_overwrites_a_file(self) -> None:
        self.write_rows([self.row()])
        output_path = Path(self.directory.name) / "report.json"
        arguments = [
            str(self.path),
            "--reference-kind",
            "intended",
            "--output",
            str(output_path),
        ]
        self.assertEqual(main(arguments), 0)
        original = output_path.read_bytes()
        with (
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as ex,
        ):
            main(arguments)
        self.assertEqual(ex.exception.code, 2)
        self.assertEqual(output_path.read_bytes(), original)

    def test_cli_cannot_overwrite_input(self) -> None:
        self.write_rows([self.row()])
        original = self.path.read_bytes()
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(
                [
                    str(self.path),
                    "--reference-kind",
                    "intended",
                    "--output",
                    str(self.path),
                ]
            )
        self.assertEqual(self.path.read_bytes(), original)

    def test_cli_requires_explicit_reference_kind(self) -> None:
        with (
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as ex,
        ):
            main([str(self.path)])
        self.assertEqual(ex.exception.code, 2)

    def test_cli_input_errors_fail_without_success_output(self) -> None:
        self.path.write_text("{bad}", encoding="utf-8")
        output, error = io.StringIO(), io.StringIO()
        with (
            contextlib.redirect_stdout(output),
            contextlib.redirect_stderr(error),
            self.assertRaises(SystemExit) as ex,
        ):
            main([str(self.path), "--reference-kind", "intended"])
        self.assertEqual(ex.exception.code, 2)
        self.assertEqual(output.getvalue(), "")
        self.assertIn("pairs.jsonl:1:", error.getvalue())

    def test_cli_invalid_utf8_and_missing_files_are_explicit_errors(self) -> None:
        self.path.write_bytes(b"\xff")
        for path in (self.path, self.path.with_name("missing.jsonl")):
            with self.subTest(path=path):
                with (
                    contextlib.redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit) as ex,
                ):
                    main([str(path), "--reference-kind", "intended"])
                self.assertEqual(ex.exception.code, 2)

    def test_synthetic_fixture_includes_a_regression_and_silence(self) -> None:
        pairs, fingerprint = load_pairs(
            ROOT / "examples" / "synthetic_transcripts.jsonl"
        )
        report = compare(pairs, fingerprint, reference_kind="intended")
        self.assertEqual(report.baseline.quality.reference_words, 8)
        self.assertEqual(report.baseline.quality.wer, 0.125)
        self.assertEqual(report.refined.quality.wer, 0.375)
        self.assertEqual(report.comparison.wer_change, 0.25)
        self.assertEqual(report.comparison.improved, 1)
        self.assertEqual(report.comparison.regressed, 2)


if __name__ == "__main__":
    unittest.main()
