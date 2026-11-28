"""Concrete bookkeeping risks, with synthetic data only."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from transcription_review import (InvalidJournal, fixed_span, lattice, lexical_matches,
                                  normalise_greek, render, source_groups, validate, write_private)


def fixture():
    return json.loads(Path(__file__).with_name("example.synthetic.json").read_text())


class ContractRisks(unittest.TestCase):
    def test_same_pixels_and_regrows_never_create_independent_votes(self):
        doc = fixture()
        alias = copy.deepcopy(doc["sources"][0])
        alias.update(id="renamed-same-bytes", site_id="different-declared-site")
        doc["sources"].append(alias)
        extra = copy.deepcopy(doc["observations"][0])
        extra.update(id="extra", reviewer_id="r-3", source_id=alias["id"])
        doc["observations"].append(extra)
        summary = validate(doc)
        self.assertEqual(summary["declared_correlated_source_groups"], 1)
        self.assertIsNone(summary["independent_evidence_count"])
        self.assertFalse(summary["ground_truth"])
        extra["reviewer_id"] = "r-1"
        with self.assertRaises(InvalidJournal):
            validate(doc)

    def test_abstention_survives_confident_other_reviewer(self):
        doc = fixture()
        _, units = lattice(doc, "line-1")
        self.assertEqual(units[0]["alternatives"], ["Α", "Λ"])
        self.assertEqual(units[1]["alternatives"], ["Ο"])
        self.assertTrue(units[1]["allow_unknown"])
        self.assertIn("| 1 | 1..1 | Ο | yes |", render(doc))
        doc["observations"][3]["allow_unknown"] = False
        with self.assertRaises(InvalidJournal):
            validate(doc)

    def test_lexical_promotion_to_ground_truth_is_rejected(self):
        doc = fixture()
        hyp = {"id": "h-1", "text": "λόγος", "line_id": "line-1", "start": 0, "stop": 5,
               "origin": "language_model", "status": "lexical_hypothesis_only",
               "ground_truth": False, "independent_evidence": False}
        doc["lexical_hypotheses"] = [hyp]
        validate(doc)
        for key in ("ground_truth", "independent_evidence"):
            hyp[key] = True
            with self.assertRaises(InvalidJournal):
                validate(doc)
            hyp[key] = False
        doc["observations"][0]["source_id"] = "h-1"
        with self.assertRaises(InvalidJournal):
            validate(doc)

    def test_unresolved_length_and_orientation_cannot_be_forced_into_a_word(self):
        doc = fixture()
        self.assertIn("0..unbounded", render(doc))
        with self.assertRaises(InvalidJournal):
            fixed_span(doc, "line-2", 0, 1)
        doc["lines"][1]["orientation"] = "identity"
        doc["observations"][-1]["orientation"] = "identity"
        validate(doc)
        with self.assertRaises(InvalidJournal):
            fixed_span(doc, "line-2", 0, 1)
        doc["observations"][0]["orientation"] = "mirror_x"
        with self.assertRaises(InvalidJournal):
            validate(doc)

    def test_missing_or_noncontiguous_units_are_not_silently_closed(self):
        doc = fixture()
        doc["observations"] = [o for o in doc["observations"] if o["id"] != "o-5"]
        with self.assertRaises(InvalidJournal):
            validate(doc)
        doc = fixture()
        doc["lines"][0]["units"][2]["index"] = 3
        with self.assertRaises(InvalidJournal):
            validate(doc)

    def test_unsegmented_block_retains_hints_without_becoming_one_character(self):
        doc = fixture()
        doc["lines"][1]["orientation"] = "identity"
        doc["observations"][-1].update(orientation="identity", status="visual_hypothesis", alternatives=["Ϲ", "Τ"])
        validate(doc)
        self.assertIn("conditional glyph hints; unit unsegmented", render(doc))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "words.txt"
            path.write_text("Τ\n", encoding="utf-8")
            with self.assertRaises(InvalidJournal):
                lexical_matches(doc, path, "line-2", 0, 1)
    def test_boxes_hashes_and_provenance_must_remain_consistent(self):
        for field, value in (("bbox_yxyx", [0, 0, 1000, 1]), ("source_sha256", "0" * 64)):
            doc = fixture()
            doc["observations"][0][field] = value
            with self.assertRaises(InvalidJournal):
                validate(doc)
        doc = fixture()
        doc["sources"][0].update(derived_from=["synthetic-regrow"], relation="derived")
        with self.assertRaises(InvalidJournal):
            validate(doc)

    def test_exact_local_lexicon_is_limited_and_never_a_reading(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "words.txt"
            path.write_text("λόγος\nΛΟΓΟϹ\nλαγός\nλόγον\nLOGOS\n", encoding="utf-8")
            result = lexical_matches(fixture(), path, "line-1", 0, 5, limit=1)
            self.assertEqual(result["matching_normalised_forms"], 2)
            self.assertEqual(result["skipped_unsupported_entries"], 1)
            self.assertTrue(result["truncated"])
            self.assertEqual(result["matches"][0]["normalised"], "ΛΑΓΟΣ")
            self.assertFalse(result["matches"][0]["ground_truth"])
            self.assertFalse(result["reading_certified"])
            self.assertEqual(result, lexical_matches(fixture(), path, "line-1", 0, 5, limit=1))

    def test_normalisation_preserves_letter_bearing_iota(self):
        self.assertEqual(normalise_greek("λόγοϲ"), "ΛΟΓΟΣ")
        self.assertEqual(normalise_greek("ᾳ"), "ΑΙ")
        with self.assertRaises(InvalidJournal):
            normalise_greek("AΛΦΑ")  # Latin A is not a Greek alpha.

    def test_output_never_overwrites_or_enters_a_git_worktree(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "report.md"
            write_private(output, render(fixture()))
            with self.assertRaises(InvalidJournal):
                write_private(output, "replacement")
            (Path(tmp) / ".git").write_text("gitdir: elsewhere")
            with self.assertRaises(InvalidJournal):
                write_private(Path(tmp) / "another.md", "private")


if __name__ == "__main__":
    unittest.main()
