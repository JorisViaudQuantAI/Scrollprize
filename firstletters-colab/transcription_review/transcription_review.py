#!/usr/bin/env python3
"""Audit a private visual-reading journal; never certify a reading or label."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import unicodedata

SCHEMA = "scrollprize.transcription-review.v1"
ORIENTATIONS = {"identity", "rot90", "rot180", "rot270", "mirror_x",
                "mirror_x_rot90", "mirror_x_rot180", "mirror_x_rot270", "unknown"}
GREEK = set("ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ")
SHA = re.compile(r"[0-9a-f]{64}\Z")
ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


class InvalidJournal(ValueError):
    """Invalid or insufficiently explicit journal contract."""


def require(ok, message):
    if not ok:
        raise InvalidJournal(message)


def fields(value, required, optional=()):
    require(isinstance(value, dict), "Expected an object")
    require(set(required) <= value.keys(), "Required fields are missing")
    require(value.keys() <= set(required) | set(optional), "Unexpected fields are forbidden")


def ident(value):
    require(isinstance(value, str) and ID.fullmatch(value), "Invalid identifier")
    return value


def integer(value, minimum=0):
    require(type(value) is int and value >= minimum, "Expected a bounded nonnegative integer")
    return value


def nonempty(value):
    require(isinstance(value, str) and bool(value.strip()), "An explicit explanation is required")


def normalise_greek(value):
    """Remove diacritics, retain iota subscript as iota, unify sigma forms."""
    require(isinstance(value, str) and bool(value), "A Greek form is required")
    result = []
    for char in unicodedata.normalize("NFD", value):
        if char == "\u0345":
            result.append("Ι")  # Letter-bearing iota must not silently disappear.
        elif unicodedata.category(char).startswith("M"):
            continue
        elif char in "ϲϹ":
            result.append("Σ")
        else:
            result.append(char.upper())
    word = "".join(result)
    require(bool(word) and set(word) <= GREEK, "Unsupported Greek form or mixed alphabet")
    return word


def unique_objects(items):
    require(isinstance(items, list), "Expected a list")
    result = {}
    for item in items:
        require(isinstance(item, dict), "Expected an object")
        key = ident(item.get("id"))
        require(key not in result, "Duplicate identifier")
        result[key] = item
    return result


def source_groups(sources):
    """Declared correlations, not an estimate of statistical independence."""
    parent = {key: key for key in sources}

    def root(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def join(a, b):
        a, b = root(a), root(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    by_hash, by_site = {}, {}
    for key, source in sources.items():
        for seen, field in ((by_hash, "sha256"), (by_site, "site_id")):
            value = source[field]
            if value in seen:
                join(key, seen[value])
            else:
                seen[value] = key
        for ancestor in source["derived_from"]:
            join(key, ancestor)
    groups = {}
    for key in sorted(sources):
        groups.setdefault(root(key), []).append(key)
    return sorted(groups.values())


def lattice(journal, line_id):
    line = next(line for line in journal["lines"] if line["id"] == line_id)
    units = []
    for unit in line["units"]:
        obs = [o for o in journal["observations"]
               if o["line_id"] == line_id and o["unit_index"] == unit["index"]]
        alternatives = sorted({normalise_greek(a) for o in obs for a in o["alternatives"]})
        units.append({**unit, "alternatives": alternatives,
                      "allow_unknown": any(o["allow_unknown"] for o in obs),
                      "observations": obs})
    return line, units


def fixed_span(journal, line_id, start, stop):
    integer(start)
    integer(stop, 1)
    require(line_id in {line["id"] for line in journal["lines"]}, "Unknown line")
    line, units = lattice(journal, line_id)
    require(line["orientation"] != "unknown", "Resolve orientation before lexical comparison")
    require(0 <= start < stop <= len(units), "Invalid contiguous half-open span")
    selected = units[start:stop]
    require([u["index"] for u in selected] == list(range(start, stop)), "Noncontiguous span")
    require(all(u["length_min"] == u["length_max"] == 1 for u in selected),
            "Lexical comparison requires explicitly fixed one-character units")
    return line, selected


def compatible(word, units):
    return len(word) == len(units) and all(
        unit["allow_unknown"] or char in unit["alternatives"]
        for char, unit in zip(word, units)
    )


def validate(journal):
    fields(journal, ("schema", "synthetic", "sources", "lines", "observations", "lexical_hypotheses"))
    require(journal["schema"] == SCHEMA, "Unsupported schema")
    require(type(journal["synthetic"]) is bool, "synthetic must be explicit")
    sources = unique_objects(journal["sources"])
    lines = unique_objects(journal["lines"])
    observations = unique_objects(journal["observations"])
    require(sources and lines and observations, "Sources, lines and observations are required")
    for source in sources.values():
        fields(source, ("id", "sha256", "kind", "shape_yx", "site_id", "derived_from", "relation"))
        require(isinstance(source["sha256"], str) and SHA.fullmatch(source["sha256"]), "Invalid source SHA256")
        require(source["kind"] in {"model_map", "display", "ct", "synthetic"}, "Not a visual evidence source")
        require(journal["synthetic"] or source["kind"] != "synthetic", "Synthetic evidence in a real journal")
        shape = source["shape_yx"]
        require(isinstance(shape, list) and len(shape) == 2, "Source shape must be H,W")
        for size in shape:
            integer(size, 1)
        ident(source["site_id"])
        ancestors = source["derived_from"]
        require(isinstance(ancestors, list) and len(set(ancestors)) == len(ancestors), "Invalid provenance parents")
        require(all(p in sources and p != source["id"] for p in ancestors), "Unknown or self provenance parent")
        require(source["relation"] in {"base", "same_pixels", "regrown_same_site", "derived"}, "Unknown source relation")
        require(bool(ancestors) == (source["relation"] != "base"), "Source relation needs explicit ancestry")
        if source["relation"] in {"same_pixels", "regrown_same_site"}:
            require(all(sources[p]["site_id"] == source["site_id"] for p in ancestors), "Related views must retain the same declared site")
    visiting, done = set(), set()

    def visit(key):
        require(key not in visiting, "Cyclic evidence ancestry")
        if key in done:
            return
        visiting.add(key)
        for ancestor in sources[key]["derived_from"]:
            visit(ancestor)
        visiting.remove(key)
        done.add(key)

    for key in sources:
        visit(key)
    for line in lines.values():
        fields(line, ("id", "orientation", "segmentation_basis", "units"))
        require(line["orientation"] in ORIENTATIONS, "Unknown orientation convention")
        nonempty(line["segmentation_basis"])
        require(isinstance(line["units"], list) and 0 < len(line["units"]) <= 1000, "Invalid reading-unit count")
        for index, unit in enumerate(line["units"]):
            fields(unit, ("index", "length_min", "length_max"))
            require(type(unit["index"]) is int and unit["index"] == index, "Units must be ordered and contiguous from zero")
            integer(unit["length_min"])
            if unit["length_max"] is not None:
                integer(unit["length_max"])
                require(unit["length_max"] >= unit["length_min"], "Invalid character-count interval")
    covered, repeated = set(), set()
    for obs in observations.values():
        fields(obs, ("id", "reviewer_id", "prior_verdict_seen", "source_id", "source_sha256", "line_id",
                     "unit_index", "orientation", "bbox_yxyx", "status", "alternatives", "allow_unknown", "strokes"))
        ident(obs["reviewer_id"])
        require(type(obs["prior_verdict_seen"]) is bool, "Review exposure must be explicit")
        require(obs["source_id"] in sources and obs["line_id"] in lines, "Observation has no visual source or line")
        source, line = sources[obs["source_id"]], lines[obs["line_id"]]
        require(obs["source_sha256"] == source["sha256"], "Observation/source hash mismatch")
        require(obs["orientation"] == line["orientation"], "Mixed line orientations; keep separate hypotheses")
        integer(obs["unit_index"])
        require(obs["unit_index"] < len(line["units"]), "Reading unit out of bounds")
        box = obs["bbox_yxyx"]
        require(isinstance(box, list) and len(box) == 4 and all(type(v) is int for v in box), "Bounding box must contain four integers")
        y0, x0, y1, x1 = box
        h, w = source["shape_yx"]
        require(0 <= y0 < y1 <= h and 0 <= x0 < x1 <= w, "Bounding box outside source raster")
        require(obs["status"] in {"visual_hypothesis", "abstain"}, "Observation cannot be lexical evidence or ground truth")
        require(type(obs["allow_unknown"]) is bool and isinstance(obs["alternatives"], list), "Unknowns and alternatives must be explicit")
        nonempty(obs["strokes"])
        alts = [normalise_greek(a) for a in obs["alternatives"]]
        require(all(len(a) == 1 for a in alts) and len(set(alts)) == len(alts), "Letter alternatives must be distinct single Greek characters")
        unit = line["units"][obs["unit_index"]]
        if obs["status"] == "abstain":
            require(not alts and obs["allow_unknown"], "Abstention must preserve an explicit unknown")
        else:
            require(bool(alts), "A visual hypothesis needs at least one letter alternative")
            if not (unit["length_min"] == unit["length_max"] == 1):
                require(obs["allow_unknown"], "Conditional glyph hints on an unsegmented unit must retain unknown content")
        key = (obs["reviewer_id"], obs["line_id"], obs["unit_index"], source["sha256"], obs["orientation"])
        require(key not in repeated, "Duplicate annotation of the same source bytes by the same reviewer")
        repeated.add(key)
        covered.add((obs["line_id"], obs["unit_index"]))
    require(covered == {(line["id"], u["index"]) for line in lines.values() for u in line["units"]},
            "Every unit needs an observation or explicit abstention")
    hypotheses = unique_objects(journal["lexical_hypotheses"])
    for hyp in hypotheses.values():
        fields(hyp, ("id", "text", "line_id", "start", "stop", "origin", "status", "ground_truth", "independent_evidence"))
        require(hyp["origin"] in {"human", "local_lexicon", "language_model"}, "Unknown lexical origin")
        require(hyp["status"] == "lexical_hypothesis_only" and hyp["ground_truth"] is False and hyp["independent_evidence"] is False,
                "A lexical hypothesis must never become independent evidence or ground truth")
        _, units = fixed_span(journal, hyp["line_id"], hyp["start"], hyp["stop"])
        require(compatible(normalise_greek(hyp["text"]), units), "Lexical form conflicts with frozen visual alternatives")
    return {"status": "schema_valid", "synthetic": journal["synthetic"],
            "sources": len(sources), "observations": len(observations),
            "declared_correlated_source_groups": len(source_groups(sources)),
            "reviewers_exposed_to_prior_verdicts": len({o["reviewer_id"] for o in observations.values() if o["prior_verdict_seen"]}),
            "source_assets_verified": False, "anatomical_alignment_certified": False,
            "independent_evidence_count": None, "ground_truth": False, "reading_certified": False}


def cell(value):
    text = str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    for char in ("\\", "|", "[", "]", "`", "*", "_"):
        text = text.replace(char, "\\" + char)
    return text.replace("\n", " ").replace("\r", " ")


def journal_digest(journal):
    return hashlib.sha256(json.dumps(journal, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def render(journal):
    summary = validate(journal)
    out = ["# Visual transcription review", "", "Synthetic fixture." if journal["synthetic"] else "Private review journal.", "",
           "This is a record of visual hypotheses and abstentions. No reading, ground truth, statistical independence or anatomical alignment is certified.", "",
           "Canonical journal SHA256: `" + journal_digest(journal) + "`.", "",
           "Source hashes, boxes, orientation and correspondence are declarations; source files have not been opened by this validator.", "",
           "## Evidence dependencies", "", "| Correlated group | Source IDs |", "| --- | --- |"]
    for index, group in enumerate(source_groups({s["id"]: s for s in journal["sources"]}), 1):
        out.append(f"| {index} | {cell(', '.join(group))} |")
    out += ["", "Same bytes, declared site or ancestry join sources into one group. Separate groups do not prove independence; no vote count is produced.", ""]
    for line in sorted(journal["lines"], key=lambda x: x["id"]):
        _, units = lattice(journal, line["id"])
        out += ["## Line " + cell(line["id"]), "", "Orientation: `" + line["orientation"] + "`. Segmentation basis: " + cell(line["segmentation_basis"]), "",
                "| Unit | Character count | Union of visual alternatives | Unknown retained | Interpretation |", "| --- | --- | --- | --- | --- |"]
        for unit in units:
            maximum = "unbounded" if unit["length_max"] is None else str(unit["length_max"])
            role = "letter alternatives" if unit["length_min"] == unit["length_max"] == 1 else "conditional glyph hints; unit unsegmented"
            out.append(f"| {unit['index']} | {unit['length_min']}..{maximum} | {cell(' / '.join(unit['alternatives']) or '—')} | {'yes' if unit['allow_unknown'] else 'no'} | {role} |")
        out += ["", "| Observation | Unit | Reviewer | Prior verdict seen | Source/hash | Native bbox y0,x0,y1,x1 | Status | Visual alternatives | Unknown | Visible strokes |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for obs in sorted((o for o in journal["observations"] if o["line_id"] == line["id"]), key=lambda x: (x["unit_index"], x["id"])):
            values = [obs["id"], obs["unit_index"], obs["reviewer_id"], obs["prior_verdict_seen"],
                      obs["source_id"] + " / " + obs["source_sha256"], obs["bbox_yxyx"], obs["status"],
                      " / ".join(obs["alternatives"]) or "—", obs["allow_unknown"], obs["strokes"]]
            out.append("| " + " | ".join(cell(v) for v in values) + " |")
        out += ["", "Alternatives are combined by union; unknown flags by logical OR. Disagreement never disappears through majority voting.", ""]
    out += ["## Lexical hypotheses", "", "All lexical forms remain hypotheses, with ground_truth=false and independent_evidence=false.", "",
            "| ID | Form | Line | Contiguous span | Origin |", "| --- | --- | --- | --- | --- |"]
    for hyp in sorted(journal["lexical_hypotheses"], key=lambda x: x["id"]):
        out.append("| " + " | ".join(cell(v) for v in (hyp["id"], hyp["text"], hyp["line_id"], f"[{hyp['start']},{hyp['stop']})", hyp["origin"])) + " |")
    if not journal["lexical_hypotheses"]:
        out += ["| — | No lexical hypothesis recorded | — | — | — |"]
    out += ["", "Exposed reviewers: " + str(summary["reviewers_exposed_to_prior_verdicts"]) + ". An exposed review must not be described as blind.", ""]
    return "\n".join(out)


def lexical_matches(journal, lexicon, line_id, start, stop, limit=25):
    validate(journal)
    integer(limit, 1)
    require(limit <= 100, "Result limit must be at most 100")
    _, units = fixed_span(journal, line_id, start, stop)
    lexicon = Path(lexicon)
    require(lexicon.stat().st_size <= 25_000_000, "Local lexicon exceeds the 25 MB limit")
    raw = lexicon.read_bytes()
    forms = {}
    skipped = 0
    for original in raw.decode("utf-8-sig").splitlines():
        original = original.strip()
        if not original or original.startswith("#"):
            continue
        try:
            word = normalise_greek(original)
        except InvalidJournal:
            skipped += 1
            continue
        if compatible(word, units):
            forms.setdefault(word, set()).add(original)
    matches = [{"normalised": word, "forms": sorted(forms[word]), "status": "lexical_hypothesis_only",
                "ground_truth": False, "independent_evidence": False} for word in sorted(forms)[:limit]]
    return {"journal_sha256": journal_digest(journal), "lexicon_sha256": hashlib.sha256(raw).hexdigest(),
            "line_id": line_id, "start": start, "stop": stop, "orientation": next(l["orientation"] for l in journal["lines"] if l["id"] == line_id),
            "matches": matches, "matching_normalised_forms": len(forms), "truncated": len(forms) > limit,
            "skipped_unsupported_entries": skipped, "sorting": "alphabetical, not confidence", "reading_certified": False}


def write_private(path, text):
    path = Path(path).resolve()
    require(not path.exists(), "Output already exists; preserve the previous review")
    require(not any((p / ".git").exists() for p in [path.parent, *path.parent.parents]), "Generated reading artifacts must remain outside Git worktrees")
    require(path.parent.is_dir(), "Create the private output directory first")
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("validate")
    check.add_argument("journal")
    report = sub.add_parser("report")
    report.add_argument("journal")
    report.add_argument("--output", required=True)
    suggest = sub.add_parser("suggest")
    suggest.add_argument("journal")
    suggest.add_argument("--lexicon", required=True)
    suggest.add_argument("--line", required=True)
    suggest.add_argument("--start", type=int, required=True)
    suggest.add_argument("--stop", type=int, required=True)
    suggest.add_argument("--limit", type=int, default=25)
    suggest.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        journal = json.loads(Path(args.journal).read_text(encoding="utf-8"))
        summary = validate(journal)
        if args.command == "report":
            write_private(args.output, render(journal))
        elif args.command == "suggest":
            result = lexical_matches(journal, args.lexicon, args.line, args.start, args.stop, args.limit)
            write_private(args.output, json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        # No candidate names, source paths, coordinates or letter suggestions on stdout.
        print(json.dumps(summary, sort_keys=True))
    except (InvalidJournal, OSError, ValueError, TypeError, KeyError, RecursionError) as error:
        # Avoid echoing filesystem paths or private values from third-party errors.
        message = str(error) if isinstance(error, InvalidJournal) else type(error).__name__
        print(json.dumps({"status": "invalid", "reason": message}), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
