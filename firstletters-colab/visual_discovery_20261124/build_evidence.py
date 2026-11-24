#!/usr/bin/env python3
"""Export the bounded numerical evidence from the original campaign directory.

This does not run an ink model, download anything, or regenerate native CT.
The complete original campaign is required for this exporter; it is not part
of the compact public package. Ordinary review only needs verify_evidence.py.
All path-bearing receipts are rewritten before publication. Historical hashes
refer to originals; manifest hashes refer to the actual public bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Original complete campaign directory")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--manifest-only", action="store_true",
                        help="Refresh public file hashes after documentation edits")
    args = parser.parse_args()
    source, out = args.source.resolve(), args.output.resolve()
    if source == out or source in out.parents or out in source.parents:
        raise ValueError("Source and publication directories must be separate")
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    origins = {}
    if args.manifest_only:
        origins = json.loads(manifest_path.read_text())["origins"]

    def clean_string(value: str) -> str:
        value = value.replace(str(source) + "/", "artifact://campaign/")
        value = value.replace(str(source.parent) + "/", "artifact://workspace/")
        # Also cover any historical session path embedded in a source receipt.
        return re.sub(r"/workspace/scratch/[^/\s]+/", "artifact://workspace/", value)

    def clean(value):
        if isinstance(value, dict):
            return {clean_string(str(k)): clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [clean(v) for v in value]
        return clean_string(value) if isinstance(value, str) else value

    def write_json(relative: str, value) -> None:
        target = out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")

    def copy(relative: str, target: str | None = None, sanitize_json: bool = False) -> None:
        target = target or relative
        origin, dest = source / relative, out / target
        dest.parent.mkdir(parents=True, exist_ok=True)
        if sanitize_json:
            write_json(target, clean(json.loads(origin.read_text())))
        else:
            dest.write_bytes(origin.read_bytes())
        origins[target] = {
            "source_relative_path": relative,
            "source_sha256": sha(origin),
            "transformation": ("JSON reserialization and workspace-path redaction"
                               if sanitize_json else "byte-for-byte copy"),
        }

    if not args.manifest_only:
        figures = [
            ("deliverables/PHerc0800_contour_contexte_reel.png", "0800_context.png"),
            ("deliverables/PHerc0125_raffinement_central.png", "0125_refinement.png"),
            ("deliverables/PHerc0125_quatre_cartes.png", "0125_four_maps.png"),
            ("deliverables/PHerc0800_comparaison_CT_modeles.png", "0800_ct_models.png"),
            ("deliverables/controle_public_PHerc0841.png", "0841_known_control.png"),
            ("pherc0800-frame-replay/numeric_profile_evidence.png", "0800_native_profiles.png"),
            ("root-review/0800_C_loop_all16depths.png", "0800_all_depths.png"),
        ]
        for origin, target in figures:
            copy(origin, "figures/" + target)

        for name in ["sample_values.csv", "historic_holdout_profiles.npz"]:
            copy("pherc0800-frame-replay/" + name, "data/" + name)

        json_sources = {
            "root-review/0125_refinement_seam_comparison.json": "metrics/0125_seams.json",
            "root-review/0800_context_extension_comparison.json": "metrics/0800_context.json",
            "root-review/author841_mosaic_numeric_calibration.json": "metrics/0841_calibration.json",
            "blind-review/pherc0800_C_context96_views/comparison_metrics.json": "metrics/0800_reader_context.json",
            "root-review/0800_C_loop_all16depths_review.json": "metrics/0800_depth_review.json",
            "pherc0800-frame-replay/all_source_receipts.json": "receipts/native_sources.json",
            "pherc0800-frame-replay/summary_receipt.json": "receipts/native_summary.json",
            "pherc0800-frame-replay/historic_holdout_results.json": "metrics/native_holdout.json",
            "pherc0800-frame-replay/historic_holdout_plan.json": "receipts/native_holdout_plan.json",
            "pherc0800-frame-replay/frozen_plan.json": "receipts/native_initial_plan.json",
            "pherc0800-official/ct_rois/C/studio_input_left_context96/receipt.json": "receipts/acquisition/0800_context96.json",
            "render-audit/0125-w025-v2/preparation_receipt.json": "receipts/acquisition/0125_preparation.json",
            "positive-control/author841_mosaic/receipt.json": "receipts/acquisition/0841_mosaic.json",
        }
        for origin, target in json_sources.items():
            copy(origin, target, sanitize_json=True)

        reviews = [
            "pherc0800_C_context96_round1_frozen.md",
            "pherc0800_C_context96_round2_comparison_frozen.md",
            "pherc0800_C_loop_depth16_frozen.md",
            "pherc0125_s48_round1_frozen.md",
            "pherc0125_s48_round2_comparison_frozen.md",
            "pherc0125_raw_snapped_round3_comparison_frozen.md",
            "author841_mosaic_visual_review_frozen.md",
        ]
        for name in reviews:
            copy("blind-review/" + name, "reviews/" + name)

        campaign = json.loads((source / "deliverables/campaign_summary.json").read_text())
        for record in campaign["records"]:
            receipt_relative = record["receipt"]
            if sha(source / receipt_relative) != record["receipt_sha256"]:
                raise ValueError("Original receipt hash mismatch: " + receipt_relative)
            receipt = json.loads((source / receipt_relative).read_text())
            for item in receipt["outputs"].values():
                path = source / "inference" / Path(item["file"]).name
                if sha(path) != item["sha256"]:
                    raise ValueError("Original model output mismatch: " + str(path.name))
            public_path = "receipts/" + receipt_relative
            copy(receipt_relative, public_path, sanitize_json=True)
            record["historical_receipt_sha256"] = record.pop("receipt_sha256")
            record["receipt"] = public_path
            record["published_receipt_sha256"] = sha(out / public_path)
        campaign["publication_scope"] = (
            "Compact evidence; historical model outputs and complete inputs are not all included. "
            "Published receipt hashes differ from originals because paths were redacted."
        )
        write_json("receipts/campaign_summary.json", campaign)
        origins["receipts/campaign_summary.json"] = {
            "source_relative_path": "deliverables/campaign_summary.json",
            "source_sha256": sha(source / "deliverables/campaign_summary.json"),
            "transformation": "Rebased public receipt links and split original/published hashes",
        }

        candidate = json.loads((source / "candidate_cards/PHerc0800_C_loop_01.json").read_text())
        candidate["evidence"] = [
            "metrics/0800_context.json", "metrics/0800_depth_review.json",
            "reviews/pherc0800_C_context96_round2_comparison_frozen.md",
            "figures/0800_context.png", "data/candidate_probability_crops.npz",
        ]
        candidate["ground_truth_eligible"] = False
        write_json("candidate.json", candidate)
        origins["candidate.json"] = {
            "source_relative_path": "candidate_cards/PHerc0800_C_loop_01.json",
            "source_sha256": sha(source / "candidate_cards/PHerc0800_C_loop_01.json"),
            "transformation": "Rebased evidence links; explicit ground_truth_eligible=false",
        }

        arrays, selections = {}, []
        crop_specs = [
            ("0125_raw_s96", "0125_raw_same_band_reverse16_s96", [180, 580, 530, 840], [530, 180, 840, 580]),
            ("0125_snapped_s96", "0125_snapped_same_band_reverse16_s96", [180, 580, 530, 840], [530, 180, 840, 580]),
            ("0125_raw_s48", "0125_raw_reverse_roi_s48", [100, 500, 98, 408], [530, 180, 840, 580]),
            ("0125_snapped_s48", "0125_snapped_reverse_roi_s48", [100, 500, 98, 408], [530, 180, 840, 580]),
            ("0800_old", "0800_C_left512_k08_to_k23_s96", [176, 416, 32, 336], [932, 826, 1236, 1066]),
            ("0800_new", "0800_C_left_context96_k08_to_k23_s96", [176, 416, 128, 432], [932, 826, 1236, 1066]),
        ]
        for prefix, job, box, global_box in crop_specs:
            y0, y1, x0, x1 = box
            for kind, suffix in [("probability", "prob"), ("context_valid", "context_valid")]:
                relative = f"inference/{job}.{suffix}.npy"
                array = np.load(source / relative, allow_pickle=False)
                crop = array[y0:y1, x0:x1].copy()
                if crop.shape != (y1-y0, x1-x0):
                    raise ValueError("Crop outside source: " + relative)
                key = prefix + "_" + kind
                arrays[key] = crop
                selections.append({
                    "key": key, "source_relative_path": relative,
                    "source_sha256": sha(source / relative),
                    "source_shape": list(array.shape), "dtype": str(array.dtype),
                    "slice_y0_y1_x0_x1": box, "global_xyxy": global_box,
                    "published_shape": list(crop.shape),
                    "published_array_sha256": hashlib.sha256(crop.tobytes(order="C")).hexdigest(),
                })
        (out / "data").mkdir(exist_ok=True)
        np.savez_compressed(out / "data/candidate_probability_crops.npz", **arrays)
        write_json("data/crop_manifest.json", {
            "scope": "Exact float32/bool crops of existing inference arrays; no new inference",
            "preview_quantizer": "numpy.rint(probability * 255).astype(uint8)",
            "array_hash_definition": "SHA256 of C-order raw array bytes, shape and dtype recorded separately",
            "selections": selections,
        })

        prob_path = source / "inference/author841_mosaic_positive_s96.prob.npy"
        mask_path = source / "inference/author841_mosaic_positive_s96.context_valid.npy"
        label_path = source / "positive-control/author841_mosaic/author841_mosaic_labels.npy"
        probability = np.load(prob_path, allow_pickle=False)
        context = np.load(mask_path, allow_pickle=False)
        labels = np.load(label_path, allow_pickle=False)
        indices = np.flatnonzero((labels == 1) | (labels == 2)).astype(np.int32)
        known = {
            "known_flat_index": indices,
            "known_label": labels.ravel()[indices],
            "known_probability": probability.ravel()[indices],
            "known_context_valid": context.ravel()[indices],
        }
        np.savez_compressed(out / "data/0841_known_predictions.npz", **known)
        write_json("data/0841_manifest.json", {
            "scope": "All and only publicly annotated pixels; model known-control diagnostic",
            "shape": list(labels.shape), "label_semantics": {"0": "unknown", "1": "background", "2": "ink"},
            "source_files": [{"source_relative_path": str(f.relative_to(source)), "sha256": sha(f)}
                             for f in [prob_path, mask_path, label_path]],
            "arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype),
                            "raw_c_order_sha256": hashlib.sha256(v.tobytes(order="C")).hexdigest()}
                       for k, v in known.items()},
        })

        report = (source / "deliverables/rapport_visuel.md").read_text()
        report = report.replace(
            "Les fichiers de ce dossier restent privés ; aucune revendication de découverte ou proposition de lecture n’a été publiée.",
            "Ce dossier public partage les observations et leurs limites, sur demande explicite de l’utilisateur. Aucune nouvelle lettre ni transcription n’est revendiquée.")
        report = report.replace("C’est un candidat de travail privé, pas une découverte de texte.",
                                "C’est un candidat de travail non confirmé, sans découverte de texte établie.")
        start, end = report.index("## 6. Dossier livré et reproduction"), report.index("### Sources principales")
        report = report[:start] + """## 6. Dossier public et vérification

Cette publication compacte conserve sept figures originales, les lectures figées, les reçus expurgés, les comparaisons numériques du replay, des crops exacts de probabilités et de masques, et les scores de tous les pixels annotés du contrôle PHerc0841. Les figures ne contiennent pas de tracé de lettre ajouté ; les contours rouges du contrôle 0841 proviennent exclusivement des annotations publiques connues.

Le script `verify_evidence.py` recalcule les métriques sur les valeurs publiées, sans modèle, GPU ni réseau. Il contrôle aussi les empreintes de chaque fichier public. Les SHA-256 des originaux figurent séparément des SHA-256 publiés lorsqu’un reçu a été expurgé ou un document adapté. Le dossier original est la source de cette extraction ; la publication ne prétend pas que les fichiers expurgés sont identiques bit à bit à leurs originaux.

Le paquet ne contient pas les volumes CT complets, les cartes entières, l’atlas HTML, les gros caches ni le checkpoint de 893 MB. Les vérifications autonomes portent donc sur les valeurs incluses. La stabilité des 57 600 pixels déjà admissibles du rectangle C est recalculable ; l’identité des 416 colonnes du plein cadre reste une observation documentée dans le reçu original. Refaire les acquisitions et les inférences demanderait les sources épinglées et des téléchargements supplémentaires ; ce vérificateur ne les réexécute pas.

`build_evidence.py` documente et reproduit l’extraction depuis le dossier complet original. Le [README](README.md) donne la commande de vérification ; [PROVENANCE.md](PROVENANCE.md) décrit les périmètres et les liens vers les sources.

""" + report[end:]
        (out / "REPORT.md").write_text(report)
        origins["REPORT.md"] = {
            "source_relative_path": "deliverables/rapport_visuel.md",
            "source_sha256": sha(source / "deliverables/rapport_visuel.md"),
            "transformation": "Public status and compact-distribution section adapted; scientific observations preserved",
            "original_word_count_whitespace_split": 3542,
        }

    files = []
    for path in sorted(out.rglob("*")):
        if not path.is_file() or path == manifest_path or "__pycache__" in path.parts:
            continue
        files.append({"path": str(path.relative_to(out)), "bytes": path.stat().st_size, "sha256": sha(path)})
    manifest = {
        "schema": "scrollprize.visual-evidence.v1", "campaign_date": "2026-11-24",
        "status": "no_new_letter_or_text_validated", "new_inference_during_publication": 0,
        "path_redaction": "Workspace paths replaced by artifact://campaign/ or artifact://workspace/; these are historical artifact identifiers, not public files",
        "hash_scope": "files lists actual public bytes; origins lists historical original bytes; manifest excludes itself",
        "origins": origins, "files": files, "listed_bytes": sum(row["bytes"] for row in files),
    }
    write_json("manifest.json", manifest)
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    if total >= 10_000_000:
        raise ValueError(f"Publication exceeds 10 MB budget: {total} bytes")
    print(json.dumps({"files": len(files) + 1, "bytes": total, "status": "built"}))


if __name__ == "__main__":
    main()
