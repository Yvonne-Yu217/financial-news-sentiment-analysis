import argparse
import csv
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

# Stable and practical filtering rules for ViT dataset construction.
POS_SCORE_TH = 0.70
NEG_SCORE_TH = 0.55
AGREE_TH = 0.60


def parse_args():
    parser = argparse.ArgumentParser(description="Build ViT dataset from annotation CSV")
    parser.add_argument(
        "--src-csv",
        default="ai_image_annotation/run_artifacts/ai_image_annotations.csv",
        help="Annotation CSV input path",
    )
    parser.add_argument(
        "--out-root",
        default="training_dataset",
        help="Output dataset root (will be recreated)",
    )
    parser.add_argument(
        "--pos-score-th",
        type=float,
        default=POS_SCORE_TH,
        help="Minimum score for review_needed to be included as positive",
    )
    parser.add_argument(
        "--neg-score-th",
        type=float,
        default=NEG_SCORE_TH,
        help="Maximum score for fail_candidate to be included as negative",
    )
    parser.add_argument(
        "--agree-th",
        type=float,
        default=AGREE_TH,
        help="Minimum agreement threshold for non-auto labels",
    )
    return parser.parse_args()


def build_dataset(src_csv: Path, out_root: Path, pos_score_th: float, neg_score_th: float, agree_th: float):
    pos_dir = out_root / "positive"
    neg_dir = out_root / "negative"
    summary_json = out_root / "dataset_build_summary.json"
    manifest_csv = out_root / "dataset_manifest.csv"

    if not src_csv.exists():
        raise FileNotFoundError(f"Annotation CSV not found: {src_csv}")

    if out_root.exists():
        shutil.rmtree(out_root)
    pos_dir.mkdir(parents=True, exist_ok=True)
    neg_dir.mkdir(parents=True, exist_ok=True)

    stats = Counter()
    raw_label_counts = Counter()
    class_scores = defaultdict(list)
    year_distribution = defaultdict(Counter)

    fields = [
        "class_label",
        "year",
        "predicted_quality_label",
        "quality_score",
        "api_agreement",
        "source_path",
        "target_path",
    ]

    with src_csv.open("r", encoding="utf-8", newline="") as src, manifest_csv.open(
        "w", encoding="utf-8", newline=""
    ) as manifest:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(manifest, fieldnames=fields)
        writer.writeheader()

        for row in reader:
            stats["total_rows"] += 1

            pred = (row.get("predicted_quality_label") or "").strip()
            raw_label_counts[pred] += 1

            if pred == "error":
                stats["skip_error"] += 1
                continue

            src_path = Path((row.get("image_path") or "").strip())
            if not src_path.exists():
                stats["skip_missing_file"] += 1
                continue

            try:
                score = float(row.get("quality_score") or 0)
            except Exception:
                score = 0.0
            try:
                agree = float(row.get("api_agreement") or 0)
            except Exception:
                agree = 0.0

            target_class = None
            if pred == "quality_pass_auto_accept":
                target_class = "positive"
            elif pred == "quality_review_needed" and score >= pos_score_th and agree >= agree_th:
                target_class = "positive"
            elif pred == "quality_fail_candidate" and score <= neg_score_th and agree >= agree_th:
                target_class = "negative"

            if target_class is None:
                stats["skip_rule_filtered"] += 1
                continue

            ext = src_path.suffix.lower() if src_path.suffix else ".jpg"
            unique_name = f"{src_path.stem}_{abs(hash(str(src_path.resolve()))) % 10**12}{ext}"
            dst_path = (pos_dir if target_class == "positive" else neg_dir) / unique_name

            if dst_path.exists() or dst_path.is_symlink():
                stats["skip_duplicate_target"] += 1
                continue

            # Symlink to avoid duplicating files and speed up dataset build.
            os.symlink(src_path.resolve(), dst_path)

            stats["included"] += 1
            stats[f"included_{target_class}"] += 1
            class_scores[target_class].append(score)
            year_distribution[target_class][str(row.get("year") or "")] += 1

            writer.writerow(
                {
                    "class_label": target_class,
                    "year": str(row.get("year") or ""),
                    "predicted_quality_label": pred,
                    "quality_score": f"{score:.6f}",
                    "api_agreement": f"{agree:.6f}",
                    "source_path": str(src_path),
                    "target_path": str(dst_path),
                }
            )

    summary = {
        "rules": {
            "positive": f"pass_auto_accept OR (review_needed & score>={pos_score_th} & agreement>={agree_th})",
            "negative": f"fail_candidate & score<={neg_score_th} & agreement>={agree_th}",
        },
        "counts": dict(stats),
        "raw_label_counts": dict(raw_label_counts),
        "score_stats": {
            cls: {
                "n": len(vals),
                "min": min(vals) if vals else None,
                "max": max(vals) if vals else None,
                "mean": (sum(vals) / len(vals)) if vals else None,
            }
            for cls, vals in class_scores.items()
        },
        "year_distribution": {
            cls: dict(sorted(year_distribution[cls].items(), key=lambda x: x[0]))
            for cls in year_distribution
        },
        "artifacts": {
            "manifest_csv": str(manifest_csv),
            "summary_json": str(summary_json),
            "dataset_root": str(out_root),
            "source_csv": str(src_csv),
        },
    }

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("BUILD_OK")
    print("included_total", stats["included"])
    print("included_positive", stats["included_positive"])
    print("included_negative", stats["included_negative"])
    print("skip_rule_filtered", stats["skip_rule_filtered"])
    print("skip_error", stats["skip_error"])
    print("skip_missing_file", stats["skip_missing_file"])
    print("skip_duplicate_target", stats["skip_duplicate_target"])


if __name__ == "__main__":
    args = parse_args()
    build_dataset(
        Path(args.src_csv),
        Path(args.out_root),
        float(args.pos_score_th),
        float(args.neg_score_th),
        float(args.agree_th),
    )
