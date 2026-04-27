"""
ai_image_sentiment_annotator.py

Sentiment-label financial-news images using a 3-model frontier-LLM ensemble
(default: gpt-4o-mini + claude-sonnet-4-6 + gemini-3-pro-preview), producing
a CSV suitable for ViT transfer-learning dataset construction.

Adapted from ai_image_quality_annotator.py. The two scripts share auth loading,
data-URI encoding, and JSON extraction utilities (imported below) but differ in:
  - the prompt (financial sentiment, not technical quality)
  - the candidate sampler (default: reuse the existing quality-annotated pool;
    optional: walk the on-disk image library directly)
  - the output schema (per-model raw scores + ensemble aggregate)
  - the safety gates (dry-run by default; --confirm required for paid runs)

Cost-saving design (the user has emphasized that each run is expensive):
  - Default sampler reuses the 6,107 quality-annotated images and keeps only
    those rated `quality_pass_auto_accept` or `quality_review_needed`. This
    avoids re-paying API cost on images already known to be junk.
  - Always-on 3-model ensemble (no escalation): sentiment requires consensus.
  - Dry-run by default: prints sample size and rough cost estimate, exits.
  - --smoke N runs only the first N images (≈$0.10 for N=20) for prompt sanity.
  - --confirm flag is required for any paid run.
  - Resume support keys on absolute image_path so re-runs cost nothing.
"""

import argparse
import csv
import datetime as dt
import json
import logging
import random
import sys
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from tqdm import tqdm

# Make sibling module importable regardless of cwd / -m invocation.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ai_image_quality_annotator import (
    atomic_write_json,
    extract_json_object,
    image_id_from_path,
    image_to_data_uri,
    load_auth_config,
    safe_image_meta,
)

import json as _json
import time
import urllib.error
import urllib.request


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


SENTIMENT_PROMPT = (
    "You are labeling a financial-news image for investor-sentiment analysis. "
    "Look at visual cues that would emotionally affect an investor: rising vs. "
    "falling charts, smiling vs. distressed faces, prosperity vs. disaster scenes, "
    "construction vs. ruins, money vs. losses. "
    "Return ONLY a JSON object with these keys: "
    '"sentiment_score": number in [0, 1] where 0=very optimistic/bullish and 1=very pessimistic/bearish; '
    '"label": one of "positive" (clearly optimistic imagery), "negative" (clearly pessimistic imagery), '
    'or "neutral" (no clear directional sentiment, e.g. abstract diagrams, neutral portraits, generic logos); '
    '"reason": short explanation under 25 words. '
    "Be conservative — if the image is ambiguous, return neutral."
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # --- Sampling source ---
    p.add_argument(
        "--source",
        choices=["quality_csv", "filesystem"],
        default="quality_csv",
        help=(
            "Where to draw candidates from. 'quality_csv' (default, cheap): reuse images "
            "previously annotated as quality_pass_auto_accept or quality_review_needed. "
            "'filesystem': walk the on-disk image library directly."
        ),
    )
    p.add_argument(
        "--quality-csv",
        default="ai_image_annotation/run_artifacts/ai_image_annotations_run_20260409_clean.csv",
        help="Path to existing quality-annotation CSV (used when --source quality_csv).",
    )
    p.add_argument(
        "--quality-labels",
        default="quality_pass_auto_accept,quality_review_needed",
        help="Comma-separated quality labels to keep when sampling from quality CSV.",
    )
    p.add_argument(
        "--images-root",
        default="images",
        help="Root of the on-disk image library (used when --source filesystem).",
    )
    p.add_argument("--start-year", type=int, default=2014)
    p.add_argument("--end-year", type=int, default=2026)
    p.add_argument("--years", type=str, default="", help="Comma-separated years; overrides start/end.")

    # --- Sample size ---
    p.add_argument(
        "--per-year",
        type=int,
        default=500,
        help="Target images per year (stratified). Matches the prior quality-annotation scale.",
    )
    p.add_argument(
        "--max-total",
        type=int,
        default=0,
        help="Hard cap on total candidates after stratification (0 = no cap).",
    )
    p.add_argument("--seed", type=int, default=42)

    # --- API ---
    p.add_argument("--auth-file", default="ai_image_annotation/local_auth.json")
    p.add_argument(
        "--api-models",
        default="",
        help="Comma-separated models. Overrides models in auth-file when provided.",
    )
    p.add_argument("--api-timeout", type=int, default=60)
    p.add_argument("--api-max-retries", type=int, default=2)
    p.add_argument("--workers", type=int, default=6)

    # --- Outputs ---
    p.add_argument("--artifacts-dir", default="ai_image_annotation/run_artifacts")
    p.add_argument("--output-csv", default="ai_image_sentiment_annotations.csv")
    p.add_argument("--output-jsonl", default="ai_image_sentiment_annotations.jsonl")
    p.add_argument("--checkpoint-every", type=int, default=100)
    p.add_argument("--resume", action="store_true", help="Skip image_path values already present in output CSV.")
    p.add_argument("--reprocess", action="store_true", help="Delete existing output CSV/JSONL before run.")

    # --- Cost-safety gates ---
    p.add_argument(
        "--confirm",
        action="store_true",
        help="REQUIRED to actually call the API. Without this flag the script only "
             "prints the sample size and an estimated cost, then exits.",
    )
    p.add_argument(
        "--smoke",
        type=int,
        default=0,
        help="Run only the first N images (after sampling) — for prompt sanity checks. "
             "Still requires --confirm. 0 means run all sampled candidates.",
    )
    p.add_argument(
        "--cost-per-call",
        type=float,
        default=0.0025,
        help="Rough USD cost per single-model vision API call (used for the cost estimate only).",
    )

    args = p.parse_args()

    if args.per_year < 0:
        raise ValueError("--per-year must be >= 0")
    if args.max_total < 0:
        raise ValueError("--max-total must be >= 0")
    if args.smoke < 0:
        raise ValueError("--smoke must be >= 0")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

    return args


# ---------- API call (sentiment-specific prompt) ----------

def call_sentiment_api(api_base_url, api_key, model, image_path, timeout=60, max_retries=2):
    """Call a single vision model with the sentiment prompt. Returns dict or raises."""
    data_uri = image_to_data_uri(Path(image_path))

    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": SENTIMENT_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        ],
    }

    url = f"{api_base_url}/chat/completions"
    body = _json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    last_err = ""
    for attempt in range(max_retries + 1):
        try:
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                response_data = _json.loads(resp.read().decode("utf-8"))

            choices = response_data.get("choices", [])
            if not choices:
                raise ValueError("API response missing choices")

            content = choices[0].get("message", {}).get("content", "")
            parsed = extract_json_object(content)
            if not parsed:
                raise ValueError(f"model response is not valid JSON: {content[:200]}")

            score = float(parsed.get("sentiment_score", 0.5))
            score = max(0.0, min(1.0, score))
            label = str(parsed.get("label", "neutral")).strip().lower()
            if label not in {"positive", "negative", "neutral"}:
                # try to coerce ambiguous outputs into one of the three buckets
                if "neg" in label or "pess" in label or "bear" in label:
                    label = "negative"
                elif "pos" in label or "opt" in label or "bull" in label:
                    label = "positive"
                else:
                    label = "neutral"
            reason = str(parsed.get("reason", ""))[:200]

            return {
                "sentiment_score": round(score, 6),
                "label": label,
                "reason": reason,
            }
        except (urllib.error.URLError, urllib.error.HTTPError, ValueError, _json.JSONDecodeError) as exc:
            last_err = str(exc)
            if attempt < max_retries:
                time.sleep(1.2 * (attempt + 1))
                continue
            break

    raise RuntimeError(f"API call failed for model={model}: {last_err}")


def run_sentiment_ensemble(image_path, api_conf, timeout, max_retries):
    """Run all configured models on one image. Returns list of per-model results + errors."""
    results = []
    errors = []
    for model in api_conf["models"]:
        try:
            r = call_sentiment_api(
                api_base_url=api_conf["api_base_url"],
                api_key=api_conf["api_key"],
                model=model,
                image_path=image_path,
                timeout=timeout,
                max_retries=max_retries,
            )
            results.append({"model": model, **r})
        except Exception as exc:
            errors.append(f"{model}:{exc}")
    return results, errors


def aggregate_sentiment(results, errors):
    if not results:
        raise RuntimeError("all sentiment models failed: " + " | ".join(errors[:3]))

    mean_score = sum(r["sentiment_score"] for r in results) / len(results)
    label_counts = {}
    for r in results:
        label_counts[r["label"]] = label_counts.get(r["label"], 0) + 1
    majority_label = sorted(label_counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    agreement = max(label_counts.values()) / len(results)

    return {
        "mean_score": round(float(mean_score), 6),
        "majority_label": majority_label,
        "api_agreement": round(float(agreement), 6),
        "api_votes": _json.dumps(label_counts, ensure_ascii=False),
        "api_models_used": ",".join(r["model"] for r in results),
        "api_error": " | ".join(errors[:3]),
        "per_model": {r["model"]: r for r in results},
    }


# ---------- Candidate samplers ----------

def collect_from_quality_csv(csv_path: Path, allowed_labels, years, per_year, seed, max_total):
    if not csv_path.exists():
        raise FileNotFoundError(f"quality CSV not found: {csv_path}")

    by_year = defaultdict(list)
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = (row.get("predicted_quality_label") or "").strip()
            if label not in allowed_labels:
                continue
            try:
                yr = int(row.get("year") or 0)
            except ValueError:
                continue
            if yr not in years:
                continue
            img_path = (row.get("image_path") or "").strip()
            if not img_path or not Path(img_path).exists():
                continue
            by_year[yr].append({
                "year": yr,
                "news_date": row.get("news_date", ""),
                "link": row.get("link", ""),
                "title": row.get("title", ""),
                "image_path": img_path,
                "image_rel_path": row.get("image_rel_path", ""),
                "source_collection": row.get("source_collection", ""),
                "source_field": row.get("source_field", ""),
                "prior_quality_label": label,
                "prior_quality_score": row.get("quality_score", ""),
            })

    rng = random.Random(seed)
    selected = []
    for yr in sorted(by_year.keys()):
        items = by_year[yr]
        rng.shuffle(items)
        selected.extend(items[: per_year if per_year > 0 else len(items)])

    if max_total and len(selected) > max_total:
        rng.shuffle(selected)
        selected = selected[:max_total]

    return selected


def collect_from_filesystem(images_root: Path, years, per_year, seed, max_total):
    if not images_root.exists():
        raise FileNotFoundError(f"images root not found: {images_root}")

    by_year = defaultdict(list)
    for year in years:
        year_dir = images_root / f"{year}_1"
        if not year_dir.exists():
            logging.warning("year dir missing: %s", year_dir)
            continue
        for date_dir in year_dir.iterdir():
            if not date_dir.is_dir():
                continue
            news_date = date_dir.name
            for news_dir in date_dir.iterdir():
                if not news_dir.is_dir():
                    continue
                title = news_dir.name
                for img_path in news_dir.iterdir():
                    if not img_path.is_file():
                        continue
                    if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                        continue
                    by_year[year].append({
                        "year": year,
                        "news_date": news_date,
                        "link": "",
                        "title": title,
                        "image_path": str(img_path),
                        "image_rel_path": str(img_path.relative_to(images_root.parent)),
                        "source_collection": f"filesystem_{year}",
                        "source_field": "filesystem",
                        "prior_quality_label": "",
                        "prior_quality_score": "",
                    })

    rng = random.Random(seed)
    selected = []
    for yr in sorted(by_year.keys()):
        items = by_year[yr]
        rng.shuffle(items)
        selected.extend(items[: per_year if per_year > 0 else len(items)])

    if max_total and len(selected) > max_total:
        rng.shuffle(selected)
        selected = selected[:max_total]

    return selected


# ---------- IO ----------

FIELDNAMES = [
    "image_id",
    "year",
    "news_date",
    "link",
    "title",
    "source_collection",
    "source_field",
    "image_path",
    "image_rel_path",
    "width",
    "height",
    "prior_quality_label",
    "prior_quality_score",
    "mean_sentiment_score",
    "majority_sentiment_label",
    "api_agreement",
    "api_votes",
    "gpt_score", "gpt_label", "gpt_reason",
    "claude_score", "claude_label", "claude_reason",
    "gemini_score", "gemini_label", "gemini_reason",
    "api_models_used",
    "api_error",
    "created_at",
]


def load_processed_paths(csv_path: Path):
    processed = set()
    if not csv_path.exists():
        return processed
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ip = row.get("image_path")
            if ip:
                processed.add(ip)
    return processed


def append_rows(csv_path: Path, jsonl_path: Path, rows, write_header=False):
    with csv_path.open("a", encoding="utf-8", newline="") as cf, jsonl_path.open("a", encoding="utf-8") as jf:
        writer = csv.DictWriter(cf, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in FIELDNAMES})
            jf.write(_json.dumps(row, ensure_ascii=False) + "\n")


def slot_for_model(model_name: str):
    """Map provider model name to one of {gpt, claude, gemini} for fixed CSV columns."""
    n = model_name.lower()
    if "gpt" in n or "openai" in n:
        return "gpt"
    if "claude" in n or "sonnet" in n or "haiku" in n or "opus" in n:
        return "claude"
    if "gemini" in n or "google" in n:
        return "gemini"
    return None


def annotate_one(item, args, api_conf):
    image_path = item["image_path"]
    width, height = safe_image_meta(image_path)
    row = {
        "image_id": image_id_from_path(image_path),
        "year": item["year"],
        "news_date": item["news_date"],
        "link": item["link"],
        "title": item["title"],
        "source_collection": item["source_collection"],
        "source_field": item["source_field"],
        "image_path": image_path,
        "image_rel_path": item["image_rel_path"],
        "width": width,
        "height": height,
        "prior_quality_label": item.get("prior_quality_label", ""),
        "prior_quality_score": item.get("prior_quality_score", ""),
        "created_at": dt.datetime.now().isoformat(),
    }

    try:
        results, errors = run_sentiment_ensemble(
            image_path=image_path,
            api_conf=api_conf,
            timeout=args.api_timeout,
            max_retries=args.api_max_retries,
        )
        agg = aggregate_sentiment(results, errors)

        row.update({
            "mean_sentiment_score": agg["mean_score"],
            "majority_sentiment_label": agg["majority_label"],
            "api_agreement": agg["api_agreement"],
            "api_votes": agg["api_votes"],
            "api_models_used": agg["api_models_used"],
            "api_error": agg["api_error"],
        })
        for r in results:
            slot = slot_for_model(r["model"])
            if slot is None:
                continue
            row[f"{slot}_score"] = r["sentiment_score"]
            row[f"{slot}_label"] = r["label"]
            row[f"{slot}_reason"] = r["reason"]
        return row, False
    except Exception as exc:
        row.update({
            "mean_sentiment_score": "",
            "majority_sentiment_label": "error",
            "api_agreement": 0,
            "api_votes": "",
            "api_models_used": ",".join(api_conf["models"]),
            "api_error": str(exc)[:600],
        })
        return row, True


def build_years(args):
    if args.years.strip():
        return sorted({int(x.strip()) for x in args.years.split(",") if x.strip()})
    return list(range(args.start_year, args.end_year + 1))


def main():
    args = parse_args()
    random.seed(args.seed)

    years = set(build_years(args))
    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    csv_path = artifacts_dir / args.output_csv
    jsonl_path = artifacts_dir / args.output_jsonl
    summary_path = artifacts_dir / "ai_image_sentiment_annotator_summary.json"
    status_path = artifacts_dir / "ai_image_sentiment_annotator_status.json"
    checkpoint_path = artifacts_dir / "ai_image_sentiment_annotator_checkpoint.json"

    if args.reprocess:
        for p in [csv_path, jsonl_path, checkpoint_path]:
            if p.exists():
                p.unlink()

    # Sample candidates
    if args.source == "quality_csv":
        allowed = {x.strip() for x in args.quality_labels.split(",") if x.strip()}
        candidates = collect_from_quality_csv(
            Path(args.quality_csv), allowed, years, args.per_year, args.seed, args.max_total
        )
    else:
        candidates = collect_from_filesystem(
            Path(args.images_root), years, args.per_year, args.seed, args.max_total
        )

    # Resume
    processed = set()
    if args.resume and not args.reprocess:
        processed = load_processed_paths(csv_path)
    todo = [c for c in candidates if c["image_path"] not in processed]

    if args.smoke and args.smoke > 0:
        todo = todo[: args.smoke]

    # Cost estimate
    api_conf = load_auth_config(Path(args.auth_file))
    if args.api_models.strip():
        api_conf["models"] = [m.strip() for m in args.api_models.split(",") if m.strip()]
    n_models = len(api_conf["models"])
    n_calls = len(todo) * n_models
    est_cost = n_calls * args.cost_per_call

    by_year_count = defaultdict(int)
    for c in todo:
        by_year_count[c["year"]] += 1

    print("=" * 70)
    print("SENTIMENT ANNOTATION — RUN PLAN")
    print("=" * 70)
    print(f"Source           : {args.source}")
    if args.source == "quality_csv":
        print(f"  quality CSV    : {args.quality_csv}")
        print(f"  kept labels    : {args.quality_labels}")
    else:
        print(f"  images root    : {args.images_root}")
    print(f"Years            : {sorted(years)}")
    print(f"Per-year target  : {args.per_year}")
    print(f"Models           : {api_conf['models']}")
    print(f"Workers          : {args.workers}")
    print()
    print(f"Sampled candidates       : {len(candidates):>6d}")
    print(f"Already processed (resume): {len(processed):>6d}")
    print(f"Smoke cap                : {args.smoke if args.smoke > 0 else 'off'}")
    print(f"To annotate              : {len(todo):>6d}")
    print(f"  by year:")
    for yr in sorted(by_year_count):
        print(f"    {yr}: {by_year_count[yr]:>4d}")
    print()
    print(f"API calls (3 models × N) : {n_calls:>6d}")
    print(f"Estimated cost (USD)     : ${est_cost:.2f}  (@ ${args.cost_per_call:.4f}/call)")
    print(f"Output CSV               : {csv_path}")
    print("=" * 70)

    if not args.confirm:
        print()
        print("DRY-RUN MODE — no API calls made. Re-run with --confirm to proceed.")
        print("To prompt-test cheaply, add: --smoke 20 --confirm")
        return

    if not todo:
        print("Nothing to do.")
        return

    # Persist run metadata
    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    status = {
        "script": "ai_image_sentiment_annotator.py",
        "run_id": run_id,
        "started_at": dt.datetime.now().isoformat(),
        "to_process": len(todo),
        "config": vars(args),
    }
    atomic_write_json(status_path, status)

    masked = api_conf["api_key"][:3] + "***" + api_conf["api_key"][-3:]
    logging.info("API base=%s, key=%s, models=%s", api_conf["api_base_url"], masked, api_conf["models"])

    # Run
    write_header = not csv_path.exists()
    buffer = []
    processed_n = 0
    failed_n = 0
    start = dt.datetime.now()

    todo_iter = iter(todo)
    futures = {}
    inflight = max(args.workers * 2, args.workers)

    def submit_next(executor):
        try:
            nxt = next(todo_iter)
        except StopIteration:
            return False
        fut = executor.submit(annotate_one, nxt, args, api_conf)
        futures[fut] = nxt
        return True

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for _ in range(inflight):
            if not submit_next(ex):
                break
        with tqdm(total=len(todo), desc="Sentiment-annotating") as pbar:
            while futures:
                done, _wait = wait(futures.keys(), return_when=FIRST_COMPLETED)
                for fut in done:
                    item = futures.pop(fut)
                    row, is_failed = fut.result()
                    if is_failed:
                        failed_n += 1
                    else:
                        processed_n += 1
                    buffer.append(row)
                    pbar.update(1)

                    if len(buffer) >= 50:
                        append_rows(csv_path, jsonl_path, buffer, write_header=write_header)
                        write_header = False
                        buffer = []

                    if (processed_n + failed_n) % args.checkpoint_every == 0:
                        atomic_write_json(checkpoint_path, {
                            "run_id": run_id,
                            "last_updated": dt.datetime.now().isoformat(),
                            "processed_in_this_run": processed_n,
                            "failed_in_this_run": failed_n,
                            "last_image_path": item["image_path"],
                        })

                    submit_next(ex)

    if buffer:
        append_rows(csv_path, jsonl_path, buffer, write_header=write_header)

    summary = {
        "script": "ai_image_sentiment_annotator.py",
        "run_id": run_id,
        "started_at": start.isoformat(),
        "finished_at": dt.datetime.now().isoformat(),
        "duration_seconds": int((dt.datetime.now() - start).total_seconds()),
        "n_candidates": len(candidates),
        "n_to_process": len(todo),
        "processed": processed_n,
        "failed": failed_n,
        "estimated_cost_usd": round(est_cost, 4),
        "api_models": api_conf["models"],
        "config": vars(args),
        "output_csv": str(csv_path),
        "output_jsonl": str(jsonl_path),
    }
    atomic_write_json(summary_path, summary)
    logging.info("Done. Processed=%d Failed=%d", processed_n, failed_n)
    logging.info("Summary: %s", summary_path)


if __name__ == "__main__":
    main()
