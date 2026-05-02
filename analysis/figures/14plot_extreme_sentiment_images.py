#!/usr/bin/env python3

import argparse
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import imread
from PIL import Image
from pymongo import MongoClient


def parse_args():
    p = argparse.ArgumentParser(description="Plot top negative and top positive news images")
    p.add_argument("--mongo-uri", default="mongodb://localhost:27017/")
    p.add_argument("--db-name", default="sina_news_dataset_test")
    p.add_argument("--start-year", type=int, default=2014)
    p.add_argument("--end-year", type=int, default=2026)
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--cols", type=int, default=5)
    p.add_argument("--candidate-pool", type=int, default=3000, help="Initial per-year DB candidate pool size")
    p.add_argument("--max-candidate-pool", type=int, default=50000, help="Max per-year pool size when filtering")
    p.add_argument("--output", default="results/extreme_sentiment_images.png")
    p.add_argument("--collection-suffix", default="_old",
                   help="Suffix for sentiment collections (e.g. '_old' reads {year}_sentiment_old)")
    p.add_argument(
        "--require-text-sentiment",
        action="store_true",
        default=True,
        help="Only keep images from articles that also appear in {year}_text_sentiment",
    )
    p.add_argument("--no-require-text-sentiment", dest="require_text_sentiment", action="store_false")
    return p.parse_args()


def _normalized_row(d, year):
    return {
        "year": year,
        "news_date": str(d.get("news_date") or ""),
        "title": str(d.get("title") or ""),
        "prob": float(d.get("negative_likelihood")),
        "image_path": str(d.get("image_path") or ""),
        "original_id": d.get("original_id"),
    }


def _filter_candidates_with_text_sentiment(db, year, candidates):
    text_col_name = f"{year}_text_sentiment"
    if text_col_name not in db.list_collection_names():
        return []

    text_col = db[text_col_name]
    oids = [d.get("original_id") for d in candidates if d.get("original_id")]
    if not oids:
        return []

    allowed = set(
        d.get("original_id")
        for d in text_col.find({"original_id": {"$in": oids}}, {"original_id": 1})
        if d.get("original_id")
    )
    if not allowed:
        return []

    return [d for d in candidates if d.get("original_id") in allowed]


def pick_extremes(db, year, top_n, require_text_sentiment, candidate_pool, max_candidate_pool, collection_suffix=""):
    sentiment_col_name = f"{year}_sentiment{collection_suffix}"
    if sentiment_col_name not in db.list_collection_names():
        return [], []

    sentiment_col = db[sentiment_col_name]
    query = {"negative_likelihood": {"$ne": None}, "image_path": {"$ne": None}}

    projection = {
        "news_date": 1,
        "title": 1,
        "negative_likelihood": 1,
        "image_path": 1,
        "original_id": 1,
    }

    def select_for_direction(sort_dir):
        pool = max(200, candidate_pool)
        selected = []
        while pool <= max_candidate_pool:
            cursor = sentiment_col.find(query, projection).sort("negative_likelihood", sort_dir).limit(pool)
            candidates = [_normalized_row(d, year) for d in cursor if d.get("negative_likelihood") is not None]
            if require_text_sentiment:
                candidates = _filter_candidates_with_text_sentiment(db, year, candidates)

            # Deduplicate by image_path within year candidates
            seen_paths: set = set()
            unique_candidates = []
            for d in candidates:
                path = d.get("image_path", "")
                if not path:
                    continue
                p = Path(path)
                # Skip missing, empty, or tiny placeholder files (< 1 KB)
                if not p.exists() or not p.is_file() or p.stat().st_size < 1024:
                    continue
                if path in seen_paths:
                    continue
                seen_paths.add(path)
                unique_candidates.append(d)

            selected = unique_candidates
            if len(selected) >= top_n:
                break

            if pool == max_candidate_pool:
                break
            pool = min(max_candidate_pool, int(math.ceil(pool * 2)))

        return selected[:top_n]

    most_negative = select_for_direction(-1)
    most_positive = select_for_direction(1)
    return most_negative, most_positive


def _dhash_hex(image_path, hash_size=16):
    img = Image.open(image_path).convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    arr = np.array(img)
    diff = arr[:, 1:] > arr[:, :-1]
    bits = "".join("1" if x else "0" for x in diff.flatten())
    return hex(int(bits, 2))[2:].zfill((hash_size * hash_size) // 4)


def _content_key(row, hash_cache):
    image_path = str(row.get("image_path") or "")
    if not image_path:
        return ""
    if image_path in hash_cache:
        return hash_cache[image_path]

    p = Path(image_path)
    if not p.exists() or not p.is_file():
        hash_cache[image_path] = image_path
        return image_path

    try:
        digest = _dhash_hex(p)
        hash_cache[image_path] = digest
        return digest
    except Exception:
        hash_cache[image_path] = image_path
        return image_path


def _hamming(a, b):
    """Hamming distance between two equal-length hex strings (-1 if incompatible)."""
    if len(a) != len(b):
        return -1
    try:
        return bin(int(a, 16) ^ int(b, 16)).count("1")
    except ValueError:
        return -1


def dedup_rows(rows, top_n, hash_cache, pre_seen=None, hamming_threshold=12):
    """Deduplicate rows by perceptual hash: reject if Hamming distance <= threshold
    to any previously-kept hash. Non-hash keys (image_paths for unreadable files)
    still use exact-match dedup."""
    seen = set(pre_seen or [])
    seen_hex = [k for k in seen if all(c in "0123456789abcdef" for c in k)]
    selected = []
    for row in rows:
        key = _content_key(row, hash_cache)
        if not key:
            continue
        is_hex = all(c in "0123456789abcdef" for c in key)
        if is_hex:
            too_close = any(0 <= _hamming(key, k) <= hamming_threshold for k in seen_hex)
            if too_close:
                continue
            seen.add(key)
            seen_hex.append(key)
        else:
            if key in seen:
                continue
            seen.add(key)
        selected.append(row)
        if len(selected) >= top_n:
            break
    return selected, seen


def draw_grid(negative_rows, positive_rows, cols, output_path):
    all_rows = negative_rows + positive_rows
    if not all_rows:
        raise RuntimeError("No rows selected for plotting.")

    n_neg = len(negative_rows)
    n_pos = len(positive_rows)

    neg_rows = (n_neg + cols - 1) // cols
    pos_rows = (n_pos + cols - 1) // cols
    total_rows = neg_rows + pos_rows

    fig, axes = plt.subplots(total_rows, cols, figsize=(4.2 * cols, 3.6 * total_rows), facecolor="#e8e8e8")
    if total_rows == 1:
        axes = [axes]

    flat_axes = []
    for row_axes in axes:
        if isinstance(row_axes, (list, tuple)):
            flat_axes.extend(row_axes)
        else:
            try:
                flat_axes.extend(list(row_axes))
            except TypeError:
                flat_axes.append(row_axes)

    for ax in flat_axes:
        ax.axis("off")

    for i, row in enumerate(all_rows):
        ax = flat_axes[i]
        try:
            img = imread(row["image_path"])
            ax.imshow(img)
        except Exception:
            ax.text(0.5, 0.5, "Image load failed", ha="center", va="center", fontsize=10)

        sent_type = "Negative" if i < n_neg else "Positive"
        ax.set_title(
            f"Date: {row['news_date']}\nProb: {row['prob']:.3f}\nType: {sent_type}",
            fontsize=10,
        )
        ax.axis("off")

    fig.suptitle("Most Negative and Most Positive News Images", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    print(f"Saved: {output_path}")


def main():
    args = parse_args()
    t0 = time.time()
    client = MongoClient(args.mongo_uri)
    db = client[args.db_name]

    neg_all = []
    pos_all = []
    for year in range(args.start_year, args.end_year + 1):
        neg, pos = pick_extremes(
            db,
            year,
            args.top_n,
            args.require_text_sentiment,
            args.candidate_pool,
            args.max_candidate_pool,
            collection_suffix=args.collection_suffix,
        )
        neg_all.extend(neg)
        pos_all.extend(pos)

    # Global top-N from year-level candidates
    neg_sorted = sorted(neg_all, key=lambda x: x["prob"], reverse=True)
    pos_sorted = sorted(pos_all, key=lambda x: x["prob"])

    # Deduplicate by image content hash to avoid repeated visuals in the collage.
    hash_cache = {}
    neg_all, seen_hashes = dedup_rows(neg_sorted, args.top_n, hash_cache)
    pos_all, _ = dedup_rows(pos_sorted, args.top_n, hash_cache, pre_seen=seen_hashes)

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path

    draw_grid(neg_all, pos_all, args.cols, output_path)
    print(f"Rows selected: negative={len(neg_all)} positive={len(pos_all)}")
    print(f"Elapsed seconds: {time.time() - t0:.2f}")


if __name__ == "__main__":
    main()
