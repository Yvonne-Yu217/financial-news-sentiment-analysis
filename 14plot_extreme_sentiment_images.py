#!/usr/bin/env python3

import argparse
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.image import imread
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


def pick_extremes(db, year, top_n, require_text_sentiment, candidate_pool, max_candidate_pool):
    sentiment_col_name = f"{year}_sentiment"
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

            selected = [d for d in candidates if d.get("image_path")]
            if len(selected) >= top_n:
                break

            if pool == max_candidate_pool:
                break
            pool = min(max_candidate_pool, int(math.ceil(pool * 2)))

        return selected[:top_n]

    most_negative = select_for_direction(-1)
    most_positive = select_for_direction(1)
    return most_negative, most_positive


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
        )
        neg_all.extend(neg)
        pos_all.extend(pos)

    # Global top-N from year-level candidates
    neg_all = sorted(neg_all, key=lambda x: x["prob"], reverse=True)[: args.top_n]
    pos_all = sorted(pos_all, key=lambda x: x["prob"])[: args.top_n]

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path

    draw_grid(neg_all, pos_all, args.cols, output_path)
    print(f"Rows selected: negative={len(neg_all)} positive={len(pos_all)}")
    print(f"Elapsed seconds: {time.time() - t0:.2f}")


if __name__ == "__main__":
    main()
