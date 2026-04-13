import argparse
import csv
import datetime as dt
import json
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from ai_image_quality_annotator import annotate_one_item, load_auth_config


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
    "quality_score",
    "predicted_quality_label",
    "review_priority",
    "api_agreement",
    "api_votes",
    "api_models_used",
    "api_error",
    "human_decision",
    "human_label",
    "human_notes",
    "created_at",
]


def parse_args():
    p = argparse.ArgumentParser(description="Retry failed rows from a previous annotation run")
    p.add_argument("--retry-source", required=True, help="CSV with failed rows to retry")
    p.add_argument("--main-csv", required=True, help="Main annotations CSV to merge replacements into")
    p.add_argument("--auth-file", required=True, help="Auth JSON path")
    p.add_argument("--output-retry-csv", required=True, help="Output CSV for retry results")
    p.add_argument("--output-summary", required=True, help="Output JSON summary")
    p.add_argument("--backup-main-csv", required=True, help="Backup path for main CSV before merge")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--api-timeout", type=int, default=25)
    p.add_argument("--api-max-retries", type=int, default=2)
    p.add_argument("--checkpoint-every", type=int, default=50)
    return p.parse_args()


def read_csv_rows(path: Path):
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def write_rows(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(rows)


def load_done_map(path: Path):
    if not path.exists():
        return {}
    done = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            p = row.get("image_path", "")
            if p:
                done[p] = row
    return done


def merge_into_main(main_csv: Path, backup_csv: Path, replacement_rows):
    if not replacement_rows:
        return 0

    if not backup_csv.exists():
        backup_csv.write_text(main_csv.read_text(encoding="utf-8"), encoding="utf-8")

    replace_map = {r["image_path"]: r for r in replacement_rows}

    with main_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        main_fields = r.fieldnames
        merged = []
        replaced = 0
        for row in r:
            p = row.get("image_path", "")
            if p in replace_map:
                merged.append(replace_map[p])
                replaced += 1
            else:
                merged.append(row)

    with main_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=main_fields)
        w.writeheader()
        w.writerows(merged)

    return replaced


def main():
    args = parse_args()

    retry_source = Path(args.retry_source)
    main_csv = Path(args.main_csv)
    auth_file = Path(args.auth_file)
    retry_csv = Path(args.output_retry_csv)
    summary_json = Path(args.output_summary)
    backup_csv = Path(args.backup_main_csv)

    source_rows = read_csv_rows(retry_source)
    done_map = load_done_map(retry_csv)

    todo_items = []
    for row in source_rows:
        p = row.get("image_path", "")
        if not p:
            continue
        if p in done_map:
            continue
        todo_items.append(
            {
                "year": int(row.get("year") or 0),
                "news_date": row.get("news_date", ""),
                "link": row.get("link", ""),
                "title": row.get("title", ""),
                "image_path": row.get("image_path", ""),
                "image_rel_path": row.get("image_rel_path", ""),
                "source_collection": row.get("source_collection", ""),
                "source_field": row.get("source_field", "high_quality_images"),
            }
        )

    api_conf = load_auth_config(auth_file)
    models = api_conf["models"]
    primary_model = models[0]
    secondary_models = [m for m in models if m != primary_model]

    class RetryArgs:
        api_timeout = args.api_timeout
        api_max_retries = args.api_max_retries
        force_full_ensemble = True
        enable_escalation = True
        review_threshold = 0.55
        accept_threshold = 0.75
        escalate_band = 0.08

    rargs = RetryArgs()

    print(
        f"retry_source={retry_source} total={len(source_rows)} already_done={len(done_map)} to_process={len(todo_items)} models={models}"
    )

    results = list(done_map.values())
    completed_now = 0

    todo_iter = iter(todo_items)
    futures = {}
    inflight = max(args.workers * 2, args.workers)

    def submit_next(executor):
        try:
            item = next(todo_iter)
        except StopIteration:
            return False
        fut = executor.submit(annotate_one_item, item, rargs, api_conf, primary_model, secondary_models)
        futures[fut] = item
        return True

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for _ in range(inflight):
            if not submit_next(executor):
                break

        while futures:
            done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)
            for fut in done:
                item = futures.pop(fut)
                row, _label, _is_failed = fut.result()
                results.append(row)
                completed_now += 1

                if completed_now % args.checkpoint_every == 0:
                    write_rows(retry_csv, results)
                    ok = sum(1 for x in results if x.get("predicted_quality_label") != "error")
                    err = len(results) - ok
                    print(f"progress_now={completed_now}/{len(todo_items)} total_saved={len(results)} ok={ok} err={err}")

                submit_next(executor)

    write_rows(retry_csv, results)

    ok_rows = [r for r in results if r.get("predicted_quality_label") != "error"]
    err_rows = [r for r in results if r.get("predicted_quality_label") == "error"]
    replaced = merge_into_main(main_csv, backup_csv, ok_rows)

    summary = {
        "started_at": dt.datetime.now().isoformat(),
        "retry_source": str(retry_source),
        "retry_csv": str(retry_csv),
        "main_csv": str(main_csv),
        "backup_main_csv": str(backup_csv),
        "total_in_retry_source": len(source_rows),
        "already_done_before_this_run": len(done_map),
        "processed_now": completed_now,
        "total_results_saved": len(results),
        "total_success_non_error": len(ok_rows),
        "total_still_error": len(err_rows),
        "merged_replacements_into_main": replaced,
        "models": models,
        "finished_at": dt.datetime.now().isoformat(),
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
