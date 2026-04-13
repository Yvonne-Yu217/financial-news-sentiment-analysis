#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import datetime as dt
import json
import logging
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pymongo import MongoClient, UpdateOne


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

MOJIBAKE_HINT_RE = re.compile(r"(�|Ã|Â|â€|â€™|â€œ|â€“|â€”|ï¼|ï½|Ð|Ñ|å.|ç.|æ.)")
PRIVATE_USE_RE = re.compile(r"[\ue000-\uf8ff]")
CJK_RE = re.compile(r"[\u4e00-\u9fff]")

# Common Chinese characters usually frequent in normal news text.
COMMON_ZH_RE = re.compile(r"[的一是在不了有和人这中大为上个国我以要他时来用们生到作地于出就分对成会可主发年动同工也能下过子说产种面而方后多定行学法所民得经十三之进着等]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair mojibake text in MongoDB and optionally rerun text-only pipeline")
    parser.add_argument("--mongo-uri", default="mongodb://localhost:27017/")
    parser.add_argument("--db-name", default="sina_news_dataset_test")
    parser.add_argument("--start-year", type=int, default=2014)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument(
        "--suffixes",
        default="_1,_filtered,_2,_mapping",
        help="Comma-separated collection suffixes to repair",
    )
    parser.add_argument(
        "--fields",
        default="title,content",
        help="Comma-separated fields to repair",
    )
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--apply", action="store_true", help="Apply updates to MongoDB. Default is dry-run")
    parser.add_argument("--min-score-gain", type=float, default=8.0)
    parser.add_argument("--artifacts-dir", default="run_artifacts")
    parser.add_argument("--rerun-textpes", action="store_true", help="After repair, rerun script 11")
    parser.add_argument("--rerun-merge", action="store_true", help="After repair, rerun script 13")
    parser.add_argument("--python-bin", default=sys.executable, help="Python executable for rerun steps")
    parser.add_argument("--project-root", default=".")
    return parser.parse_args()


def text_quality_score(text: str) -> float:
    if text is None:
        return -1e9
    if not isinstance(text, str):
        text = str(text)
    t = text.strip()
    if not t:
        return -1e9

    n = max(len(t), 1)
    cjk = len(CJK_RE.findall(t))
    common = len(COMMON_ZH_RE.findall(t))
    mojibake_hits = len(MOJIBAKE_HINT_RE.findall(t))
    replacement_hits = t.count("�")
    private_hits = len(PRIVATE_USE_RE.findall(t))

    cjk_ratio = cjk / n
    common_ratio = common / n

    score = 0.0
    score += cjk_ratio * 120.0
    score += common_ratio * 280.0
    score += min(common, 200) * 0.05
    score -= mojibake_hits * 3.5
    score -= replacement_hits * 10.0
    score -= private_hits * 5.0
    return score


def is_suspicious(text: Optional[str]) -> bool:
    if text is None:
        return False
    if not isinstance(text, str):
        text = str(text)
    t = text.strip()
    if not t:
        return False

    if "�" in t:
        return True
    if MOJIBAKE_HINT_RE.search(t):
        return True
    if PRIVATE_USE_RE.search(t):
        return True

    # Heuristic: CJK-heavy but with very low common-char ratio can be mojibake like "鍚戠潃..."
    cjk = len(CJK_RE.findall(t))
    if cjk >= 8:
        common = len(COMMON_ZH_RE.findall(t))
        if common / max(len(t), 1) < 0.035:
            return True

    return False


def repair_candidates(text: str) -> List[str]:
    vals = [text]

    def try_convert(fn):
        try:
            out = fn(text)
            if isinstance(out, str) and out and out not in vals:
                vals.append(out)
        except Exception:
            return

    # Common inverse transforms for UTF-8/GBK mojibake.
    try_convert(lambda s: s.encode("gbk", errors="strict").decode("utf-8", errors="strict"))
    try_convert(lambda s: s.encode("gb18030", errors="strict").decode("utf-8", errors="strict"))
    try_convert(lambda s: s.encode("latin1", errors="strict").decode("utf-8", errors="strict"))

    # Less strict fallbacks.
    try_convert(lambda s: s.encode("gbk", errors="ignore").decode("utf-8", errors="ignore"))
    try_convert(lambda s: s.encode("gb18030", errors="ignore").decode("utf-8", errors="ignore"))

    return vals


def best_repair(text: str, min_gain: float) -> Tuple[bool, str, float, float]:
    cands = repair_candidates(text)
    base = cands[0]
    base_score = text_quality_score(base)

    best = base
    best_score = base_score
    for c in cands[1:]:
        s = text_quality_score(c)
        if s > best_score:
            best_score = s
            best = c

    changed = best != base and (best_score - base_score) >= min_gain
    return changed, best, base_score, best_score


def run_textpes_and_merge(args: argparse.Namespace) -> None:
    root = Path(args.project_root).resolve()
    script11 = root / "11calculate_daily_textpes.py"
    script13 = root / "13merge_data_and_calculate_returns.py"

    years = [str(y) for y in range(args.start_year, args.end_year + 1)]

    if args.rerun_textpes:
        cmd11 = [
            args.python_bin,
            str(script11),
            "--years",
            *years,
            "--output",
            str(root / "results" / "weighted_textpes.csv"),
        ]
        logging.info("Running script 11: %s", " ".join(cmd11))
        subprocess.run(cmd11, check=True)

    if args.rerun_merge:
        cmd13 = [args.python_bin, str(script13)]
        logging.info("Running script 13: %s", " ".join(cmd13))
        subprocess.run(cmd13, check=True)


def main() -> None:
    args = parse_args()

    client = MongoClient(args.mongo_uri)
    db = client[args.db_name]

    suffixes = [s.strip() for s in args.suffixes.split(",") if s.strip()]
    fields = [f.strip() for f in args.fields.split(",") if f.strip()]

    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = artifacts_dir / f"text_encoding_repair_report_{ts}.json"

    total_scanned = 0
    total_suspicious = 0
    total_repaired = 0
    per_collection: Dict[str, Dict[str, int]] = {}

    for y in range(args.start_year, args.end_year + 1):
        for suffix in suffixes:
            cname = f"{y}{suffix}"
            if cname not in db.list_collection_names():
                continue

            coll = db[cname]
            projection = {"_id": 1}
            for f in fields:
                projection[f] = 1

            stats = {"scanned": 0, "suspicious": 0, "repaired": 0}
            ops: List[UpdateOne] = []

            cursor = coll.find({}, projection, no_cursor_timeout=True)
            for doc in cursor:
                stats["scanned"] += 1
                total_scanned += 1

                updates = {}
                any_suspicious = False
                for f in fields:
                    val = doc.get(f)
                    if val is None or not isinstance(val, str):
                        continue
                    if not is_suspicious(val):
                        continue

                    any_suspicious = True
                    changed, fixed, _base_score, _new_score = best_repair(val, args.min_score_gain)
                    if changed:
                        updates[f] = fixed

                if any_suspicious:
                    stats["suspicious"] += 1
                    total_suspicious += 1
                if updates:
                    stats["repaired"] += 1
                    total_repaired += 1
                    if args.apply:
                        updates["encoding_repaired_at"] = dt.datetime.now()
                        ops.append(UpdateOne({"_id": doc["_id"]}, {"$set": updates}))

                if args.apply and len(ops) >= args.batch_size:
                    coll.bulk_write(ops, ordered=False)
                    ops = []

            if args.apply and ops:
                coll.bulk_write(ops, ordered=False)

            if stats["scanned"] > 0:
                per_collection[cname] = stats
                logging.info(
                    "%s scanned=%d suspicious=%d repaired=%d",
                    cname,
                    stats["scanned"],
                    stats["suspicious"],
                    stats["repaired"],
                )

    report = {
        "generated_at": dt.datetime.now().isoformat(),
        "mode": "apply" if args.apply else "dry-run",
        "mongo_uri": args.mongo_uri,
        "db_name": args.db_name,
        "year_range": [args.start_year, args.end_year],
        "suffixes": suffixes,
        "fields": fields,
        "min_score_gain": args.min_score_gain,
        "summary": {
            "total_scanned": total_scanned,
            "total_suspicious": total_suspicious,
            "total_repaired": total_repaired,
        },
        "per_collection": per_collection,
    }

    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("Repair report saved: %s", report_path)

    if args.apply and (args.rerun_textpes or args.rerun_merge):
        run_textpes_and_merge(args)


if __name__ == "__main__":
    main()
