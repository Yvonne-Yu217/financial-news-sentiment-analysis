import argparse
import datetime
import hashlib
import json
import logging
import os
import re
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytesseract
from PIL import Image
from pymongo import MongoClient, UpdateOne
from pytesseract import Output
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

START_YEAR = 2014
END_YEAR = 2026
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "sina_news_dataset_test"
SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPT_STEM = Path(__file__).stem
DEFAULT_ARTIFACTS_DIR = SCRIPT_DIR / "run_artifacts"

DEFAULT_QUALITY_SCORE_THRESHOLD = 0.5
DEFAULT_MIN_TEXT_LENGTH = 1
DEFAULT_OCR_LANG = "chi_sim+eng"


def format_eta(seconds: float) -> str:
    if seconds <= 0:
        return "00:00:00"
    total = int(seconds)
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def resolve_artifact_path(path_arg: Optional[str], artifacts_dir: Path, default_name: str) -> str:
    if not path_arg:
        return str(artifacts_dir / default_name)

    user_path = Path(path_arg)
    if user_path.is_absolute() or user_path.parent != Path("."):
        user_path.parent.mkdir(parents=True, exist_ok=True)
        return str(user_path)

    return str(artifacts_dir / user_path.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process filtered images into quality-scored records")
    parser.add_argument("--mongo-uri", type=str, default=MONGO_URI, help="MongoDB URI")
    parser.add_argument("--db-name", type=str, default=DB_NAME, help="MongoDB database name")
    parser.add_argument("--start-year", type=int, default=START_YEAR, help="Start year")
    parser.add_argument("--end-year", type=int, default=END_YEAR, help="End year")
    parser.add_argument("--years", type=str, default="", help="Optional comma-separated years")
    parser.add_argument("--batch-size", type=int, default=200, help="MongoDB bulk write batch size")
    parser.add_argument("--checkpoint-every", type=int, default=100, help="Save checkpoint/status every N records")
    parser.add_argument("--max-items-per-year", type=int, default=0, help="Limit records per year for testing")
    parser.add_argument("--image-workers", type=int, default=4, help="Parallel workers for per-record image scoring")
    parser.add_argument("--max-image-retries", type=int, default=1, help="Retries per image when processing fails")

    parser.add_argument("--quality-score-threshold", type=float, default=DEFAULT_QUALITY_SCORE_THRESHOLD, help="High-quality threshold")
    parser.add_argument("--min-text-length", type=int, default=DEFAULT_MIN_TEXT_LENGTH, help="Min OCR text length")
    parser.add_argument("--ocr-lang", type=str, default=DEFAULT_OCR_LANG, help="OCR language")
    parser.add_argument("--ocr-psm", type=int, default=6, help="Tesseract page segmentation mode")
    parser.add_argument("--ocr-oem", type=int, default=1, help="Tesseract OCR engine mode")
    parser.add_argument("--ocr-timeout", type=int, default=30, help="Tesseract timeout (seconds) per image, 0 disables")
    parser.add_argument("--eta-log-interval", type=int, default=30, help="Log ETA every N seconds for year progress")
    parser.add_argument("--image-cache-size", type=int, default=20000, help="Max in-memory cached image results")
    parser.add_argument(
        "--text-penalty-mode",
        type=str,
        default="hybrid",
        choices=["length", "signal", "hybrid"],
        help="Text penalty mode: length-only, signal-only, or length-dominant hybrid",
    )
    parser.add_argument("--text-length-midpoint", type=float, default=180.0, help="Midpoint for length-based text penalty")
    parser.add_argument("--text-length-steepness", type=float, default=0.02, help="Steepness for length-based text penalty")
    parser.add_argument(
        "--text-conf-adjust-weight",
        type=float,
        default=0.1,
        help="Confidence adjustment weight in hybrid mode; 0 keeps pure length penalty",
    )
    parser.add_argument("--text-hard-cap-length", type=int, default=400, help="Hard cap length threshold in hybrid mode")
    parser.add_argument("--text-hard-cap-weight", type=float, default=0.2, help="Max text weight when hard cap is triggered")

    parser.add_argument(
        "--quality-score-mode",
        type=str,
        default="hybrid",
        choices=["product", "weighted-average", "hybrid"],
        help="Quality score combination mode",
    )

    parser.add_argument("--clarity-summary-file", type=str, default="", help="Path to script5 summary JSON")
    parser.add_argument("--text-summary-file", type=str, default="", help="Path to script6 summary JSON")
    parser.add_argument("--auto-discover-summaries", dest="auto_discover_summaries", action="store_true", help="Auto discover latest script5/6 summary files")
    parser.add_argument("--no-auto-discover-summaries", dest="auto_discover_summaries", action="store_false", help="Disable automatic summary discovery")

    parser.add_argument("--summary-file", type=str, default=None, help="Summary JSON output file")
    parser.add_argument("--checkpoint-file", type=str, default=None, help="Checkpoint JSON file")
    parser.add_argument("--status-file", type=str, default=None, help="Status JSON file")
    parser.add_argument("--artifacts-dir", type=str, default=str(DEFAULT_ARTIFACTS_DIR), help="Runtime artifacts directory")

    parser.add_argument("--no-resume", action="store_true", help="Disable checkpoint resume")
    parser.add_argument("--reset-checkpoint", action="store_true", help="Delete checkpoint before run")
    parser.add_argument("--reprocess", action="store_true", help="Reprocess even if quality_processed=True")
    parser.add_argument("--reset-target-collections", action="store_true", help="Dangerous: clear year_2 before run")
    parser.add_argument(
        "--disable-article-dedup",
        action="store_true",
        help="Disable article-level dedup by title+content fingerprint (earliest date is kept by default)",
    )

    parser.set_defaults(auto_discover_summaries=True)
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    args.artifacts_dir = str(artifacts_dir)
    args.summary_file = resolve_artifact_path(args.summary_file, artifacts_dir, f"{SCRIPT_STEM}_summary.json")
    args.checkpoint_file = resolve_artifact_path(args.checkpoint_file, artifacts_dir, f"{SCRIPT_STEM}_checkpoint.json")
    args.status_file = resolve_artifact_path(args.status_file, artifacts_dir, f"{SCRIPT_STEM}_status.json")

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.checkpoint_every < 1:
        raise ValueError("--checkpoint-every must be >= 1")
    if args.image_workers < 1:
        raise ValueError("--image-workers must be >= 1")
    if args.max_image_retries < 0:
        raise ValueError("--max-image-retries must be >= 0")
    if args.min_text_length < 0:
        raise ValueError("--min-text-length must be >= 0")
    if args.ocr_timeout < 0:
        raise ValueError("--ocr-timeout must be >= 0")
    if args.eta_log_interval < 1:
        raise ValueError("--eta-log-interval must be >= 1")
    if args.image_cache_size < 0:
        raise ValueError("--image-cache-size must be >= 0")
    if args.quality_score_threshold < 0 or args.quality_score_threshold > 1:
        raise ValueError("--quality-score-threshold must be between 0 and 1")
    if args.text_conf_adjust_weight < 0 or args.text_conf_adjust_weight > 1:
        raise ValueError("--text-conf-adjust-weight must be between 0 and 1")
    if args.text_hard_cap_length < 0:
        raise ValueError("--text-hard-cap-length must be >= 0")
    if args.text_hard_cap_weight < 0 or args.text_hard_cap_weight > 1:
        raise ValueError("--text-hard-cap-weight must be between 0 and 1")
    if args.text_length_steepness <= 0:
        raise ValueError("--text-length-steepness must be > 0")

    return args


def build_years(args: argparse.Namespace) -> List[int]:
    env_years = os.environ.get("YEARS_TO_PROCESS", "").strip()
    if args.years.strip():
        return sorted({int(y.strip()) for y in args.years.split(",") if y.strip()})
    if env_years:
        return sorted({int(y.strip()) for y in env_years.split(",") if y.strip()})
    return list(range(args.start_year, args.end_year + 1))


def normalize_link(link: Any) -> str:
    return str(link or "").strip().lower()


def normalize_text(value: Any) -> str:
    text = str(value or "").strip()
    return re.sub(r"\s+", " ", text)


def parse_news_date(raw: Any, year: int) -> str:
    if isinstance(raw, datetime.datetime):
        return raw.strftime("%Y-%m-%d")
    if isinstance(raw, datetime.date):
        return raw.strftime("%Y-%m-%d")
    text = str(raw or "").strip()
    if text:
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "%Y年%m月%d日"):
            try:
                return datetime.datetime.strptime(text, fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue
    return f"{year}-01-01"


def article_fingerprint(doc: Dict[str, Any]) -> str:
    title = normalize_text(doc.get("title"))
    content = normalize_text(doc.get("content"))
    if not title and not content:
        return ""
    return f"{title}\n{content}"


def deduplicate_docs_keep_earliest(docs: List[Dict[str, Any]], year: int) -> Tuple[List[Dict[str, Any]], int]:
    if not docs:
        return docs, 0

    keep_map: Dict[str, Dict[str, Any]] = {}

    for doc in docs:
        doc["_news_date"] = doc.get("_news_date") or parse_news_date(doc.get("news_date"), year)
        doc["_link"] = doc.get("_link") or normalize_link(doc.get("link"))

        fp = article_fingerprint(doc)
        key = fp if fp else f"__id__::{doc.get('_id')}"
        prev = keep_map.get(key)

        if prev is None:
            keep_map[key] = doc
            continue

        prev_key = (str(prev.get("_news_date") or "9999-12-31"), str(prev.get("_id")))
        curr_key = (str(doc.get("_news_date") or "9999-12-31"), str(doc.get("_id")))
        if curr_key < prev_key:
            keep_map[key] = doc

    deduped = list(keep_map.values())
    removed = len(docs) - len(deduped)
    deduped.sort(key=lambda x: (x.get("_news_date", ""), x.get("_link", ""), str(x.get("_id", ""))))
    return deduped, removed


class WeightParams:
    def __init__(self, midpoint: float, steepness: float, min_weight: float, max_weight: float):
        self.midpoint = float(midpoint)
        self.steepness = float(steepness)
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)


def file_sha256(path: str) -> str:
    try:
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception:
        return ""


def discover_latest_summary(artifacts_dir: str, kind: str) -> str:
    base = Path(artifacts_dir)
    if not base.exists():
        return ""

    candidates: List[Path] = []
    for p in base.rglob("*_summary.json"):
        name = p.name.lower()
        if kind == "clarity" and "clarity" in name:
            candidates.append(p)
        if kind == "text" and "text" in name:
            candidates.append(p)

    if not candidates:
        return ""

    newest = max(candidates, key=lambda x: x.stat().st_mtime)
    return str(newest)


def load_summary_weight_params(path: str, default_params: WeightParams) -> Tuple[WeightParams, bool, Dict[str, Any]]:
    if not path:
        return default_params, False, {}
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        params = data.get("weight_params") or {}
        parsed = WeightParams(
            midpoint=params.get("midpoint", default_params.midpoint),
            steepness=params.get("steepness", default_params.steepness),
            min_weight=params.get("min_weight", default_params.min_weight),
            max_weight=params.get("max_weight", default_params.max_weight),
        )
        meta = {
            "path": path,
            "sha256": file_sha256(path),
            "run_id": data.get("run_id", ""),
            "run_finished_at": data.get("run_finished_at", ""),
        }
        return parsed, True, meta
    except Exception as e:
        logging.warning(f"Failed to load weight params from {path}: {e}")
        return default_params, False, {}


def calculate_clarity_features(image_path: str) -> Dict[str, float]:
    try:
        cv_image = cv2.imread(image_path)
        if cv_image is None:
            return {"blur": 0.0, "contrast": 0.0, "edge_density": 0.0, "clarity_signal": 0.0}

        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY) if len(cv_image.shape) == 3 else cv_image
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        contrast = float(np.std(gray))
        edges = cv2.Canny(gray, 100, 200)
        edge_density = float(np.count_nonzero(edges) / edges.size)
        clarity_signal = 0.6 * np.log1p(blur) + 0.3 * np.log1p(max(contrast, 0.0)) + 0.1 * (edge_density * 100.0)

        return {
            "blur": blur,
            "contrast": contrast,
            "edge_density": edge_density,
            "clarity_signal": float(clarity_signal),
        }
    except Exception:
        return {"blur": 0.0, "contrast": 0.0, "edge_density": 0.0, "clarity_signal": 0.0}


def extract_text_with_confidence(
    image_path: str,
    lang: str,
    psm: int,
    oem: int,
    timeout: int,
) -> Dict[str, Any]:
    try:
        config = f"--psm {psm} --oem {oem}"
        kwargs: Dict[str, Any] = {"lang": lang, "output_type": Output.DICT, "config": config}
        if timeout > 0:
            kwargs["timeout"] = timeout

        with Image.open(image_path) as img:
            data = pytesseract.image_to_data(img, **kwargs)

        text_tokens: List[str] = []
        conf_values: List[float] = []
        for raw_text, raw_conf in zip(data.get("text", []), data.get("conf", [])):
            token = str(raw_text or "").strip()
            if token:
                text_tokens.append(token)
            try:
                conf_val = float(raw_conf)
                if conf_val >= 0:
                    conf_values.append(conf_val)
            except Exception:
                continue

        cleaned = " ".join(text_tokens)
        avg_conf = float(np.mean(conf_values)) if conf_values else 0.0
        return {
            "text": cleaned,
            "text_length": len(cleaned),
            "ocr_confidence": avg_conf,
            "text_signal": len(cleaned) * (avg_conf / 100.0),
        }
    except Exception:
        return {
            "text": "",
            "text_length": 0,
            "ocr_confidence": 0.0,
            "text_signal": 0.0,
        }


def text_level_from_length(length: int) -> str:
    if length <= 0:
        return "none"
    if length < 50:
        return "low"
    if length < 200:
        return "medium"
    return "high"


def logistic_up(x: float, params: WeightParams) -> float:
    return float(
        params.min_weight
        + (params.max_weight - params.min_weight)
        / (1 + np.exp(-params.steepness * (x - params.midpoint)))
    )


def logistic_down(x: float, params: WeightParams) -> float:
    return float(
        params.max_weight
        - (params.max_weight - params.min_weight)
        / (1 + np.exp(-params.steepness * (x - params.midpoint)))
    )


class ImageQualityProcessor:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.client = MongoClient(args.mongo_uri)
        self.db = self.client[args.db_name]

        self.metrics: Dict[str, int] = {
            "records_attempted": 0,
            "records_completed": 0,
            "article_dedup_removed_docs": 0,
            "images_attempted": 0,
            "images_processed": 0,
            "images_cache_hits": 0,
            "images_missing": 0,
            "images_failed": 0,
            "high_quality_images": 0,
            "low_quality_images": 0,
            "images_retried": 0,
        }
        self.error_reason_counts: Dict[str, int] = {}
        self.image_cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.cache_lock = Lock()

        self.clarity_default = WeightParams(midpoint=5.0, steepness=0.08, min_weight=0.1, max_weight=1.0)
        self.text_default = WeightParams(midpoint=120.0, steepness=0.03, min_weight=0.1, max_weight=1.0)
        self.text_length_params = WeightParams(
            midpoint=args.text_length_midpoint,
            steepness=args.text_length_steepness,
            min_weight=self.text_default.min_weight,
            max_weight=self.text_default.max_weight,
        )

        if self.args.auto_discover_summaries:
            if not self.args.clarity_summary_file:
                discovered = discover_latest_summary(self.args.artifacts_dir, "clarity")
                if discovered:
                    self.args.clarity_summary_file = discovered
                    logging.info(f"Auto-discovered clarity summary: {discovered}")
            if not self.args.text_summary_file:
                discovered = discover_latest_summary(self.args.artifacts_dir, "text")
                if discovered:
                    self.args.text_summary_file = discovered
                    logging.info(f"Auto-discovered text summary: {discovered}")

        self.clarity_params, self.clarity_loaded, self.clarity_meta = load_summary_weight_params(
            args.clarity_summary_file, self.clarity_default
        )
        self.text_params, self.text_loaded, self.text_meta = load_summary_weight_params(
            args.text_summary_file, self.text_default
        )

    def add_error_reason(self, reason: str) -> None:
        self.error_reason_counts[reason] = self.error_reason_counts.get(reason, 0) + 1

    def cache_get(self, image_path: str) -> Optional[Dict[str, Any]]:
        if self.args.image_cache_size == 0:
            return None
        with self.cache_lock:
            cached = self.image_cache.get(image_path)
            if cached is None:
                return None
            self.image_cache.move_to_end(image_path)
            return dict(cached)

    def cache_set(self, image_path: str, data: Dict[str, Any]) -> None:
        if self.args.image_cache_size == 0:
            return
        with self.cache_lock:
            self.image_cache[image_path] = dict(data)
            self.image_cache.move_to_end(image_path)
            while len(self.image_cache) > self.args.image_cache_size:
                self.image_cache.popitem(last=False)

    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        path = self.args.checkpoint_file
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logging.warning(f"Failed to load checkpoint {path}: {e}")
            return None

    def save_checkpoint(self, data: Dict[str, Any]) -> None:
        path = self.args.checkpoint_file
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)

    def save_status(self, data: Dict[str, Any]) -> None:
        path = self.args.status_file
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)

    def get_query(self) -> Dict[str, Any]:
        if self.args.reprocess:
            return {"basic_filtered": True, "has_valid_images": True}
        return {
            "basic_filtered": True,
            "has_valid_images": True,
            "$or": [
                {"quality_processed": {"$exists": False}},
                {"quality_processed": {"$ne": True}},
            ],
        }

    def calculate_quality_score(self, clarity_weight: float, text_weight: float) -> float:
        if self.args.quality_score_mode == "product":
            return float(clarity_weight * text_weight)

        weighted = 0.6 * clarity_weight + 0.4 * text_weight
        if self.args.quality_score_mode == "weighted-average":
            return float(weighted)

        product = clarity_weight * text_weight
        return float(0.5 * product + 0.5 * weighted)

    def process_one_image(self, image_path: str) -> Dict[str, Any]:
        self.metrics["images_attempted"] += 1

        if not image_path or not os.path.exists(image_path):
            self.metrics["images_missing"] += 1
            self.add_error_reason("missing_file")
            return {"valid": False, "reason": "missing_file"}

        cached = self.cache_get(image_path)
        if cached is not None and cached.get("valid"):
            self.metrics["images_cache_hits"] += 1
            self.metrics["images_processed"] += 1
            if cached.get("is_high_quality"):
                self.metrics["high_quality_images"] += 1
            else:
                self.metrics["low_quality_images"] += 1
            return dict(cached)

        last_reason = "error:unknown"
        for attempt in range(self.args.max_image_retries + 1):
            if attempt > 0:
                self.metrics["images_retried"] += 1
            try:
                clarity = calculate_clarity_features(image_path)
                text = extract_text_with_confidence(
                    image_path,
                    self.args.ocr_lang,
                    self.args.ocr_psm,
                    self.args.ocr_oem,
                    self.args.ocr_timeout,
                )

                if text["text_length"] < self.args.min_text_length:
                    text_level = "none"
                else:
                    text_level = text_level_from_length(text["text_length"])

                clarity_weight = logistic_up(clarity["clarity_signal"], self.clarity_params)
                text_weight_signal = logistic_down(text["text_signal"], self.text_params)
                text_weight_length = logistic_down(float(text["text_length"]), self.text_length_params)

                if self.args.text_penalty_mode == "signal":
                    text_weight = text_weight_signal
                elif self.args.text_penalty_mode == "length":
                    text_weight = text_weight_length
                else:
                    # Hybrid keeps your original intent: long-text images are penalized first,
                    # OCR confidence only slightly adjusts the strength of that penalty.
                    conf_ratio = max(0.0, min(1.0, float(text["ocr_confidence"]) / 100.0))
                    adjust = (1.0 - self.args.text_conf_adjust_weight) + self.args.text_conf_adjust_weight * conf_ratio
                    text_weight = text_weight_length * adjust
                    text_weight = max(self.text_default.min_weight, min(self.text_default.max_weight, text_weight))
                    if self.args.text_hard_cap_length > 0 and int(text["text_length"]) >= self.args.text_hard_cap_length:
                        text_weight = min(text_weight, self.args.text_hard_cap_weight)

                quality_score = self.calculate_quality_score(clarity_weight, text_weight)
                is_high_quality = bool(quality_score >= self.args.quality_score_threshold)

                if is_high_quality:
                    self.metrics["high_quality_images"] += 1
                else:
                    self.metrics["low_quality_images"] += 1

                self.metrics["images_processed"] += 1

                result = {
                    "valid": True,
                    "blur_score": float(clarity["blur"]),
                    "contrast": float(clarity["contrast"]),
                    "edge_density": float(clarity["edge_density"]),
                    "clarity_signal": float(clarity["clarity_signal"]),
                    "clarity_weight": float(clarity_weight),
                    "text_length": int(text["text_length"]),
                    "ocr_confidence": float(text["ocr_confidence"]),
                    "text_signal": float(text["text_signal"]),
                    "text_weight_signal": float(text_weight_signal),
                    "text_weight_length": float(text_weight_length),
                    "text_weight": float(text_weight),
                    "text_level": text_level,
                    "quality_score": float(quality_score),
                    "is_high_quality": is_high_quality,
                }
                self.cache_set(image_path, result)
                return result
            except Exception as e:
                last_reason = f"error:{type(e).__name__}"

        self.metrics["images_failed"] += 1
        self.add_error_reason(last_reason)
        return {"valid": False, "reason": last_reason}

    def process_record(self, year: int, doc: Dict[str, Any]) -> Tuple[UpdateOne, UpdateOne, bool]:
        self.metrics["records_attempted"] += 1

        original_id = doc.get("original_id")
        link = normalize_link(doc.get("link"))
        news_date = parse_news_date(doc.get("news_date"), year)

        valid_images = doc.get("valid_images", [])
        processed_images: List[Dict[str, Any]] = []
        high_quality_images: List[Dict[str, Any]] = []
        seen_hashes = set()

        image_inputs = []
        seen_abs_paths = set()
        for img_info in valid_images:
            abs_path = str(img_info.get("abs_path") or "").strip()
            rel_path = str(img_info.get("path") or "").strip()
            file_hash = str(img_info.get("file_hash") or "").strip()

            if file_hash and file_hash in seen_hashes:
                continue
            if file_hash:
                seen_hashes.add(file_hash)
            if abs_path in seen_abs_paths:
                continue
            seen_abs_paths.add(abs_path)

            image_inputs.append((abs_path, rel_path, file_hash))

        if self.args.image_workers == 1 or len(image_inputs) <= 1:
            output_items = []
            for abs_path, rel_path, file_hash in image_inputs:
                output_items.append((abs_path, rel_path, file_hash, self.process_one_image(abs_path)))
        else:
            output_items = []
            with ThreadPoolExecutor(max_workers=self.args.image_workers) as executor:
                futures = {
                    executor.submit(self.process_one_image, abs_path): (abs_path, rel_path, file_hash)
                    for abs_path, rel_path, file_hash in image_inputs
                }
                for future in as_completed(futures):
                    abs_path, rel_path, file_hash = futures[future]
                    result = future.result()
                    output_items.append((abs_path, rel_path, file_hash, result))

        for abs_path, rel_path, file_hash, result in output_items:
            if not result.get("valid"):
                continue

            item = {
                "path": rel_path,
                "abs_path": abs_path,
                "file_hash": file_hash,
                "blur_score": result["blur_score"],
                "contrast": result["contrast"],
                "edge_density": result["edge_density"],
                "clarity_signal": result["clarity_signal"],
                "clarity_weight": result["clarity_weight"],
                "text_length": result["text_length"],
                "ocr_confidence": result["ocr_confidence"],
                "text_signal": result["text_signal"],
                "text_weight": result["text_weight"],
                "text_level": result["text_level"],
                "quality_score": result["quality_score"],
                "is_high_quality": result["is_high_quality"],
            }
            processed_images.append(item)
            if item["is_high_quality"]:
                high_quality_images.append(item)

        quality_scores = [x["quality_score"] for x in processed_images]
        clarity_weights = [x["clarity_weight"] for x in processed_images]
        text_weights = [x["text_weight"] for x in processed_images]

        unique_hashes = {x.get("file_hash", "") for x in processed_images if x.get("file_hash")}
        total_processed = len(processed_images)
        unique_ratio = float(len(unique_hashes) / total_processed) if total_processed > 0 else 0.0
        diversity_score = unique_ratio if unique_ratio > 0 else 1.0

        new_doc = {
            "original_id": original_id,
            "title": doc.get("title", ""),
            "content": doc.get("content", ""),
            "news_date": news_date,
            "publish_date": news_date,
            "category": doc.get("category", ""),
            "link": link,
            "processed_images": processed_images,
            "all_images": processed_images,
            "high_quality_images": high_quality_images,
            "has_high_quality_images": len(high_quality_images) > 0,
            "avg_clarity_weight": float(np.mean(clarity_weights)) if clarity_weights else 0.0,
            "avg_text_weight": float(np.mean(text_weights)) if text_weights else 0.0,
            "avg_quality_score": float(np.mean(quality_scores)) if quality_scores else 0.0,
            "diversity_score": float(diversity_score),
            "unique_ratio": float(unique_ratio),
            "similar_groups": max(0, total_processed - len(unique_hashes)),
            "quality_processed": True,
            "quality_processed_at": datetime.datetime.now(),
            "quality_score_mode": self.args.quality_score_mode,
            "quality_param_sources": {
                "clarity_summary_path": self.clarity_meta.get("path", ""),
                "clarity_summary_sha256": self.clarity_meta.get("sha256", ""),
                "clarity_summary_run_id": self.clarity_meta.get("run_id", ""),
                "text_summary_path": self.text_meta.get("path", ""),
                "text_summary_sha256": self.text_meta.get("sha256", ""),
                "text_summary_run_id": self.text_meta.get("run_id", ""),
            },
        }

        result_update = UpdateOne(
            {"link": link, "news_date": news_date},
            {"$set": new_doc},
            upsert=True,
        )

        filtered_update = UpdateOne(
            {"_id": doc["_id"]},
            {
                "$set": {
                    "quality_processed": True,
                    "quality_processed_at": datetime.datetime.now(),
                    "has_high_quality_images": len(high_quality_images) > 0,
                    "high_quality_image_count": len(high_quality_images),
                    "avg_quality_score": new_doc["avg_quality_score"],
                    "updated_at": datetime.datetime.now(),
                }
            },
            upsert=False,
        )

        self.metrics["records_completed"] += 1
        return result_update, filtered_update, len(high_quality_images) > 0

    def process_year(
        self,
        year: int,
        checkpoint_data: Dict[str, Any],
        status_data: Dict[str, Any],
        resume_key: Optional[Tuple[str, str]],
    ) -> Dict[str, Any]:
        filtered_collection_name = f"{year}_filtered"
        if filtered_collection_name not in self.db.list_collection_names():
            logging.warning(f"{filtered_collection_name} missing, skip")
            return {"year": year, "pending": 0, "high_quality_docs": 0}

        filtered_collection = self.db[filtered_collection_name]
        result_collection = self.db[f"{year}_2"]

        if self.args.reset_target_collections:
            result_collection.drop()
            logging.warning(f"Dropped target collection due to --reset-target-collections: {year}_2")

        query = self.get_query()
        projection = {
            "_id": 1,
            "original_id": 1,
            "title": 1,
            "content": 1,
            "news_date": 1,
            "category": 1,
            "link": 1,
            "valid_images": 1,
            "quality_processed": 1,
        }

        docs = list(filtered_collection.find(query, projection))

        for d in docs:
            d["_news_date"] = parse_news_date(d.get("news_date"), year)
            d["_link"] = normalize_link(d.get("link"))

        docs.sort(key=lambda x: (x.get("_news_date", ""), x.get("_link", ""), str(x.get("_id", ""))))

        if not self.args.disable_article_dedup:
            before = len(docs)
            docs, removed = deduplicate_docs_keep_earliest(docs, year)
            self.metrics["article_dedup_removed_docs"] += removed
            if removed > 0:
                logging.info(
                    "Year %d article dedup removed %d docs (from %d to %d)",
                    year,
                    removed,
                    before,
                    len(docs),
                )

        if self.args.max_items_per_year > 0:
            docs = docs[: self.args.max_items_per_year]

        if resume_key:
            resume_news_date, resume_link = resume_key
            docs = [
                d
                for d in docs
                if (d.get("_news_date", ""), d.get("_link", "")) > (resume_news_date, resume_link)
            ]

        total_pending = len(docs)
        if total_pending == 0:
            logging.info(f"Year {year}: nothing pending")
            return {"year": year, "pending": 0, "high_quality_docs": 0}

        result_updates: List[UpdateOne] = []
        filtered_updates: List[UpdateOne] = []
        high_quality_docs = 0
        processed_in_year = 0
        year_started = datetime.datetime.now()
        last_eta_log = year_started

        with tqdm(total=total_pending, desc=f"Year {year} quality", unit="doc") as pbar:
            for doc in docs:
                result_update, filtered_update, has_high_quality = self.process_record(year, doc)
                result_updates.append(result_update)
                filtered_updates.append(filtered_update)
                processed_in_year += 1
                pbar.update(1)

                if has_high_quality:
                    high_quality_docs += 1

                checkpoint_data["current_year"] = year
                checkpoint_data["last_completed_key"] = {
                    "news_date": doc.get("_news_date", f"{year}-01-01"),
                    "link": doc.get("_link", ""),
                }
                checkpoint_data["processed_items"] = checkpoint_data.get("processed_items", 0) + 1
                checkpoint_data["updated_at"] = datetime.datetime.now().isoformat()

                if len(result_updates) >= self.args.batch_size:
                    result_collection.bulk_write(result_updates, ordered=False)
                    filtered_collection.bulk_write(filtered_updates, ordered=False)
                    result_updates = []
                    filtered_updates = []

                if processed_in_year % self.args.checkpoint_every == 0:
                    self.save_checkpoint(checkpoint_data)
                    status_data["current_year"] = year
                    status_data["processed_items"] = checkpoint_data.get("processed_items", 0)
                    status_data["updated_at"] = datetime.datetime.now().isoformat()
                    status_data["last_error"] = None
                    self.save_status(status_data)

                now = datetime.datetime.now()
                if (now - last_eta_log).total_seconds() >= self.args.eta_log_interval or processed_in_year == total_pending:
                    elapsed = (now - year_started).total_seconds()
                    rate = processed_in_year / max(elapsed, 1e-6)
                    remaining = max(total_pending - processed_in_year, 0)
                    eta_seconds = remaining / max(rate, 1e-6)
                    logging.info(
                        "[ETA][quality][year=%d] %d/%d (%.2f%%) | rate=%.2f doc/s | elapsed=%s | eta=%s",
                        year,
                        processed_in_year,
                        total_pending,
                        (processed_in_year / total_pending) * 100.0,
                        rate,
                        format_eta(elapsed),
                        format_eta(eta_seconds),
                    )
                    last_eta_log = now

        if result_updates:
            result_collection.bulk_write(result_updates, ordered=False)
            filtered_collection.bulk_write(filtered_updates, ordered=False)

        self.save_checkpoint(checkpoint_data)
        return {"year": year, "pending": total_pending, "high_quality_docs": high_quality_docs}

    def run(self) -> Dict[str, Any]:
        run_started_at = datetime.datetime.now()
        run_id = run_started_at.strftime("%Y%m%d_%H%M%S")

        years = build_years(self.args)
        if not years:
            raise ValueError("No years to process")

        if self.args.reset_checkpoint and os.path.exists(self.args.checkpoint_file):
            os.remove(self.args.checkpoint_file)
            logging.info(f"Reset checkpoint file: {self.args.checkpoint_file}")

        checkpoint = self.load_checkpoint() if not self.args.no_resume else None

        checkpoint_data: Dict[str, Any] = {
            "run_id": run_id,
            "status": "in_progress",
            "db_name": self.args.db_name,
            "years": years,
            "summary_file": self.args.summary_file,
            "checkpoint_file": self.args.checkpoint_file,
            "status_file": self.args.status_file,
            "current_year": years[0],
            "last_completed_key": None,
            "processed_items": 0,
            "updated_at": datetime.datetime.now().isoformat(),
        }

        status_data: Dict[str, Any] = {
            "run_id": run_id,
            "status": "running",
            "pid": os.getpid(),
            "db_name": self.args.db_name,
            "years": years,
            "year_range": f"{years[0]}-{years[-1]}",
            "current_year": years[0],
            "processed_items": 0,
            "updated_at": datetime.datetime.now().isoformat(),
            "files": {
                "checkpoint_file": self.args.checkpoint_file,
                "summary_file": self.args.summary_file,
                "status_file": self.args.status_file,
            },
            "last_error": None,
        }
        self.save_status(status_data)

        resumed = False
        resume_year = years[0]
        resume_key: Optional[Tuple[str, str]] = None

        if checkpoint:
            same_job = (
                checkpoint.get("db_name") == self.args.db_name
                and checkpoint.get("years") == years
                and checkpoint.get("status") in {"in_progress", "completed"}
            )
            if same_job:
                resumed = True
                resume_year = int(checkpoint.get("current_year", years[0]))
                key_obj = checkpoint.get("last_completed_key") or {}
                news_date = key_obj.get("news_date")
                link = key_obj.get("link")
                if news_date is not None and link is not None:
                    resume_key = (news_date, link)
                checkpoint_data["processed_items"] = int(checkpoint.get("processed_items", 0))
                logging.info(
                    f"Resume enabled: year={resume_year}, key={resume_key}, processed_items={checkpoint_data['processed_items']}"
                )

        year_stats: List[Dict[str, Any]] = []
        try:
            for year in years:
                if year < resume_year:
                    continue

                use_resume_key = resume_key if year == resume_year else None
                stats = self.process_year(year, checkpoint_data, status_data, use_resume_key)
                year_stats.append(stats)

                checkpoint_data["current_year"] = year + 1
                checkpoint_data["last_completed_key"] = None
                checkpoint_data["updated_at"] = datetime.datetime.now().isoformat()
                self.save_checkpoint(checkpoint_data)

                status_data["current_year"] = year
                status_data["processed_items"] = checkpoint_data.get("processed_items", 0)
                status_data["updated_at"] = datetime.datetime.now().isoformat()
                status_data["last_error"] = None
                self.save_status(status_data)

        except KeyboardInterrupt:
            logging.warning("Interrupted by user; checkpoint saved for resume")
            self.save_checkpoint(checkpoint_data)
            status_data["status"] = "interrupted"
            status_data["updated_at"] = datetime.datetime.now().isoformat()
            self.save_status(status_data)
            raise
        except Exception as e:
            checkpoint_data["status"] = "failed"
            checkpoint_data["updated_at"] = datetime.datetime.now().isoformat()
            self.save_checkpoint(checkpoint_data)
            status_data["status"] = "failed"
            status_data["last_error"] = str(e)
            status_data["updated_at"] = datetime.datetime.now().isoformat()
            self.save_status(status_data)
            raise
        finally:
            self.client.close()

        checkpoint_data["status"] = "completed"
        checkpoint_data["updated_at"] = datetime.datetime.now().isoformat()
        self.save_checkpoint(checkpoint_data)

        status_data["status"] = "completed"
        status_data["current_year"] = years[-1]
        status_data["processed_items"] = checkpoint_data.get("processed_items", 0)
        status_data["updated_at"] = datetime.datetime.now().isoformat()
        status_data["last_error"] = None
        self.save_status(status_data)

        run_finished_at = datetime.datetime.now()

        summary = {
            "run_id": run_id,
            "run_started_at": run_started_at.isoformat(),
            "run_finished_at": run_finished_at.isoformat(),
            "elapsed_seconds": (run_finished_at - run_started_at).total_seconds(),
            "config": {
                "db_name": self.args.db_name,
                "years": years,
                "batch_size": self.args.batch_size,
                "checkpoint_every": self.args.checkpoint_every,
                "quality_score_threshold": self.args.quality_score_threshold,
                "min_text_length": self.args.min_text_length,
                "ocr_lang": self.args.ocr_lang,
                "ocr_psm": self.args.ocr_psm,
                "ocr_oem": self.args.ocr_oem,
                "ocr_timeout": self.args.ocr_timeout,
                "quality_score_mode": self.args.quality_score_mode,
                "clarity_summary_file": self.args.clarity_summary_file,
                "text_summary_file": self.args.text_summary_file,
                "text_penalty_mode": self.args.text_penalty_mode,
                "text_length_midpoint": self.args.text_length_midpoint,
                "text_length_steepness": self.args.text_length_steepness,
                "text_conf_adjust_weight": self.args.text_conf_adjust_weight,
                "text_hard_cap_length": self.args.text_hard_cap_length,
                "text_hard_cap_weight": self.args.text_hard_cap_weight,
                "clarity_params_loaded": self.clarity_loaded,
                "text_params_loaded": self.text_loaded,
                "clarity_weight_params": self.clarity_params.__dict__,
                "text_weight_params": self.text_params.__dict__,
                "text_length_weight_params": self.text_length_params.__dict__,
                "resumed_from_checkpoint": resumed,
                "max_items_per_year": self.args.max_items_per_year,
                "image_workers": self.args.image_workers,
                "image_cache_size": self.args.image_cache_size,
                "max_image_retries": self.args.max_image_retries,
                "auto_discover_summaries": self.args.auto_discover_summaries,
                "reprocess": self.args.reprocess,
            },
            "metrics": self.metrics,
            "error_reason_counts": self.error_reason_counts,
            "parameter_sources": {
                "clarity": self.clarity_meta,
                "text": self.text_meta,
            },
            "year_stats": year_stats,
        }

        with open(self.args.summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        logging.info(f"Summary written to {self.args.summary_file}")
        return summary


def main() -> None:
    args = parse_args()
    start = time.time()

    logging.info("=== Starting Image Quality Processor ===")
    logging.info(f"Database: {args.db_name}")
    logging.info(f"Years: {build_years(args)}")
    logging.info(f"Quality score mode: {args.quality_score_mode}")

    processor = ImageQualityProcessor(args)
    processor.run()

    elapsed = time.time() - start
    logging.info(f"=== Image Quality Processor Complete, elapsed: {elapsed:.2f}s ===")


if __name__ == "__main__":
    main()
