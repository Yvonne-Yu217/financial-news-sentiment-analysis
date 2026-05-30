import argparse
import datetime
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from pymongo import MongoClient, UpdateOne
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

START_YEAR = 2014
END_YEAR = 2026
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "sina_news_dataset_test"
SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPT_STEM = Path(__file__).stem
DEFAULT_ARTIFACTS_DIR = SCRIPT_DIR / "run_artifacts"

# Basic filter thresholds
ASPECT_RATIO_THRESHOLD = 3.0
MIN_IMAGE_SIZE = 150
MAX_IMAGE_SIZE = 4000
MIN_CONTENT_AREA = 0.15
MIN_FILE_BYTES = 1024

VALID_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp")


def resolve_artifact_path(path_arg: Optional[str], artifacts_dir: Path, default_name: str) -> str:
    if not path_arg:
        return str(artifacts_dir / default_name)

    user_path = Path(path_arg)
    if user_path.is_absolute() or user_path.parent != Path("."):
        user_path.parent.mkdir(parents=True, exist_ok=True)
        return str(user_path)

    return str(artifacts_dir / user_path.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run basic image filtering for stage-3 mapping outputs")
    parser.add_argument("--mongo-uri", type=str, default=MONGO_URI, help="MongoDB URI")
    parser.add_argument("--db-name", type=str, default=DB_NAME, help="MongoDB database name")
    parser.add_argument("--start-year", type=int, default=START_YEAR, help="Start year")
    parser.add_argument("--end-year", type=int, default=END_YEAR, help="End year")
    parser.add_argument(
        "--years",
        type=str,
        default="",
        help="Optional comma-separated years. Overrides --start-year/--end-year, e.g. 2020,2021,2022",
    )
    parser.add_argument("--checkpoint-every", type=int, default=100, help="Save checkpoint/status every N mappings")
    parser.add_argument("--batch-size", type=int, default=200, help="MongoDB bulk update batch size")
    parser.add_argument("--max-items-per-year", type=int, default=0, help="Limit mappings per year for testing; 0 means no limit")

    parser.add_argument("--aspect-ratio-threshold", type=float, default=ASPECT_RATIO_THRESHOLD, help="Max allowed image aspect ratio")
    parser.add_argument("--min-image-size", type=int, default=MIN_IMAGE_SIZE, help="Min width/height in pixels")
    parser.add_argument("--max-image-size", type=int, default=MAX_IMAGE_SIZE, help="Max width/height in pixels")
    parser.add_argument("--min-content-area", type=float, default=MIN_CONTENT_AREA, help="Min foreground content ratio")
    parser.add_argument("--min-file-bytes", type=int, default=MIN_FILE_BYTES, help="Reject files smaller than this byte size")
    parser.add_argument("--disable-content-area-filter", action="store_true", help="Disable content area filtering")
    parser.add_argument("--disable-qr-filter", action="store_true", help="Disable QR code filtering")
    parser.add_argument("--disable-dedup", action="store_true", help="Disable duplicate image filtering by file hash")

    parser.add_argument("--summary-file", type=str, default=None, help="Summary JSON output file")
    parser.add_argument("--checkpoint-file", type=str, default=None, help="Checkpoint JSON file")
    parser.add_argument("--status-file", type=str, default=None, help="Status JSON file")
    parser.add_argument("--artifacts-dir", type=str, default=str(DEFAULT_ARTIFACTS_DIR), help="Directory for runtime artifacts")

    parser.add_argument("--no-resume", action="store_true", help="Disable checkpoint resume")
    parser.add_argument("--reset-checkpoint", action="store_true", help="Delete existing checkpoint before run")
    parser.add_argument(
        "--reprocess",
        action="store_true",
        help="Reprocess records even if mapping.basic_filtered=True",
    )

    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    args.artifacts_dir = str(artifacts_dir)
    args.summary_file = resolve_artifact_path(args.summary_file, artifacts_dir, f"{SCRIPT_STEM}_summary.json")
    args.checkpoint_file = resolve_artifact_path(args.checkpoint_file, artifacts_dir, f"{SCRIPT_STEM}_checkpoint.json")
    args.status_file = resolve_artifact_path(args.status_file, artifacts_dir, f"{SCRIPT_STEM}_status.json")

    if args.checkpoint_every < 1:
        raise ValueError("--checkpoint-every must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.min_image_size < 1 or args.max_image_size < 1:
        raise ValueError("--min-image-size/--max-image-size must be >= 1")
    if args.min_image_size > args.max_image_size:
        raise ValueError("--min-image-size must be <= --max-image-size")
    if args.min_content_area < 0 or args.min_content_area > 1:
        raise ValueError("--min-content-area must be between 0 and 1")
    if args.aspect_ratio_threshold < 1:
        raise ValueError("--aspect-ratio-threshold must be >= 1")
    if args.min_file_bytes < 0:
        raise ValueError("--min-file-bytes must be >= 0")

    return args


def normalize_link(link: Any) -> str:
    if not link:
        return ""
    return str(link).strip().lower()


def parse_news_date(raw: Any, year: int) -> str:
    if isinstance(raw, datetime.datetime):
        return raw.strftime("%Y-%m-%d")
    if isinstance(raw, datetime.date):
        return raw.strftime("%Y-%m-%d")
    if isinstance(raw, str) and raw.strip():
        text = raw.strip()
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "%Y年%m月%d日"):
            try:
                return datetime.datetime.strptime(text, fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue
    return f"{year}-01-01"


def build_years(args: argparse.Namespace) -> List[int]:
    env_years = os.environ.get("YEARS_TO_PROCESS", "").strip()
    if args.years.strip():
        years = [int(y.strip()) for y in args.years.split(",") if y.strip()]
        logging.info(f"Using --years from CLI: {years}")
        return sorted(set(years))
    if env_years:
        years = [int(y.strip()) for y in env_years.split(",") if y.strip()]
        logging.info(f"Using YEARS_TO_PROCESS from env: {years}")
        return sorted(set(years))
    years = list(range(args.start_year, args.end_year + 1))
    logging.info(f"Using year range from args: {years}")
    return years


class ImageBasicFilter:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.qr_detector = cv2.QRCodeDetector()
        self.metrics: Dict[str, int] = {
            "total_images": 0,
            "invalid_images": 0,
            "small_file_rejected": 0,
            "size_ratio_rejected": 0,
            "qr_code_images": 0,
            "content_area_rejected": 0,
            "duplicate_images": 0,
            "passed_images": 0,
            "folders_missing": 0,
            "mappings_attempted": 0,
            "mappings_completed": 0,
        }

    def compute_file_hash(self, image_path: str) -> str:
        try:
            hasher = hashlib.md5()
            with open(image_path, "rb") as f:
                while True:
                    chunk = f.read(8192)
                    if not chunk:
                        break
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception:
            return ""

    def is_valid_image(self, image_path: str) -> Tuple[bool, Optional[Image.Image]]:
        try:
            img = Image.open(image_path)
            img.verify()
            img = Image.open(image_path)
            if img.size[0] == 0 or img.size[1] == 0:
                return False, None
            return True, img
        except Exception:
            return False, None

    def check_image_size(self, img: Image.Image) -> bool:
        width, height = img.size
        if width < self.args.min_image_size or height < self.args.min_image_size:
            return False
        if width > self.args.max_image_size or height > self.args.max_image_size:
            return False
        aspect_ratio = max(width, height) / max(1, min(width, height))
        if aspect_ratio > self.args.aspect_ratio_threshold:
            return False
        return True

    def check_content_area(self, cv_image: np.ndarray) -> bool:
        try:
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
            content_ratio = np.count_nonzero(binary) / float(binary.shape[0] * binary.shape[1])
            return content_ratio >= self.args.min_content_area
        except Exception:
            return True

    def has_qr_code(self, cv_image: np.ndarray) -> bool:
        try:
            _, bbox, _ = self.qr_detector.detectAndDecode(cv_image)
            return bbox is not None and len(bbox) > 0
        except Exception:
            return False

    def process_single_image(self, image_path: str) -> Dict[str, Any]:
        self.metrics["total_images"] += 1

        try:
            size_in_bytes = os.path.getsize(image_path)
            if size_in_bytes < self.args.min_file_bytes:
                self.metrics["small_file_rejected"] += 1
                return {"valid": False, "reject_reason": "small_file"}
        except Exception:
            self.metrics["invalid_images"] += 1
            return {"valid": False, "reject_reason": "stat_failed"}

        is_valid, pil_image = self.is_valid_image(image_path)
        if not is_valid:
            self.metrics["invalid_images"] += 1
            return {"valid": False, "reject_reason": "invalid_image"}

        try:
            if not self.check_image_size(pil_image):
                self.metrics["size_ratio_rejected"] += 1
                pil_image.close()
                return {"valid": False, "reject_reason": "size_ratio"}

            cv_image = cv2.imread(image_path)
            if cv_image is None:
                self.metrics["invalid_images"] += 1
                pil_image.close()
                return {"valid": False, "reject_reason": "opencv_read_failed"}

            if (not self.args.disable_content_area_filter) and (not self.check_content_area(cv_image)):
                self.metrics["content_area_rejected"] += 1
                pil_image.close()
                return {"valid": False, "reject_reason": "content_area"}

            if (not self.args.disable_qr_filter) and self.has_qr_code(cv_image):
                self.metrics["qr_code_images"] += 1
                pil_image.close()
                return {"valid": False, "reject_reason": "qr_code"}

            self.metrics["passed_images"] += 1
            pil_image.close()
            return {"valid": True, "path": image_path}

        except Exception as e:
            if pil_image:
                pil_image.close()
            self.metrics["invalid_images"] += 1
            return {"valid": False, "reject_reason": "error", "error": str(e)}


class ImageBasicFilterRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.filter = ImageBasicFilter(args)
        self.client = MongoClient(args.mongo_uri)
        self.db = self.client[args.db_name]

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

    def resolve_folder_path(self, folder_path: str) -> Path:
        p = Path(folder_path).expanduser()
        if p.is_absolute():
            return p
        return (SCRIPT_DIR / p).resolve()

    def get_mapping_query(self) -> Dict[str, Any]:
        if self.args.reprocess:
            return {
                "has_images": True,
                "folder_path": {"$exists": True, "$ne": ""},
            }
        return {
            "has_images": True,
            "folder_path": {"$exists": True, "$ne": ""},
            "$or": [
                {"basic_filtered": {"$exists": False}},
                {"basic_filtered": {"$ne": True}},
            ],
        }

    def process_mapping_record(self, year: int, mapping: Dict[str, Any]) -> Tuple[UpdateOne, UpdateOne, int]:
        self.filter.metrics["mappings_attempted"] += 1

        folder_path = mapping.get("folder_path", "")
        abs_folder = self.resolve_folder_path(folder_path)
        original_id = mapping.get("original_id")
        link = normalize_link(mapping.get("link"))
        news_date = parse_news_date(mapping.get("news_date"), year)

        valid_images: List[Dict[str, str]] = []
        rejected_images: List[Dict[str, str]] = []
        seen_hashes = set()

        if not folder_path or not abs_folder.exists():
            self.filter.metrics["folders_missing"] += 1
            rejected_images.append({"path": "", "reason": "folder_not_found"})
        else:
            for root, _, files in os.walk(abs_folder):
                for file_name in files:
                    if file_name.startswith("._"):
                        continue
                    if not file_name.lower().endswith(VALID_IMAGE_SUFFIXES):
                        continue
                    image_path = os.path.join(root, file_name)
                    rel_path = os.path.relpath(image_path, abs_folder)

                    result = self.filter.process_single_image(image_path)
                    if result["valid"]:
                        file_hash = self.filter.compute_file_hash(image_path)
                        if (not self.args.disable_dedup) and file_hash:
                            if file_hash in seen_hashes:
                                self.filter.metrics["duplicate_images"] += 1
                                rejected_images.append({
                                    "path": rel_path,
                                    "reason": "duplicate_image",
                                })
                                continue
                            seen_hashes.add(file_hash)
                        valid_images.append({
                            "path": rel_path,
                            "abs_path": image_path,
                            "file_hash": file_hash,
                        })
                    else:
                        rejected_images.append({
                            "path": rel_path,
                            "reason": result.get("reject_reason", "unknown"),
                        })

        source_collection = self.db[f"{year}_1"]
        source_doc = source_collection.find_one(
            {"link": link, "news_date": news_date},
            {"title": 1, "content": 1, "category": 1, "news_date": 1, "link": 1},
        )

        final_title = mapping.get("title") or (source_doc.get("title") if source_doc else "") or ""
        final_content = (source_doc.get("content") if source_doc else "") or ""
        final_category = (source_doc.get("category") if source_doc else "") or ""

        has_valid_images = len(valid_images) > 0

        filtered_doc = {
            "original_id": original_id,
            "title": final_title,
            "content": final_content,
            "news_date": news_date,
            "category": final_category,
            "link": link,
            "folder_path": str(abs_folder),
            "valid_images": valid_images,
            "rejected_images": rejected_images,
            "has_valid_images": has_valid_images,
            "basic_filtered": True,
            "quality_processed": False,
            "basic_filtered_at": datetime.datetime.now(),
            "year": year,
        }

        filtered_update = UpdateOne(
            {"link": link, "news_date": news_date},
            {"$set": filtered_doc},
            upsert=True,
        )

        mapping_update = UpdateOne(
            {"_id": mapping["_id"]},
            {
                "$set": {
                    "basic_filtered": True,
                    "has_valid_images": has_valid_images,
                    "valid_image_count": len(valid_images),
                    "quality_processed": False,
                    "basic_filtered_at": datetime.datetime.now(),
                    "updated_at": datetime.datetime.now(),
                }
            },
            upsert=False,
        )

        self.filter.metrics["mappings_completed"] += 1
        return filtered_update, mapping_update, len(valid_images)

    def process_year(
        self,
        year: int,
        checkpoint_data: Dict[str, Any],
        status_data: Dict[str, Any],
        resume_key: Optional[Tuple[str, str]],
    ) -> Dict[str, Any]:
        mapping_collection_name = f"{year}_mapping"
        if mapping_collection_name not in self.db.list_collection_names():
            logging.warning(f"{mapping_collection_name} missing, skip")
            return {
                "year": year,
                "source_total": 0,
                "pending": 0,
                "has_valid_images": 0,
            }

        mapping_collection = self.db[mapping_collection_name]
        filtered_collection = self.db[f"{year}_filtered"]

        query = self.get_mapping_query()
        projection = {
            "_id": 1,
            "original_id": 1,
            "title": 1,
            "news_date": 1,
            "link": 1,
            "folder_path": 1,
            "has_images": 1,
            "basic_filtered": 1,
        }

        mappings = list(mapping_collection.find(query, projection))
        if self.args.max_items_per_year > 0:
            mappings = mappings[: self.args.max_items_per_year]

        if not mappings:
            logging.info(f"Year {year}: no mappings to process")
            return {
                "year": year,
                "source_total": 0,
                "pending": 0,
                "has_valid_images": 0,
            }

        for m in mappings:
            m["_news_date"] = parse_news_date(m.get("news_date"), year)
            m["_link"] = normalize_link(m.get("link"))

        mappings.sort(key=lambda x: (x.get("_news_date", ""), x.get("_link", ""), str(x.get("_id", ""))))

        if resume_key:
            resume_news_date, resume_link = resume_key
            mappings = [
                m
                for m in mappings
                if (m.get("_news_date", ""), m.get("_link", "")) > (resume_news_date, resume_link)
            ]

        total_pending = len(mappings)
        if total_pending == 0:
            logging.info(f"Year {year}: nothing pending after resume")
            return {
                "year": year,
                "source_total": 0,
                "pending": 0,
                "has_valid_images": 0,
            }

        logging.info(f"Year {year}: pending mappings={total_pending}")

        filtered_updates: List[UpdateOne] = []
        mapping_updates: List[UpdateOne] = []

        year_has_valid = 0
        processed_in_year = 0

        with tqdm(total=total_pending, desc=f"Year {year} basic_filter", unit="mapping") as pbar:
            for mapping in mappings:
                filtered_update, mapping_update, valid_count = self.process_mapping_record(year, mapping)
                filtered_updates.append(filtered_update)
                mapping_updates.append(mapping_update)
                processed_in_year += 1
                if valid_count > 0:
                    year_has_valid += 1
                pbar.update(1)

                checkpoint_data["current_year"] = year
                checkpoint_data["last_completed_key"] = {
                    "news_date": mapping.get("_news_date", f"{year}-01-01"),
                    "link": mapping.get("_link", ""),
                }
                checkpoint_data["processed_items"] = checkpoint_data.get("processed_items", 0) + 1
                checkpoint_data["updated_at"] = datetime.datetime.now().isoformat()

                if len(filtered_updates) >= self.args.batch_size:
                    filtered_collection.bulk_write(filtered_updates, ordered=False)
                    mapping_collection.bulk_write(mapping_updates, ordered=False)
                    filtered_updates = []
                    mapping_updates = []

                if processed_in_year % self.args.checkpoint_every == 0:
                    self.save_checkpoint(checkpoint_data)
                    status_data["current_year"] = year
                    status_data["processed_items"] = checkpoint_data.get("processed_items", 0)
                    status_data["updated_at"] = datetime.datetime.now().isoformat()
                    status_data["last_error"] = None
                    self.save_status(status_data)

        if filtered_updates:
            filtered_collection.bulk_write(filtered_updates, ordered=False)
            mapping_collection.bulk_write(mapping_updates, ordered=False)

        self.save_checkpoint(checkpoint_data)

        return {
            "year": year,
            "source_total": total_pending,
            "pending": total_pending,
            "has_valid_images": year_has_valid,
        }

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
            "year_range": f"{years[0]}-{years[-1]}",
            "years": years,
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
                key_news_date = key_obj.get("news_date")
                key_link = key_obj.get("link")
                if key_news_date is not None and key_link is not None:
                    resume_key = (key_news_date, key_link)
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
                "min_image_size": self.args.min_image_size,
                "max_image_size": self.args.max_image_size,
                "aspect_ratio_threshold": self.args.aspect_ratio_threshold,
                "min_content_area": self.args.min_content_area,
                "min_file_bytes": self.args.min_file_bytes,
                "disable_content_area_filter": self.args.disable_content_area_filter,
                "disable_qr_filter": self.args.disable_qr_filter,
                "disable_dedup": self.args.disable_dedup,
                "checkpoint_file": self.args.checkpoint_file,
                "status_file": self.args.status_file,
                "summary_file": self.args.summary_file,
                "resumed_from_checkpoint": resumed,
                "max_items_per_year": self.args.max_items_per_year,
                "reprocess": self.args.reprocess,
            },
            "metrics": self.filter.metrics,
            "year_stats": year_stats,
        }

        with open(self.args.summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        logging.info(f"Summary written to {self.args.summary_file}")
        return summary


def main() -> None:
    args = parse_args()
    start = time.time()

    logging.info("=== Starting Basic Image Filter ===")
    logging.info(f"Database: {args.db_name}")
    logging.info(f"Years: {build_years(args)}")
    logging.info(f"Batch size: {args.batch_size}")

    runner = ImageBasicFilterRunner(args)
    runner.run()

    elapsed = time.time() - start
    logging.info(f"=== Basic Image Filter Complete, elapsed: {elapsed:.2f}s ===")


if __name__ == "__main__":
    main()
