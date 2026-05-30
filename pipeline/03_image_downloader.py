import argparse
import asyncio
import datetime
import hashlib
import json
import logging
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import aiohttp
from pymongo import ASCENDING, MongoClient, UpdateOne
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("image_downloader.log"),
        logging.StreamHandler(),
    ],
)

START_YEAR = 2014
END_YEAR = 2026
BATCH_SIZE = 100
CONCURRENT_REQUESTS = 48
MAX_RETRIES = 2
HTTP_TIMEOUT = 20
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "sina_news_dataset_test"
SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPT_STEM = Path(__file__).stem
DEFAULT_ARTIFACTS_DIR = SCRIPT_DIR / "run_artifacts"
DEFAULT_IMAGE_ROOT = SCRIPT_DIR / "images"

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
]

SKIP_KEYWORDS = {
    "icon",
    "logo",
    "button",
    "banner",
    "avatar",
    "loading",
    "background",
    "ad_",
    "adv_",
    "advertisement",
    "favicon",
    "footer",
    "header",
    "sidebar",
    "thumbnail",
    "placeholder",
}

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}


def resolve_artifact_path(path_arg: Optional[str], artifacts_dir: Path, default_name: str) -> str:
    if not path_arg:
        return str(artifacts_dir / default_name)

    user_path = Path(path_arg)
    if user_path.is_absolute() or user_path.parent != Path("."):
        user_path.parent.mkdir(parents=True, exist_ok=True)
        return str(user_path)

    return str(artifacts_dir / user_path.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download article images from year collections")
    parser.add_argument("--mongo-uri", type=str, default=MONGO_URI, help="MongoDB URI")
    parser.add_argument("--db-name", type=str, default=DB_NAME, help="MongoDB database name")
    parser.add_argument("--start-year", type=int, default=START_YEAR, help="Start year")
    parser.add_argument("--end-year", type=int, default=END_YEAR, help="End year")
    parser.add_argument("--start-date", type=str, default=None, help="Optional start date filter (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None, help="Optional end date filter (YYYY-MM-DD)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="MongoDB bulk update batch size")
    parser.add_argument("--concurrent", type=int, default=CONCURRENT_REQUESTS, help="Concurrent HTTP requests")
    parser.add_argument("--inflight-multiplier", type=int, default=8, help="Queued tasks = concurrent * inflight-multiplier")
    parser.add_argument("--max-retries", type=int, default=MAX_RETRIES, help="Max retries per image URL")
    parser.add_argument("--timeout", type=int, default=HTTP_TIMEOUT, help="HTTP timeout seconds")
    parser.add_argument("--delay-min", type=float, default=0.0, help="Random delay lower bound per image request")
    parser.add_argument("--delay-max", type=float, default=0.03, help="Random delay upper bound per image request")
    parser.add_argument("--checkpoint-every", type=int, default=100, help="Save checkpoint/status every N processed articles")
    parser.add_argument("--image-root", type=str, default=str(DEFAULT_IMAGE_ROOT), help="Root folder to store downloaded images")
    parser.add_argument("--summary-file", type=str, default=None, help="Summary JSON output file")
    parser.add_argument("--checkpoint-file", type=str, default=None, help="Checkpoint JSON file")
    parser.add_argument("--status-file", type=str, default=None, help="Status JSON file")
    parser.add_argument("--failed-urls-file", type=str, default=None, help="Failed image URL JSON file")
    parser.add_argument("--artifacts-dir", type=str, default=str(DEFAULT_ARTIFACTS_DIR), help="Directory for runtime artifacts")
    parser.add_argument("--no-resume", action="store_true", help="Disable checkpoint resume")
    parser.add_argument("--reset-checkpoint", action="store_true", help="Delete existing checkpoint before start")
    parser.add_argument("--max-items-per-year", type=int, default=0, help="Limit articles per year for testing; 0 means no limit")
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    args.artifacts_dir = str(artifacts_dir)
    args.summary_file = resolve_artifact_path(args.summary_file, artifacts_dir, f"{SCRIPT_STEM}_summary.json")
    args.checkpoint_file = resolve_artifact_path(args.checkpoint_file, artifacts_dir, f"{SCRIPT_STEM}_checkpoint.json")
    args.status_file = resolve_artifact_path(args.status_file, artifacts_dir, f"{SCRIPT_STEM}_status.json")
    args.failed_urls_file = resolve_artifact_path(args.failed_urls_file, artifacts_dir, f"{SCRIPT_STEM}_failed_urls.json")

    image_root = Path(args.image_root)
    image_root.mkdir(parents=True, exist_ok=True)
    args.image_root = str(image_root)

    if args.start_date:
        datetime.datetime.strptime(args.start_date, "%Y-%m-%d")
    if args.end_date:
        datetime.datetime.strptime(args.end_date, "%Y-%m-%d")
    if args.start_date and args.end_date and args.start_date > args.end_date:
        raise ValueError("--start-date must be <= --end-date")
    if args.inflight_multiplier < 1:
        raise ValueError("--inflight-multiplier must be >= 1")
    if args.checkpoint_every < 1:
        raise ValueError("--checkpoint-every must be >= 1")
    if args.delay_min < 0 or args.delay_max < 0:
        raise ValueError("--delay-min/--delay-max must be >= 0")
    if args.delay_min > args.delay_max:
        raise ValueError("--delay-min must be <= --delay-max")

    return args


def normalize_url(url: str) -> str:
    parsed = urlparse(url)
    normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/")
    return normalized.lower()


def safe_folder_name(text: str, max_len: int = 120) -> str:
    if not text:
        return "untitled"
    cleaned = re.sub(r"[<>:\"/\\|?*\x00-\x1f]", "_", text).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    if not cleaned:
        cleaned = "untitled"
    return cleaned[:max_len]


def parse_news_date(raw: Any, year: int) -> str:
    if isinstance(raw, datetime.datetime):
        return raw.strftime("%Y-%m-%d")
    if isinstance(raw, datetime.date):
        return raw.strftime("%Y-%m-%d")
    if isinstance(raw, str) and raw:
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "%Y年%m月%d日"):
            try:
                return datetime.datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue
        match = re.search(r"(\d{4})[-/.年](\d{1,2})[-/.月](\d{1,2})", raw)
        if match:
            y, m, d = match.groups()
            return f"{int(y):04d}-{int(m):02d}-{int(d):02d}"
    return f"{year}-01-01"


def parse_image_links(image_links: Any) -> List[str]:
    if isinstance(image_links, list):
        candidates = [str(x).strip() for x in image_links if x]
    elif isinstance(image_links, str):
        raw = image_links.strip()
        if not raw or raw.lower() == "no images":
            return []
        candidates = [x.strip() for x in raw.split(",") if x.strip()]
    else:
        return []

    seen = set()
    out = []
    for url in candidates:
        lower = url.lower()
        if not (lower.startswith("http://") or lower.startswith("https://")):
            continue
        if any(k in lower for k in SKIP_KEYWORDS):
            continue
        if url not in seen:
            seen.add(url)
            out.append(url)
    return out


def get_image_filename(url: str) -> str:
    path = urlparse(url).path
    name = os.path.basename(path)
    ext = os.path.splitext(name)[1].lower()
    if name and ext in VALID_EXTENSIONS:
        return name

    digest = hashlib.md5(url.encode("utf-8", errors="ignore")).hexdigest()
    if ext not in VALID_EXTENSIONS:
        ext = ".jpg"
    return f"image_{digest}{ext}"


class ImageDownloadScraper:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.db = MongoClient(args.mongo_uri)[args.db_name]
        self.semaphore = asyncio.Semaphore(args.concurrent)
        self.session: Optional[aiohttp.ClientSession] = None

        self.metrics: Dict[str, int] = {
            "article_attempted": 0,
            "article_with_image_links": 0,
            "article_no_valid_images": 0,
            "article_has_images_true": 0,
            "image_attempted": 0,
            "image_success": 0,
            "image_http_non_200": 0,
            "image_timeout": 0,
            "image_request_error": 0,
            "image_small_file": 0,
            "image_not_image_content": 0,
            "failed_urls": 0,
            "mongo_updated": 0,
        }
        self.failed_reason_counter: Dict[str, int] = {}
        self.failed_records: List[Dict[str, Any]] = []

    async def init_session(self) -> None:
        if self.session:
            return
        timeout = aiohttp.ClientTimeout(total=self.args.timeout)
        connector = aiohttp.TCPConnector(
            limit=self.args.concurrent * 4,
            limit_per_host=max(8, self.args.concurrent * 2),
            ttl_dns_cache=600,
            enable_cleanup_closed=True,
            ssl=False,
        )
        self.session = aiohttp.ClientSession(timeout=timeout, connector=connector)

    async def close_session(self) -> None:
        if self.session:
            await self.session.close()

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

    def log_failed(self, url: str, reason: str, year: int, article_key: Dict[str, str]) -> None:
        self.metrics["failed_urls"] += 1
        self.failed_reason_counter[reason] = self.failed_reason_counter.get(reason, 0) + 1
        self.failed_records.append(
            {
                "url": url,
                "reason": reason,
                "year": year,
                "article_key": article_key,
                "timestamp": datetime.datetime.now().isoformat(),
            }
        )

    async def download_one_image(self, url: str, output_file: Path, year: int, article_key: Dict[str, str]) -> bool:
        assert self.session is not None
        success = False

        # Reuse existing files to avoid duplicate downloads during schema-repair runs.
        if output_file.exists() and output_file.stat().st_size >= 1024:
            self.metrics["image_success"] += 1
            return True

        async with self.semaphore:
            for attempt in range(self.args.max_retries):
                self.metrics["image_attempted"] += 1
                try:
                    headers = {
                        "User-Agent": random.choice(USER_AGENTS),
                        "Accept": "image/webp,image/*,*/*;q=0.8",
                        "Accept-Language": "zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7",
                    }
                    async with self.session.get(url, headers=headers) as resp:
                        if resp.status != 200:
                            self.metrics["image_http_non_200"] += 1
                            if attempt == self.args.max_retries - 1:
                                self.log_failed(url, f"http_{resp.status}", year, article_key)
                            await asyncio.sleep(0.15)
                            continue

                        content_type = (resp.headers.get("Content-Type") or "").lower()
                        if "image" not in content_type:
                            self.metrics["image_not_image_content"] += 1
                            self.log_failed(url, f"content_type:{content_type or 'unknown'}", year, article_key)
                            return False

                        content = await resp.read()
                        if len(content) < 1024:
                            self.metrics["image_small_file"] += 1
                            self.log_failed(url, "small_file", year, article_key)
                            return False

                        output_file.parent.mkdir(parents=True, exist_ok=True)
                        await asyncio.to_thread(output_file.write_bytes, content)
                        self.metrics["image_success"] += 1
                        success = True
                        break

                except asyncio.TimeoutError:
                    self.metrics["image_timeout"] += 1
                    if attempt == self.args.max_retries - 1:
                        self.log_failed(url, "timeout", year, article_key)
                    await asyncio.sleep(0.15)
                except Exception as e:
                    self.metrics["image_request_error"] += 1
                    if attempt == self.args.max_retries - 1:
                        self.log_failed(url, f"request_error:{type(e).__name__}", year, article_key)
                    await asyncio.sleep(0.15)

        if success:
            delay = random.uniform(self.args.delay_min, self.args.delay_max)
            if delay > 0:
                await asyncio.sleep(delay)
        return success

    async def process_article(self, item: Dict[str, Any], year: int) -> Tuple[Dict[str, Any], List[Path], Dict[str, str]]:
        self.metrics["article_attempted"] += 1

        doc_id = str(item.get("_id", ""))
        title = item.get("title") or ""
        news_date = parse_news_date(item.get("news_date"), year)
        link = normalize_url(item.get("link") or item.get("url") or "")

        article_key = {
            "doc_id": doc_id,
            "news_date": news_date,
            "link": link,
        }

        image_urls = parse_image_links(item.get("image_links"))
        if image_urls:
            self.metrics["article_with_image_links"] += 1
        else:
            self.metrics["article_no_valid_images"] += 1

        title_part = safe_folder_name(title)
        date_part = news_date.replace("-", "")
        article_dir = Path(self.args.image_root) / f"{year}_1" / date_part / title_part

        downloaded_paths: List[Path] = []
        if image_urls:
            tasks = []
            path_pairs: List[Tuple[str, Path]] = []
            for url in image_urls:
                filename = get_image_filename(url)
                dst = article_dir / filename
                path_pairs.append((url, dst))
                tasks.append(self.download_one_image(url, dst, year, article_key))

            results = await asyncio.gather(*tasks)
            for (url, dst), ok in zip(path_pairs, results):
                if ok:
                    downloaded_paths.append(dst)

        has_images = len(downloaded_paths) > 0
        if has_images:
            self.metrics["article_has_images_true"] += 1

        mapping_doc = {
            "original_id": doc_id,
            "title": title,
            "news_date": news_date,
            "link": link,
            "folder_path": str(article_dir),
            "image_paths": [str(p) for p in downloaded_paths],
            "image_urls": image_urls,
            "has_images": has_images,
            "year": year,
            "processed": True,
            "updated_at": datetime.datetime.now(),
        }

        return mapping_doc, downloaded_paths, article_key

    async def bulk_write_updates(self, year: int, article_updates: List[UpdateOne], mapping_updates: List[UpdateOne]) -> None:
        if not article_updates and not mapping_updates:
            return

        if article_updates:
            collection = self.db[f"{year}_1"]
            result = await asyncio.to_thread(collection.bulk_write, article_updates, ordered=False)
            self.metrics["mongo_updated"] += result.modified_count

        if mapping_updates:
            mapping_collection = self.db[f"{year}_mapping"]
            await asyncio.to_thread(mapping_collection.bulk_write, mapping_updates, ordered=False)

    async def process_year(
        self,
        year: int,
        checkpoint_data: Dict[str, Any],
        status_data: Dict[str, Any],
        resume_key: Optional[Tuple[str, str]],
    ) -> Tuple[int, int]:
        source_collection = self.db[f"{year}_1"]
        mapping_collection = self.db[f"{year}_mapping"]

        await asyncio.to_thread(
            mapping_collection.create_index,
            [("link", ASCENDING), ("news_date", ASCENDING)],
            unique=True,
            name="uniq_link_news_date",
        )

        source_docs = list(
            source_collection.find(
                {},
                {"_id": 1, "title": 1, "link": 1, "url": 1, "news_date": 1, "image_links": 1, "has_images": 1},
            )
        )

        if self.args.max_items_per_year > 0:
            source_docs = source_docs[: self.args.max_items_per_year]

        if not source_docs:
            logging.info(f"Year {year}: no source documents, skip")
            return 0, 0

        pending: List[Dict[str, Any]] = []
        for d in source_docs:
            news_date = parse_news_date(d.get("news_date"), year)
            if self.args.start_date and news_date < self.args.start_date:
                continue
            if self.args.end_date and news_date > self.args.end_date:
                continue

            norm_link = normalize_url(d.get("link") or d.get("url") or "")
            if not norm_link:
                continue

            d["news_date"] = news_date
            d["link"] = norm_link
            pending.append(d)

        pending.sort(key=lambda x: (x.get("news_date", ""), x.get("link", "")))

        if resume_key:
            pending = [p for p in pending if (p.get("news_date", ""), p.get("link", "")) > resume_key]

        total_pending = len(pending)
        if total_pending == 0:
            logging.info(f"Year {year}: nothing pending after dedupe/resume")
            return len(source_docs), 0

        logging.info(f"Year {year}: source={len(source_docs)}, pending={total_pending}, concurrent={self.args.concurrent}")

        article_updates: List[UpdateOne] = []
        mapping_updates: List[UpdateOne] = []
        processed_in_year = 0
        success_in_year = 0

        pending_iter = iter(pending)
        in_flight: Dict[asyncio.Task, Dict[str, Any]] = {}
        max_in_flight = self.args.concurrent * self.args.inflight_multiplier

        with tqdm(total=total_pending, desc=f"Year {year} images", unit="article") as pbar:
            while True:
                while len(in_flight) < max_in_flight:
                    try:
                        item = next(pending_iter)
                    except StopIteration:
                        break
                    task = asyncio.create_task(self.process_article(item, year))
                    in_flight[task] = item

                if not in_flight:
                    break

                done, _ = await asyncio.wait(in_flight.keys(), return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    item = in_flight.pop(task)
                    mapping_doc, downloaded_paths, article_key = await task
                    processed_in_year += 1
                    pbar.update(1)

                    has_images = len(downloaded_paths) > 0
                    if has_images:
                        success_in_year += 1

                    article_updates.append(
                        UpdateOne(
                            {"_id": item["_id"]},
                            {
                                "$set": {
                                    "has_images": has_images,
                                    "image_downloaded_at": datetime.datetime.now(),
                                }
                            },
                            upsert=False,
                        )
                    )

                    mapping_updates.append(
                        UpdateOne(
                            {"link": mapping_doc["link"], "news_date": mapping_doc["news_date"]},
                            {"$set": mapping_doc},
                            upsert=True,
                        )
                    )

                    if len(article_updates) >= self.args.batch_size:
                        await self.bulk_write_updates(year, article_updates, mapping_updates)
                        article_updates = []
                        mapping_updates = []

                    checkpoint_data["current_year"] = year
                    checkpoint_data["last_completed_key"] = {
                        "news_date": article_key["news_date"],
                        "link": article_key["link"],
                    }
                    checkpoint_data["processed_items"] = checkpoint_data.get("processed_items", 0) + 1
                    checkpoint_data["updated_at"] = datetime.datetime.now().isoformat()

                    if processed_in_year % self.args.checkpoint_every == 0:
                        status_data["current_year"] = year
                        status_data["processed_items"] = checkpoint_data.get("processed_items", 0)
                        status_data["updated_at"] = datetime.datetime.now().isoformat()
                        status_data["last_error"] = None
                        self.save_checkpoint(checkpoint_data)
                        self.save_status(status_data)

        if article_updates or mapping_updates:
            await self.bulk_write_updates(year, article_updates, mapping_updates)

        self.save_checkpoint(checkpoint_data)
        return len(source_docs), success_in_year

    async def run(self) -> Dict[str, Any]:
        run_started_at = datetime.datetime.now()
        run_id = run_started_at.strftime("%Y%m%d_%H%M%S")

        if self.args.reset_checkpoint and os.path.exists(self.args.checkpoint_file):
            os.remove(self.args.checkpoint_file)
            logging.info(f"Reset checkpoint file: {self.args.checkpoint_file}")

        checkpoint = self.load_checkpoint() if not self.args.no_resume else None

        checkpoint_data: Dict[str, Any] = {
            "run_id": run_id,
            "status": "in_progress",
            "db_name": self.args.db_name,
            "start_year": self.args.start_year,
            "end_year": self.args.end_year,
            "summary_file": self.args.summary_file,
            "checkpoint_file": self.args.checkpoint_file,
            "status_file": self.args.status_file,
            "current_year": self.args.start_year,
            "last_completed_key": None,
            "processed_items": 0,
            "updated_at": datetime.datetime.now().isoformat(),
        }

        status_data: Dict[str, Any] = {
            "run_id": run_id,
            "status": "running",
            "pid": os.getpid(),
            "db_name": self.args.db_name,
            "year_range": f"{self.args.start_year}-{self.args.end_year}",
            "date_range": {
                "start_date": self.args.start_date,
                "end_date": self.args.end_date,
            },
            "current_year": self.args.start_year,
            "processed_items": 0,
            "updated_at": datetime.datetime.now().isoformat(),
            "files": {
                "checkpoint_file": self.args.checkpoint_file,
                "summary_file": self.args.summary_file,
                "failed_urls_file": self.args.failed_urls_file,
            },
            "last_error": None,
        }
        self.save_status(status_data)

        resumed = False
        resume_year = self.args.start_year
        resume_key: Optional[Tuple[str, str]] = None

        if checkpoint:
            same_job = (
                checkpoint.get("db_name") == self.args.db_name
                and checkpoint.get("start_year") == self.args.start_year
                and checkpoint.get("end_year") == self.args.end_year
                and checkpoint.get("status") in {"in_progress", "completed"}
            )
            if same_job:
                resumed = True
                resume_year = int(checkpoint.get("current_year", self.args.start_year))
                key_obj = checkpoint.get("last_completed_key") or {}
                key_news_date = key_obj.get("news_date")
                key_link = key_obj.get("link")
                if key_news_date and key_link:
                    resume_key = (key_news_date, key_link)
                checkpoint_data["processed_items"] = int(checkpoint.get("processed_items", 0))
                logging.info(
                    f"Resume enabled: year={resume_year}, key={resume_key}, "
                    f"processed_items={checkpoint_data['processed_items']}"
                )

        await self.init_session()

        year_stats: List[Dict[str, Any]] = []
        years = range(self.args.start_year, self.args.end_year + 1)

        try:
            for year in years:
                if year < resume_year:
                    continue

                use_resume_key = resume_key if year == resume_year else None
                source_total, with_images = await self.process_year(year, checkpoint_data, status_data, use_resume_key)
                year_stats.append(
                    {
                        "year": year,
                        "source_total": source_total,
                        "article_has_images_true": with_images,
                    }
                )

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
            await self.close_session()

        checkpoint_data["status"] = "completed"
        checkpoint_data["updated_at"] = datetime.datetime.now().isoformat()
        self.save_checkpoint(checkpoint_data)

        status_data["status"] = "completed"
        status_data["current_year"] = self.args.end_year
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
                "start_year": self.args.start_year,
                "end_year": self.args.end_year,
                "start_date": self.args.start_date,
                "end_date": self.args.end_date,
                "batch_size": self.args.batch_size,
                "concurrent": self.args.concurrent,
                "inflight_multiplier": self.args.inflight_multiplier,
                "max_retries": self.args.max_retries,
                "timeout": self.args.timeout,
                "delay_min": self.args.delay_min,
                "delay_max": self.args.delay_max,
                "checkpoint_every": self.args.checkpoint_every,
                "image_root": self.args.image_root,
                "checkpoint_file": self.args.checkpoint_file,
                "status_file": self.args.status_file,
                "summary_file": self.args.summary_file,
                "resumed_from_checkpoint": resumed,
                "max_items_per_year": self.args.max_items_per_year,
            },
            "metrics": self.metrics,
            "failed_reason_counts": self.failed_reason_counter,
            "failed_records_count": len(self.failed_records),
            "year_stats": year_stats,
        }

        with open(self.args.summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        failed_file = Path(self.args.failed_urls_file)
        with open(failed_file, "w", encoding="utf-8") as f:
            json.dump(self.failed_records, f, ensure_ascii=False, indent=2)

        logging.info(f"Summary written to {self.args.summary_file}")
        logging.info(f"Failed URL records written to {failed_file}")
        return summary


def main() -> None:
    args = parse_args()
    start = time.time()

    logging.info("=== Starting Async Image Downloader ===")
    logging.info(f"Database: {args.db_name}")
    logging.info(f"Year range: {args.start_year}-{args.end_year}")
    logging.info(f"Concurrent requests: {args.concurrent}")
    logging.info(f"Batch size: {args.batch_size}")

    scraper = ImageDownloadScraper(args)
    asyncio.run(scraper.run())

    elapsed = time.time() - start
    logging.info(f"=== Image Download Complete, elapsed: {elapsed:.2f}s ===")


if __name__ == "__main__":
    main()
