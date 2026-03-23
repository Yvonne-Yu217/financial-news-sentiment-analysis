import argparse
import asyncio
import datetime
import json
import logging
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import aiohttp
from bs4 import BeautifulSoup
from pymongo import ASCENDING, MongoClient, UpdateOne
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("news_scraper.log"),
        logging.StreamHandler(),
    ],
)

# Defaults
START_YEAR = 2014
END_YEAR = 2026
BATCH_SIZE = 100
CONCURRENT_REQUESTS = 48
MAX_RETRIES = 3
HTTP_TIMEOUT = 20
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

# MongoDB defaults
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "sina_news_dataset_test"
SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPT_STEM = Path(__file__).stem
DEFAULT_ARTIFACTS_DIR = SCRIPT_DIR / "run_artifacts"


def resolve_artifact_path(path_arg: Optional[str], artifacts_dir: Path, default_name: str) -> str:
    if not path_arg:
        return str(artifacts_dir / default_name)

    user_path = Path(path_arg)
    if user_path.is_absolute() or user_path.parent != Path("."):
        user_path.parent.mkdir(parents=True, exist_ok=True)
        return str(user_path)

    return str(artifacts_dir / user_path.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape full article content from stored Sina links")
    parser.add_argument("--mongo-uri", type=str, default=MONGO_URI, help="MongoDB URI")
    parser.add_argument("--db-name", type=str, default=DB_NAME, help="MongoDB database name")
    parser.add_argument("--start-year", type=int, default=START_YEAR, help="Start year")
    parser.add_argument("--end-year", type=int, default=END_YEAR, help="End year")
    parser.add_argument("--start-date", type=str, default=None, help="Optional start date filter (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None, help="Optional end date filter (YYYY-MM-DD)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="MongoDB bulk upsert batch size")
    parser.add_argument("--concurrent", type=int, default=CONCURRENT_REQUESTS, help="Concurrent HTTP requests")
    parser.add_argument("--inflight-multiplier", type=int, default=8, help="Queued tasks = concurrent * inflight-multiplier")
    parser.add_argument("--max-retries", type=int, default=MAX_RETRIES, help="Max retries per URL")
    parser.add_argument("--timeout", type=int, default=HTTP_TIMEOUT, help="HTTP timeout seconds")
    parser.add_argument("--delay-min", type=float, default=0.0, help="Random delay lower bound per request")
    parser.add_argument("--delay-max", type=float, default=0.05, help="Random delay upper bound per request")
    parser.add_argument("--checkpoint-every", type=int, default=200, help="Save checkpoint/status every N processed items")
    parser.add_argument("--summary-file", type=str, default=None, help="Summary JSON output file")
    parser.add_argument("--checkpoint-file", type=str, default=None, help="Checkpoint file")
    parser.add_argument("--status-file", type=str, default=None, help="Human-readable status JSON file")
    parser.add_argument("--failed-urls-file", type=str, default=None, help="Failed URL JSON output file")
    parser.add_argument("--artifacts-dir", type=str, default=str(DEFAULT_ARTIFACTS_DIR), help="Directory for runtime artifacts")
    parser.add_argument("--no-resume", action="store_true", help="Disable checkpoint resume")
    parser.add_argument("--reset-checkpoint", action="store_true", help="Delete existing checkpoint before start")
    parser.add_argument("--max-items-per-year", type=int, default=0, help="Limit items per year for testing; 0 means no limit")
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    args.artifacts_dir = str(artifacts_dir)
    args.summary_file = resolve_artifact_path(args.summary_file, artifacts_dir, f"{SCRIPT_STEM}_summary.json")
    args.checkpoint_file = resolve_artifact_path(args.checkpoint_file, artifacts_dir, f"{SCRIPT_STEM}_checkpoint.json")
    args.status_file = resolve_artifact_path(args.status_file, artifacts_dir, f"{SCRIPT_STEM}_status.json")
    args.failed_urls_file = resolve_artifact_path(args.failed_urls_file, artifacts_dir, f"{SCRIPT_STEM}_failed_urls.json")

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


def decode_bytes(content: bytes) -> str:
    for encoding in ["gb18030", "gbk", "gb2312", "utf-8"]:
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="replace")


class NewsContentScraper:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.db = MongoClient(args.mongo_uri)[args.db_name]
        self.semaphore = asyncio.Semaphore(args.concurrent)
        self.session: Optional[aiohttp.ClientSession] = None

        self.metrics: Dict[str, int] = {
            "fetch_attempted": 0,
            "fetch_success": 0,
            "http_non_200": 0,
            "timeout": 0,
            "request_error": 0,
            "parse_failed": 0,
            "content_empty": 0,
            "failed_urls": 0,
            "mongo_upsert_new": 0,
            "mongo_upsert_duplicate": 0,
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
        )
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,application/xml",
                "Accept-Language": "zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7",
            },
        )

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

    def log_failed(self, url: str, reason: str, year: int) -> None:
        self.metrics["failed_urls"] += 1
        self.failed_reason_counter[reason] = self.failed_reason_counter.get(reason, 0) + 1
        self.failed_records.append(
            {
                "url": url,
                "reason": reason,
                "year": year,
                "timestamp": datetime.datetime.now().isoformat(),
            }
        )

    def extract_article(self, html: str, url: str, source_title: str, source_category: str, source_news_date: str) -> Optional[Dict[str, Any]]:
        try:
            soup = BeautifulSoup(html, "html.parser")

            title = source_title.strip() if source_title else ""
            if not title:
                title_tag = soup.find("h1")
                title = title_tag.get_text(strip=True) if title_tag else ""

            article_container = (
                soup.find("div", id="artibody")
                or soup.find("div", class_="article")
                or soup.find("div", class_=re.compile(r"article|content", re.I))
            )
            if not article_container:
                self.metrics["parse_failed"] += 1
                return None

            paragraphs = [p.get_text(" ", strip=True) for p in article_container.find_all("p")]
            content = "\n".join([p for p in paragraphs if p])
            if not content:
                self.metrics["content_empty"] += 1
                return None

            image_links: List[str] = []
            for img in article_container.find_all("img"):
                src = img.get("src")
                if src:
                    image_links.append(urljoin(url, src))

            normalized_link = normalize_url(url)
            return {
                # Compatibility with previous pipeline output
                "title": title,
                "category": source_category,
                "news_date": source_news_date,
                "content": content,
                "link": normalized_link,
                "url": normalized_link,
                "fetch_date": datetime.datetime.now(),
                "image_links": image_links,
            }
        except Exception as e:
            logging.error(f"Parse error for {url}: {e}")
            self.metrics["parse_failed"] += 1
            return None

    async def fetch_one(self, item: Dict[str, Any], year: int) -> Tuple[Optional[Dict[str, Any]], Tuple[str, str]]:
        raw_url = item.get("link") or item.get("url") or ""
        url = raw_url.strip()
        news_date = item.get("news_date") or f"{year}-01-01"
        key = (news_date, normalize_url(url) if url else "")

        if not url:
            self.log_failed("", "missing_url", year)
            return None, key

        assert self.session is not None
        success_doc: Optional[Dict[str, Any]] = None
        async with self.semaphore:
            for attempt in range(self.args.max_retries):
                self.metrics["fetch_attempted"] += 1
                try:
                    async with self.session.get(url) as resp:
                        if resp.status != 200:
                            self.metrics["http_non_200"] += 1
                            if attempt == self.args.max_retries - 1:
                                self.log_failed(url, f"http_{resp.status}", year)
                            await asyncio.sleep(0.2)
                            continue

                        content = await resp.read()
                        text = decode_bytes(content)
                        parsed = self.extract_article(
                            text,
                            url,
                            source_title=item.get("title", ""),
                            source_category=item.get("category", ""),
                            source_news_date=news_date,
                        )
                        if parsed:
                            self.metrics["fetch_success"] += 1
                            success_doc = parsed
                            break

                        if attempt == self.args.max_retries - 1:
                            self.log_failed(url, "parse_or_content_empty", year)
                        await asyncio.sleep(0.2)

                except asyncio.TimeoutError:
                    self.metrics["timeout"] += 1
                    if attempt == self.args.max_retries - 1:
                        self.log_failed(url, "timeout", year)
                    await asyncio.sleep(0.2)
                except Exception as e:
                    self.metrics["request_error"] += 1
                    if attempt == self.args.max_retries - 1:
                        self.log_failed(url, f"request_error:{type(e).__name__}", year)
                    await asyncio.sleep(0.2)

        if success_doc:
            delay = random.uniform(self.args.delay_min, self.args.delay_max)
            if delay > 0:
                await asyncio.sleep(delay)
            return success_doc, key

        return None, key

    async def bulk_upsert(self, collection_name: str, docs: List[Dict[str, Any]]) -> None:
        if not docs:
            return

        collection = self.db[collection_name]
        operations = [
            UpdateOne(
                {"link": d["link"], "news_date": d["news_date"]},
                {"$set": d},
                upsert=True,
            )
            for d in docs
        ]
        result = await asyncio.to_thread(collection.bulk_write, operations, ordered=False)
        new_docs = result.upserted_count
        dup_docs = len(operations) - new_docs
        self.metrics["mongo_upsert_new"] += new_docs
        self.metrics["mongo_upsert_duplicate"] += dup_docs

    async def process_year(
        self,
        year: int,
        checkpoint_data: Dict[str, Any],
        status_data: Dict[str, Any],
        resume_key: Optional[Tuple[str, str]],
    ) -> Tuple[int, int, int]:
        source_collection = self.db[str(year)]
        target_collection_name = f"{year}_1"
        target_collection = self.db[target_collection_name]

        await asyncio.to_thread(
            target_collection.create_index,
            [("link", ASCENDING), ("news_date", ASCENDING)],
            unique=True,
            name="uniq_link_news_date",
        )

        source_docs = list(
            source_collection.find({}, {"title": 1, "link": 1, "url": 1, "category": 1, "news_date": 1})
        )
        source_docs = [d for d in source_docs if d.get("link") or d.get("url")]

        if self.args.max_items_per_year > 0:
            source_docs = source_docs[: self.args.max_items_per_year]

        if not source_docs:
            logging.info(f"Year {year}: no source data, skip")
            return 0, 0, 0

        existing_pairs = {
            (doc.get("news_date", ""), doc.get("link", ""))
            for doc in target_collection.find({}, {"news_date": 1, "link": 1})
        }

        pending: List[Dict[str, Any]] = []
        for d in source_docs:
            news_date = d.get("news_date") or f"{year}-01-01"
            if self.args.start_date and news_date < self.args.start_date:
                continue
            if self.args.end_date and news_date > self.args.end_date:
                continue
            norm_link = normalize_url(d.get("link") or d.get("url"))
            if (news_date, norm_link) in existing_pairs:
                continue
            d["link"] = norm_link
            pending.append(d)

        pending.sort(key=lambda x: (x.get("news_date", ""), x.get("link", "")))

        if resume_key:
            pending = [
                p
                for p in pending
                if (p.get("news_date", ""), p.get("link", "")) > resume_key
            ]

        total_pending = len(pending)
        if total_pending == 0:
            logging.info(f"Year {year}: nothing pending after dedupe/resume")
            return len(source_docs), 0, 0

        logging.info(f"Year {year}: source={len(source_docs)}, pending={total_pending}, concurrent={self.args.concurrent}")

        saved_new_before = self.metrics["mongo_upsert_new"]
        saved_dup_before = self.metrics["mongo_upsert_duplicate"]

        batch_docs: List[Dict[str, Any]] = []
        processed_in_year = 0
        pending_iter = iter(pending)
        in_flight: Dict[asyncio.Task, Dict[str, Any]] = {}
        max_in_flight = self.args.concurrent * self.args.inflight_multiplier

        with tqdm(total=total_pending, desc=f"Year {year}", unit="article") as pbar:
            while True:
                while len(in_flight) < max_in_flight:
                    try:
                        item = next(pending_iter)
                    except StopIteration:
                        break
                    task = asyncio.create_task(self.fetch_one(item, year))
                    in_flight[task] = item

                if not in_flight:
                    break

                done, _ = await asyncio.wait(in_flight.keys(), return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    _ = in_flight.pop(task)
                    doc, key = await task
                    processed_in_year += 1
                    pbar.update(1)

                    if doc:
                        batch_docs.append(doc)

                    if len(batch_docs) >= self.args.batch_size:
                        await self.bulk_upsert(target_collection_name, batch_docs)
                        batch_docs = []

                    checkpoint_data["current_year"] = year
                    checkpoint_data["last_completed_key"] = {
                        "news_date": key[0],
                        "link": key[1],
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

        if batch_docs:
            await self.bulk_upsert(target_collection_name, batch_docs)

        self.save_checkpoint(checkpoint_data)

        year_new = self.metrics["mongo_upsert_new"] - saved_new_before
        year_dup = self.metrics["mongo_upsert_duplicate"] - saved_dup_before
        return len(source_docs), year_new, year_dup

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
                source_total, year_new, year_dup = await self.process_year(year, checkpoint_data, status_data, use_resume_key)
                year_stats.append(
                    {
                        "year": year,
                        "source_total": source_total,
                        "upsert_new": year_new,
                        "upsert_duplicate": year_dup,
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
                "max_retries": self.args.max_retries,
                "timeout": self.args.timeout,
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

    logging.info("=== Starting Async News Content Scraper ===")
    logging.info(f"Database: {args.db_name}")
    logging.info(f"Year range: {args.start_year}-{args.end_year}")
    logging.info(f"Concurrent requests: {args.concurrent}")
    logging.info(f"Batch size: {args.batch_size}")

    scraper = NewsContentScraper(args)
    asyncio.run(scraper.run())

    elapsed = time.time() - start
    logging.info(f"=== Scraping Complete, elapsed: {elapsed:.2f}s ===")


if __name__ == "__main__":
    main()
