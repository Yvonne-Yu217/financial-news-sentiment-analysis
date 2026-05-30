import aiohttp
import asyncio
from bs4 import BeautifulSoup
from pymongo import MongoClient, UpdateOne
import datetime
import time
import random
from typing import List, Dict, Optional
from dataclasses import dataclass, field
import logging
import chardet
import re
from tqdm import tqdm
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import json
from pathlib import Path
import requests
import argparse
import os
import pandas as pd
import sys
from pymongo import ASCENDING
from collections import Counter


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPT_STEM = Path(__file__).stem
DEFAULT_ARTIFACTS_DIR = SCRIPT_DIR / "run_artifacts" / SCRIPT_STEM


def _resolve_artifact_path(path_arg: Optional[str], artifacts_dir: Path, default_name: str) -> str:
    if not path_arg:
        return str(artifacts_dir / default_name)

    user_path = Path(path_arg)
    if user_path.is_absolute() or user_path.parent != Path('.'):
        user_path.parent.mkdir(parents=True, exist_ok=True)
        return str(user_path)

    return str(artifacts_dir / user_path.name)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


@dataclass
class Config:
    # MongoDB configuration
    MONGO_URI: str = "mongodb://localhost:27017/"
    DB_NAME: str = "sina_news_dataset_test"
    
    # Crawler configuration
    BASE_URL: str = "https://news.sina.com.cn/head/news{YYYYMMDD}{AMPM}.shtml"
    MAX_RETRIES_PER_URL: int = 3
    CONCURRENT_REQUESTS: int = 10  # Number of concurrent requests
    BATCH_SIZE: int = 100  # MongoDB batch write size
    HTTP_TIMEOUT: int = 30
    
    # Date configuration
    START_DATE: str = "2014-01-01"  # Default start date
    END_DATE: str = "2026-3-20"    # Default end date
    EXCEL_FILE: str = "Stock Market Index.xlsx"  # Excel file path
    DATE_COLUMN: str = "Date"       # Date column name
    CLEAR_EXISTING: bool = False     # Clear existing year collections before crawling
    ARTIFACTS_DIR: str = str(DEFAULT_ARTIFACTS_DIR)  # Fixed directory for runtime artifacts
    SUMMARY_FILE: str = str(DEFAULT_ARTIFACTS_DIR / f"{SCRIPT_STEM}_summary.json")
    CHECKPOINT_FILE: str = str(DEFAULT_ARTIFACTS_DIR / f"{SCRIPT_STEM}_checkpoint.json")
    STATUS_FILE: str = str(DEFAULT_ARTIFACTS_DIR / f"{SCRIPT_STEM}_status.json")
    RESUME_FROM_CHECKPOINT: bool = True
    
    # Category mapping
    CATEGORY_MAPPING: Dict[str, str] = field(default_factory=lambda: {
        "headline": "Headlines",
        "news_item": "Headlines",
        "blk_cjxwgngjcj_01": "Domestic·International Finance",
        "blk_cjxwgpggmg_01": "Stocks·Hong Kong·US Stocks",
        "blk_cjxwlcsh_01": "Finance·Life",
        "blk_gnxw_01": "Domestic News",
        "blk_ndxw_01": "Domestic News",
        "blk_cjkjqcfc_01": "Finance·Tech·Auto·Real Estate",
        "blk_kjxwhlw_01": "Tech·Internet",
        "blk_kjxwkjts_01": "Tech·Exploration"
    })
    
    # News block configuration
    NEWS_BLOCKS: Dict[str, str] = field(default_factory=lambda: {
        "blk_yw_01": "Headlines",
        "blk_cjxwgngjcj_01": "Domestic·International Finance",
        "blk_cjxwgpggmg_01": "Stocks·Hong Kong·US Stocks",
        "blk_cjxwlcsh_01": "Finance·Life",
        "blk_gnxw_01": "Domestic News",
        "blk_ndxw_01": "Domestic News",
        "blk_cjkjqcfc_01": "Finance·Tech·Auto·Real Estate",
        "blk_kjxwhlw_01": "Tech·Internet",
        "blk_kjxwkjts_01": "Tech·Exploration"
    })

    # Failed URLs file path
    FAILED_URLS_FILE: str = str(DEFAULT_ARTIFACTS_DIR / f"{SCRIPT_STEM}_failed_urls.json")

    HEADERS: Dict[str, str] = field(default_factory=lambda: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })

    # Encoding configuration
    ENCODINGS: List[str] = field(default_factory=lambda: [
        'gb18030',  # Prioritize GB18030 (backward compatible with GB2312 and GBK)
        'gbk',
        'gb2312',
        'utf-8'
    ])

    @classmethod
    def from_args(cls):
        """Create configuration from command line arguments"""
        parser = argparse.ArgumentParser(description='Sina News Crawler')
        
        # Add command line arguments
        parser.add_argument('--mongo-uri', type=str, help='MongoDB connection URI')
        parser.add_argument('--db-name', type=str, help='MongoDB database name')
        parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
        parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
        parser.add_argument('--excel-file', type=str, help='Path to Excel file containing date list')
        parser.add_argument('--date-column', type=str, help='Date column name in Excel file')
        parser.add_argument('--batch-size', type=int, help='Batch write size')
        parser.add_argument('--concurrent', type=int, help='Number of concurrent requests')
        parser.add_argument('--clear-existing', action='store_true',
                    help='Clear existing year collections before crawl (dangerous)')
        parser.add_argument('--summary-file', type=str,
                    help='Path to structured run summary JSON file')
        parser.add_argument('--checkpoint-file', type=str,
                help='Path to checkpoint JSON file for resume')
        parser.add_argument('--status-file', type=str,
            help='Path to human-readable status JSON file')
        parser.add_argument('--failed-urls-file', type=str,
            help='Path to failed urls JSON file')
        parser.add_argument('--artifacts-dir', type=str,
            help='Directory for summary/checkpoint/failed_urls outputs')
        parser.add_argument('--no-resume', action='store_true',
                help='Disable resuming from existing checkpoint')
        parser.add_argument('--reset-checkpoint', action='store_true',
                help='Delete existing checkpoint file before crawl starts')
        
        args = parser.parse_args()
        
        # Create configuration instance
        config = cls()
        
        # Update configuration
        if args.mongo_uri:
            config.MONGO_URI = args.mongo_uri
        if args.db_name:
            config.DB_NAME = args.db_name
        if args.start_date:
            config.START_DATE = args.start_date
        if args.end_date:
            config.END_DATE = args.end_date
        if args.excel_file:
            config.EXCEL_FILE = args.excel_file
        if args.date_column:
            config.DATE_COLUMN = args.date_column
        if args.batch_size:
            config.BATCH_SIZE = args.batch_size
        if args.concurrent:
            config.CONCURRENT_REQUESTS = args.concurrent
        if args.clear_existing:
            config.CLEAR_EXISTING = True

        artifacts_dir = Path(args.artifacts_dir) if args.artifacts_dir else Path(config.ARTIFACTS_DIR)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        config.ARTIFACTS_DIR = str(artifacts_dir)
        config.SUMMARY_FILE = _resolve_artifact_path(
            args.summary_file,
            artifacts_dir,
            f"{SCRIPT_STEM}_summary.json"
        )
        config.CHECKPOINT_FILE = _resolve_artifact_path(
            args.checkpoint_file,
            artifacts_dir,
            f"{SCRIPT_STEM}_checkpoint.json"
        )
        config.STATUS_FILE = _resolve_artifact_path(
            args.status_file,
            artifacts_dir,
            f"{SCRIPT_STEM}_status.json"
        )
        config.FAILED_URLS_FILE = _resolve_artifact_path(
            args.failed_urls_file,
            artifacts_dir,
            f"{SCRIPT_STEM}_failed_urls.json"
        )

        if args.no_resume:
            config.RESUME_FROM_CHECKPOINT = False

        if args.reset_checkpoint and os.path.exists(config.CHECKPOINT_FILE):
            os.remove(config.CHECKPOINT_FILE)
            logging.info(f"Reset checkpoint file: {config.CHECKPOINT_FILE}")
            
        return config
    
    def get_date_list(self) -> List[datetime.date]:
        """Get list of dates to crawl"""
        # Parse start and end dates
        try:
            start_date = datetime.datetime.strptime(self.START_DATE, "%Y-%m-%d").date()
            end_date = datetime.datetime.strptime(self.END_DATE, "%Y-%m-%d").date()
            logging.info(f"Date range set: {start_date} to {end_date}")
        except Exception as e:
            logging.error(f"Error parsing date range: {e}")
            sys.exit(1)
            
        # Read dates from Excel file
        if self.EXCEL_FILE and os.path.exists(self.EXCEL_FILE):
            try:
                logging.info(f"Reading date list from Excel file {self.EXCEL_FILE}")
                df = pd.read_excel(self.EXCEL_FILE)
                
                if self.DATE_COLUMN not in df.columns:
                    logging.error(f"Column '{self.DATE_COLUMN}' not found in Excel file")
                    logging.info(f"Available columns: {', '.join(df.columns)}")
                    sys.exit(1)
                
                # Extract date column and convert to datetime.date objects
                dates = []
                filtered_dates = []
                invalid_dates = 0
                
                for date_str in df[self.DATE_COLUMN]:
                    try:
                        # Try to convert various date formats
                        date_obj = None
                        
                        if isinstance(date_str, datetime.datetime) or isinstance(date_str, pd.Timestamp):
                            date_obj = date_str.date()
                        elif isinstance(date_str, str):
                            # Try parsing different date string formats
                            try:
                                date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
                            except ValueError:
                                try:
                                    date_obj = datetime.datetime.strptime(date_str, "%Y/%m/%d").date()
                                except ValueError:
                                    try:
                                        date_obj = datetime.datetime.strptime(date_str, "%d/%m/%Y").date()
                                    except ValueError:
                                        logging.warning(f"Cannot parse date: {date_str}, skipping")
                                        invalid_dates += 1
                                        continue
                        
                        if date_obj:
                            dates.append(date_obj)
                            # Only keep dates between start and end dates
                            if start_date <= date_obj <= end_date:
                                filtered_dates.append(date_obj)
                    except Exception as e:
                        logging.warning(f"Error processing date {date_str}: {e}")
                        invalid_dates += 1
                
                # Remove duplicates and sort
                unique_dates = sorted(list(set(filtered_dates)))
                
                logging.info(f"Read {len(dates)} dates from Excel file")
                logging.info(f"Found {len(unique_dates)} dates within specified range {start_date} to {end_date}")
                
                if invalid_dates > 0:
                    logging.warning(f"{invalid_dates} dates could not be parsed")
                
                if not unique_dates:
                    logging.error("No valid dates found in Excel file within specified range")
                    sys.exit(1)
                
                return unique_dates
                
            except Exception as e:
                logging.error(f"Error reading Excel file: {e}")
                logging.warning("Fallback to date range mode")
        else:
            logging.warning(f"Excel file {self.EXCEL_FILE} not found, fallback to date range mode")

        # Fallback: use full date range if Excel file is missing/invalid.
        delta_days = (end_date - start_date).days
        if delta_days < 0:
            logging.error("End date is earlier than start date")
            sys.exit(1)

        date_list = [start_date + datetime.timedelta(days=i) for i in range(delta_days + 1)]
        logging.info(f"Fallback date mode active, generated {len(date_list)} dates")
        return date_list

class NewsSpider:
    def __init__(self, config: Config):
        self.config = config
        self.db = self._init_mongo()
        self.session = None
        self.failed_urls = []
        self.failed_url_records = []
        self.indexed_collections = set()
        self.semaphore = asyncio.Semaphore(config.CONCURRENT_REQUESTS)
        self.request_metrics = {
            'fetch_attempted': 0,
            'fetch_success': 0,
            'http_non_200': 0,
            'timeout': 0,
            'request_error': 0,
            'parse_no_chinese': 0,
            'failed_urls_logged': 0
        }
        self.failed_reason_counter = Counter()

    def _init_mongo(self) -> MongoClient:
        """Initialize MongoDB connection"""
        client = MongoClient(self.config.MONGO_URI)
        return client[self.config.DB_NAME]

    async def _init_session(self):
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=self.config.HTTP_TIMEOUT)
            connector = aiohttp.TCPConnector(limit=self.config.CONCURRENT_REQUESTS * 2)
            self.session = aiohttp.ClientSession(
                headers=self.config.HEADERS,
                timeout=timeout,
                connector=connector
            )

    def _detect_and_decode(self, content: bytes) -> str:
        """Smartly detect and decode content"""
        # List of encodings to prioritize
        encodings = self.config.ENCODINGS
        
        # First, use chardet to detect
        detected = chardet.detect(content)
        if detected and detected['confidence'] > 0.8:
            try:
                return content.decode(detected['encoding'])
            except (UnicodeDecodeError, LookupError):
                pass
        
        # If chardet detection fails, try other encodings in sequence
        for encoding in encodings:
            try:
                return content.decode(encoding)
            except (UnicodeDecodeError, LookupError):
                continue
        
        # If all attempts fail, use errors='replace' for downgrade processing
        try:
            # Try gb18030 first as it's the most comprehensive Chinese encoding
            return content.decode('gb18030', errors='replace')
        except Exception:
            # Last resort: use utf-8 with replacement
            return content.decode('utf-8', errors='replace')

    async def fetch_url(self, url: str) -> Optional[str]:
        """Get URL content"""
        try:
            async with self.semaphore:
                for attempt in range(self.config.MAX_RETRIES_PER_URL):
                    try:
                        async with self.session.get(url) as response:
                            if response.status != 200:
                                if attempt == self.config.MAX_RETRIES_PER_URL - 1:
                                    logging.warning(f"Failed to fetch URL: {url}, status code: {response.status}")
                                await asyncio.sleep(1)
                                continue
                            
                            content = await response.read()
                            return self._detect_and_decode(content)
                    except asyncio.TimeoutError:
                        if attempt == self.config.MAX_RETRIES_PER_URL - 1:
                            logging.warning(f"Timeout fetching URL: {url}")
                        await asyncio.sleep(1)
                    except Exception as e:
                        if attempt == self.config.MAX_RETRIES_PER_URL - 1:
                            logging.error(f"Failed to get URL {url}: {e}")
                        await asyncio.sleep(1)
            
        except Exception as e:
            logging.error(f"Failed to get URL {url}: {e}")
            self.failed_urls.append({
                "url": url,
                "error": str(e),
                "timestamp": datetime.datetime.now().isoformat()
            })
            return None

    def _clean_text(self, text: str) -> str:
        """Clean text content"""
        if not text:
            return ""
        
        # Remove special characters and control characters
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        # Normalize whitespace characters
        text = re.sub(r'\s+', ' ', text)
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Remove repeated punctuation symbols
        text = re.sub(r'[!?。！？，,]{2,}', lambda m: m.group()[0], text)
        # Remove empty parentheses
        text = re.sub(r'\(\s*\)|\[\s*\]|【\s*】|（\s*）', '', text)
        return text.strip()

    async def parse_news_list(self, html: str, date: datetime.date) -> List[Dict]:
        """Parse news list"""
        if not html:
            return []
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            news_list = []
            
            for block_id, category in self.config.NEWS_BLOCKS.items():
                block = soup.find('div', {'id': block_id})
                if not block:
                    continue
                    
                for link in block.find_all('a', href=True):
                    title = self._clean_text(link.get_text())
                    href = link['href']
                    
                    if not title or not href:
                        continue
                        
                    # Verify if title contains valid Chinese characters
                    if not re.search('[\u4e00-\u9fff]', title):
                        continue
                        
                    news_list.append({
                        'title': title,
                        'link': href,
                        'category': category,
                        'news_date': date.strftime('%Y-%m-%d'),
                        'fetch_date': datetime.datetime.now()
                    })
            
            return news_list
            
        except Exception as e:
            logging.error(f"Failed to parse news list: {e}")
            return []

    def _get_encoding(self, response: requests.Response) -> str:
        """Get correct encoding"""
        # Use chardet to detect encoding
        detected = chardet.detect(response.content)
        if detected and detected['confidence'] > 0.8:
            return detected['encoding']
        
        # If chardet detection fails or confidence is low, try to get from Content-Type
        content_type = response.headers.get('Content-Type', '').lower()
        if 'charset=' in content_type:
            charset = content_type.split('charset=')[-1].strip()
            try:
                '测试'.encode(charset)  # Verify if encoding is valid
                return charset
            except:
                pass
        
        # If all else fails, return default encoding
        return 'gb18030'  # Use GB18030 (most comprehensive Chinese encoding)

    async def fetch_news(self, url: str, date_str: str) -> Optional[List[Dict]]:
        """Get news data"""
        try:
            for attempt in range(self.config.MAX_RETRIES_PER_URL):
                try:
                    self.request_metrics['fetch_attempted'] += 1
                    async with self.session.get(url) as response:
                        if response.status != 200:
                            self.request_metrics['http_non_200'] += 1
                            if attempt == self.config.MAX_RETRIES_PER_URL - 1:
                                self._log_failed_url(url, date_str, 
                                                'am' if 'am' in url else 'pm', 
                                                f"HTTP error: {response.status}")
                            await asyncio.sleep(1)
                            continue
                        
                        # Use chardet to detect encoding
                        content = await response.read()
                        
                        # Try to decode with different encodings
                        text = None
                        for encoding in self.config.ENCODINGS:
                            try:
                                text = content.decode(encoding)
                                # Verify if decoded text contains Chinese characters
                                if re.search('[\u4e00-\u9fff]', text):
                                    break
                            except UnicodeDecodeError:
                                continue
                        
                        # If all encodings fail, use errors='replace'
                        if text is None:
                            text = content.decode('gb18030', errors='replace')
                            if not re.search('[\u4e00-\u9fff]', text):
                                text = content.decode('utf-8', errors='replace')
                        
                        # Verify decoded text
                        if not re.search('[\u4e00-\u9fff]', text):
                            self.request_metrics['parse_no_chinese'] += 1
                            if attempt == self.config.MAX_RETRIES_PER_URL - 1:
                                logging.warning(f"Decoded text does not contain Chinese characters: {url}")
                                self._log_failed_url(url, date_str, 
                                                'am' if 'am' in url else 'pm', 
                                                "No Chinese characters in content")
                            await asyncio.sleep(1)
                            continue
                        
                        soup = BeautifulSoup(text, 'html.parser')
                        news_data = []

                        # Convert YYYYMMDD format to YYYY-MM-DD format
                        formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"

                        # Parse all news blocks
                        for block_id in self.config.NEWS_BLOCKS:
                            if block := soup.find('div', id=block_id):
                                block_news = self._parse_news_block(block, block_id, formatted_date)
                                if block_news:
                                    # Verify if title of each news contains valid Chinese characters
                                    valid_news = [
                                        news for news in block_news 
                                        if re.search('[\u4e00-\u9fff]', news.get('title', ''))
                                    ]
                                    news_data.extend(valid_news)
                                    
                                    if len(valid_news) < len(block_news):
                                        logging.warning(f"Found {len(block_news) - len(valid_news)} invalid titles")

                        self.request_metrics['fetch_success'] += 1
                        await asyncio.sleep(random.uniform(1, 3))
                        return news_data
                
                except asyncio.TimeoutError:
                    self.request_metrics['timeout'] += 1
                    if attempt == self.config.MAX_RETRIES_PER_URL - 1:
                        self._log_failed_url(url, date_str, 
                                        'am' if 'am' in url else 'pm', 
                                        "Timeout")
                    await asyncio.sleep(1)
                except Exception as e:
                    self.request_metrics['request_error'] += 1
                    if attempt == self.config.MAX_RETRIES_PER_URL - 1:
                        logging.error(f"Failed to get URL {url}: {str(e)}")
                        self._log_failed_url(url, date_str, 
                                        'am' if 'am' in url else 'pm', 
                                        f"Request error: {str(e)}")
                    await asyncio.sleep(1)

        except Exception as e:
            logging.error(f"Failed to get URL {url}: {str(e)}")
            self._log_failed_url(url, date_str, 
                             'am' if 'am' in url else 'pm', 
                             f"Request error: {str(e)}")
        
        return None

    def _parse_news_block(self, block: BeautifulSoup, block_id: str, date_str: str) -> List[Dict]:
        """Parse news block"""
        news_data = []
        
        try:
            # Handle different block types
            if block_id == "blk_yw_01":  # Headlines block
                for link in block.find_all('a', href=True):
                    title = self._clean_text(link.get_text())
                    if not title or not re.search('[\u4e00-\u9fff]', title):
                        continue
                        
                    news_data.append({
                        'title': title,
                        'link': link['href'],
                        'category': self.config.NEWS_BLOCKS[block_id],
                        'news_date': date_str,
                        'fetch_date': datetime.datetime.now()
                    })
            else:
                # Handle other blocks
                for link in block.find_all('a', href=True):
                    title = self._clean_text(link.get_text())
                    if not title or not re.search('[\u4e00-\u9fff]', title):
                        continue
                        
                    news_data.append({
                        'title': title,
                        'link': link['href'],
                        'category': self.config.NEWS_BLOCKS[block_id],
                        'news_date': date_str,
                        'fetch_date': datetime.datetime.now()
                    })
                    
        except Exception as e:
            logging.error(f"Failed to parse block {block_id}: {e}")
            
        return news_data

    def _log_failed_url(self, url: str, date: str, period: str, reason: str):
        """Record failed URL in memory and flush once at shutdown"""
        failed_data = {
            "url": url,
            "date": date,
            "period": period,
            "reason": reason,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.failed_url_records.append(failed_data)
        self.request_metrics['failed_urls_logged'] += 1
        self.failed_reason_counter[reason] += 1
        logging.info(f"Recorded failed URL: {url} ({date} {period})")

    def flush_failed_urls_file(self):
        """Persist failed URL records to JSON file once, minimizing blocking file I/O."""
        if not self.failed_url_records:
            return

        try:
            if Path(self.config.FAILED_URLS_FILE).exists():
                with open(self.config.FAILED_URLS_FILE, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
            else:
                existing = []
        except json.JSONDecodeError:
            existing = []

        existing.extend(self.failed_url_records)
        with open(self.config.FAILED_URLS_FILE, 'w', encoding='utf-8') as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)

        logging.info(f"Persisted {len(self.failed_url_records)} failed URL records")

    async def process_date(self, date: datetime.date) -> List[Dict]:
        date_str = date.strftime("%Y%m%d")
        urls = [
            self.config.BASE_URL.format(YYYYMMDD=date_str, AMPM=period)
            for period in ["am", "pm"]
        ]
        responses = await asyncio.gather(
            *(self.fetch_news(url, date_str) for url in urls),
            return_exceptions=True
        )

        day_news = []
        for response in responses:
            if isinstance(response, Exception):
                logging.error(f"Error while processing {date_str}: {response}")
                continue
            if response:
                day_news.extend(response)

        return day_news

    async def save_to_mongo(self, year: int, news_batch: List[Dict]):
        """Save news data to collection without suffix"""
        if not news_batch:
            return 0, 0, 0  # Return three zeros: total document count, new document count, duplicate document count

        # Use collection name without suffix
        collection = self.db[str(year)]  # For example: "2014" instead of "2014_1"

        # Ensure efficient upsert key once per collection.
        if year not in self.indexed_collections:
            await asyncio.to_thread(
                collection.create_index,
                [('link', ASCENDING), ('news_date', ASCENDING)],
                unique=True,
                name='uniq_link_news_date'
            )
            self.indexed_collections.add(year)
        
        operations = [
            UpdateOne(
                {'link': news['link'], 'news_date': news['news_date']},  # Match both link and news_date
                {'$set': news},
                upsert=True
            ) for news in news_batch
        ]
        
        try:
            result = await asyncio.to_thread(
                collection.bulk_write,
                operations,
                ordered=False
            )

            # Calculate new and duplicate document counts from write result.
            new_docs = result.upserted_count
            duplicate_docs = len(operations) - new_docs
            after_count = collection.count_documents({})
            
            logging.info(f"Saved {len(operations)} news to {year} collection")
            logging.info(f"New documents: {new_docs} documents")
            logging.info(f"Duplicate documents: {duplicate_docs} documents")
            logging.info(f"Current collection document count: {after_count} documents")
            
            return after_count, new_docs, duplicate_docs
            
        except Exception as e:
            logging.error(f"Failed to save to MongoDB: {e}")
            after_count = collection.count_documents({})
            return after_count, 0, 0

async def print_date_statistics(db: MongoClient):
    """Count documents for each date"""
    logging.info("\n=== News Date Statistics ===")
    
    # Iterate through all collections (years)
    for collection_name in db.list_collection_names():
        if not collection_name.isdigit():  # Skip non-year collections
            continue
            
        collection = db[collection_name]
        
        # Group by date and count
        pipeline = [
            {
                "$group": {
                    "_id": "$news_date",
                    "count": {"$sum": 1},
                    "categories": {"$addToSet": "$category"}
                }
            },
            {"$sort": {"_id": 1}}  # Sort by date
        ]
        
        results = collection.aggregate(pipeline)
        
        logging.info(f"\nYear: {collection_name}")
        logging.info("-" * 30)
        
        for result in results:
            date = result["_id"]
            count = result["count"]
            categories = result["categories"]
            
            logging.info(f"Date: {date}")
            logging.info(f"Total documents: {count}")
            logging.info(f"Contains categories: {', '.join(categories)}")
            logging.info("-" * 20)

async def clear_collections(db: MongoClient, start_year: int, end_year: int):
    """Clear all collections in specified year range (using collection without suffix)"""
    logging.info("\n=== Clear Collections ===")
    for year in range(start_year, end_year + 1):
        collection_name = str(year)  # Use collection name without suffix
        if collection_name in db.list_collection_names():
            result = db[collection_name].delete_many({})
            logging.info(f"Cleared {year} year collection, deleted {result.deleted_count} documents")
        else:
            logging.info(f"{year} year collection does not exist, no need to clear")
    logging.info("Collection clear completed\n")


def load_checkpoint(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        return None

    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logging.warning(f"Failed to load checkpoint file {path}: {e}")
        return None


def save_checkpoint(path: str, data: Dict):
    tmp_path = f"{path}.tmp"
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def _format_elapsed(seconds: float) -> str:
    sec = max(0, int(seconds))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def save_status(path: str, data: Dict):
    tmp_path = f"{path}.tmp"
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)

async def main():
    # Create configuration from command line arguments
    config = Config.from_args()
    run_started_at = datetime.datetime.now()
    run_id = run_started_at.strftime('%Y%m%d_%H%M%S')
    
    # Get list of dates to crawl
    original_date_list = config.get_date_list()
    date_list = list(original_date_list)
    total_dates_planned = len(original_date_list)

    checkpoint = load_checkpoint(config.CHECKPOINT_FILE) if config.RESUME_FROM_CHECKPOINT else None
    resumed_from_checkpoint = False
    resumed_last_completed = None
    completed_before_run = 0

    if checkpoint:
        same_job = (
            checkpoint.get('db_name') == config.DB_NAME
            and checkpoint.get('start_date') == config.START_DATE
            and checkpoint.get('end_date') == config.END_DATE
        )
        last_completed = checkpoint.get('last_completed_date')

        if same_job and last_completed:
            try:
                last_completed_date = datetime.datetime.strptime(last_completed, "%Y-%m-%d").date()
                date_list = [d for d in original_date_list if d > last_completed_date]
                resumed_from_checkpoint = True
                resumed_last_completed = last_completed
                completed_before_run = total_dates_planned - len(date_list)
                logging.info(
                    f"Resume from checkpoint enabled, last completed date: {last_completed}, "
                    f"remaining dates: {len(date_list)}/{total_dates_planned}"
                )
            except ValueError:
                logging.warning("Checkpoint last_completed_date is invalid, start from beginning")
    
    if not date_list:
        logging.info("No remaining dates to crawl. The range is already completed.")
        return
    
    # Initialize crawler
    spider = NewsSpider(config)
    await spider._init_session()
    
    try:
        # Get all years in date range
        years = set(date.year for date in original_date_list)
        
        # Clear related collections only when explicitly requested.
        if config.CLEAR_EXISTING:
            await clear_collections(spider.db, min(years), max(years))
        else:
            logging.info("Skip collection clearing (safe mode). Use --clear-existing to enable destructive cleanup.")
        
        news_batch = []
        current_year = None
        
        # Add document counter
        total_documents = 0
        processed_dates = 0
        total_new_docs = 0
        total_duplicate_docs = 0

        checkpoint_data = {
            'run_id': run_id,
            'status': 'in_progress',
            'db_name': config.DB_NAME,
            'start_date': config.START_DATE,
            'end_date': config.END_DATE,
            'summary_file': config.SUMMARY_FILE,
            'checkpoint_file': config.CHECKPOINT_FILE,
            'last_completed_date': resumed_last_completed,
            'processed_dates': 0,
            'total_dates': total_dates_planned,
            'updated_at': datetime.datetime.now().isoformat()
        }
        save_checkpoint(config.CHECKPOINT_FILE, checkpoint_data)

        status_data = {
            'run_id': run_id,
            'status': 'running',
            'pid': os.getpid(),
            'db_name': config.DB_NAME,
            'start_date': config.START_DATE,
            'end_date': config.END_DATE,
            'current_processing_date': None,
            'last_completed_date': resumed_last_completed,
            'processed_dates': completed_before_run,
            'total_dates': total_dates_planned,
            'remaining_dates': total_dates_planned - completed_before_run,
            'progress_percent': round((completed_before_run / total_dates_planned) * 100, 2) if total_dates_planned else 0,
            'progress_text': f"{completed_before_run}/{total_dates_planned}",
            'elapsed': '00:00:00',
            'speed_dates_per_min': 0,
            'updated_at': datetime.datetime.now().isoformat(),
            'files': {
                'checkpoint_file': config.CHECKPOINT_FILE,
                'summary_file': config.SUMMARY_FILE,
                'failed_urls_file': config.FAILED_URLS_FILE
            },
            'last_error': None
        }
        save_status(config.STATUS_FILE, status_data)
        
        # Iterate through date list with tqdm progress bar
        with tqdm(total=len(date_list), desc='Processing dates', unit='date') as date_bar:
            for current_date in date_list:
                # Show progress
                processed_dates += 1
                absolute_processed = completed_before_run + processed_dates
                logging.info(f"Processing date {current_date} ({absolute_processed}/{total_dates_planned})")
                date_bar.update(1)

                # If year changes, save previous batch
                if current_year and current_year != current_date.year and news_batch:
                    doc_count, new_docs, duplicate_docs = await spider.save_to_mongo(current_year, news_batch)
                    total_documents = doc_count
                    total_new_docs += new_docs
                    total_duplicate_docs += duplicate_docs
                    logging.info(f"Year changed, current database has {total_documents} documents")
                    news_batch = []

                current_year = current_date.year
                date_news = await spider.process_date(current_date)

                if date_news:
                    news_batch.extend(date_news)
                    logging.info(f"Got {len(date_news)} news from {current_date}")
                else:
                    logging.warning(f"No news got from {current_date}")

                # If batch reaches specified size, save to database
                if len(news_batch) >= config.BATCH_SIZE:
                    doc_count, new_docs, duplicate_docs = await spider.save_to_mongo(current_year, news_batch)
                    total_documents = doc_count
                    total_new_docs += new_docs
                    total_duplicate_docs += duplicate_docs
                    logging.info(f"Batch saved, current database has {total_documents} documents")
                    news_batch = []

                # Show progress
                logging.info(f"---------------------Completed processing date: {current_date}---------------------")

                now = datetime.datetime.now()
                elapsed_seconds = (now - run_started_at).total_seconds()
                speed_dates_per_min = round((absolute_processed / (elapsed_seconds / 60)), 2) if elapsed_seconds > 0 else 0
                progress_percent = round((absolute_processed / total_dates_planned) * 100, 2) if total_dates_planned else 0

                checkpoint_data['last_completed_date'] = current_date.strftime('%Y-%m-%d')
                checkpoint_data['processed_dates'] = absolute_processed
                checkpoint_data['updated_at'] = now.isoformat()
                save_checkpoint(config.CHECKPOINT_FILE, checkpoint_data)

                status_data.update({
                    'status': 'running',
                    'current_processing_date': current_date.strftime('%Y-%m-%d'),
                    'last_completed_date': current_date.strftime('%Y-%m-%d'),
                    'processed_dates': absolute_processed,
                    'remaining_dates': total_dates_planned - absolute_processed,
                    'progress_percent': progress_percent,
                    'progress_text': f"{absolute_processed}/{total_dates_planned}",
                    'elapsed': _format_elapsed(elapsed_seconds),
                    'speed_dates_per_min': speed_dates_per_min,
                    'updated_at': now.isoformat(),
                    'last_error': None
                })
                save_status(config.STATUS_FILE, status_data)
            
        # Save last batch data
        if news_batch:
            doc_count, new_docs, duplicate_docs = await spider.save_to_mongo(current_year, news_batch)
            total_documents = doc_count
            total_new_docs += new_docs
            total_duplicate_docs += duplicate_docs
            logging.info(f"Final saved, database has {total_documents} documents")

        checkpoint_data['status'] = 'completed'
        checkpoint_data['processed_dates'] = total_dates_planned
        checkpoint_data['updated_at'] = datetime.datetime.now().isoformat()
        save_checkpoint(config.CHECKPOINT_FILE, checkpoint_data)

        status_data.update({
            'status': 'completed',
            'current_processing_date': None,
            'last_completed_date': checkpoint_data.get('last_completed_date'),
            'processed_dates': total_dates_planned,
            'remaining_dates': 0,
            'progress_percent': 100,
            'progress_text': f"{total_dates_planned}/{total_dates_planned}",
            'elapsed': _format_elapsed((datetime.datetime.now() - run_started_at).total_seconds()),
            'updated_at': datetime.datetime.now().isoformat(),
            'last_error': None
        })
        save_status(config.STATUS_FILE, status_data)
            
    except Exception as e:
        logging.error(f"Error occurred during processing: {e}")
        checkpoint_data['status'] = 'failed'
        checkpoint_data['updated_at'] = datetime.datetime.now().isoformat()
        save_checkpoint(config.CHECKPOINT_FILE, checkpoint_data)

        status_data.update({
            'status': 'failed',
            'updated_at': datetime.datetime.now().isoformat(),
            'last_error': str(e),
            'elapsed': _format_elapsed((datetime.datetime.now() - run_started_at).total_seconds())
        })
        save_status(config.STATUS_FILE, status_data)
    finally:
        if spider.session:
            await spider.session.close()

        spider.flush_failed_urls_file()
        
        # Save failed URL records
        if spider.failed_urls:
            spider.db.failed_urls.insert_many(spider.failed_urls)
            logging.info(f"Recorded {len(spider.failed_urls)} failed URLs")
        
        # Output document statistics for all years
        logging.info("\n=== All Year Document Statistics ===")
        for year in years:
            collection = spider.db[str(year)]
            doc_count = collection.count_documents({})
            logging.info(f"{year} year: {doc_count} documents")
        
        # Output date statistics
        await print_date_statistics(spider.db)

        run_finished_at = datetime.datetime.now()
        elapsed_seconds = (run_finished_at - run_started_at).total_seconds()
        summary = {
            'run_id': run_id,
            'run_started_at': run_started_at.isoformat(),
            'run_finished_at': run_finished_at.isoformat(),
            'elapsed_seconds': elapsed_seconds,
            'config': {
                'db_name': config.DB_NAME,
                'start_date': config.START_DATE,
                'end_date': config.END_DATE,
                'batch_size': config.BATCH_SIZE,
                'concurrent_requests': config.CONCURRENT_REQUESTS,
                'clear_existing': config.CLEAR_EXISTING,
                'checkpoint_file': config.CHECKPOINT_FILE,
                'status_file': config.STATUS_FILE,
                'resume_from_checkpoint': config.RESUME_FROM_CHECKPOINT,
                'resumed_from_checkpoint': resumed_from_checkpoint,
                'excel_file': config.EXCEL_FILE,
                'date_column': config.DATE_COLUMN
            },
            'dates': {
                'total_dates': total_dates_planned,
                'processed_dates': processed_dates,
                'remaining_dates': len(date_list) - processed_dates
            },
            'documents': {
                'total_new_documents': total_new_docs,
                'total_duplicate_documents': total_duplicate_docs
            },
            'requests': spider.request_metrics,
            'failed_url_records': len(spider.failed_url_records),
            'failed_reasons': dict(spider.failed_reason_counter)
        }

        with open(config.SUMMARY_FILE, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logging.info(f"Wrote structured run summary to {config.SUMMARY_FILE}")

if __name__ == "__main__":
    asyncio.run(main())