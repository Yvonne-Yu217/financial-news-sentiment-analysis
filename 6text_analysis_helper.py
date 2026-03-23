import argparse
import datetime
import json
import logging
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pytesseract
from PIL import Image
from pymongo import MongoClient
from tqdm import tqdm

try:
    from scipy.stats import gaussian_kde
except Exception:  # pragma: no cover
    gaussian_kde = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

START_YEAR = 2014
END_YEAR = 2026
DB_NAME = "sina_news_dataset_test"
MONGO_URI = "mongodb://localhost:27017/"
SAMPLE_RATIO = 0.1
NUM_WORKERS = 4

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPT_STEM = Path(__file__).stem
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "run_artifacts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze OCR text-length distribution from {year}_filtered collections")
    parser.add_argument("--mongo-uri", type=str, default=MONGO_URI, help="MongoDB URI")
    parser.add_argument("--db-name", type=str, default=DB_NAME, help="MongoDB database name")
    parser.add_argument("--start-year", type=int, default=START_YEAR, help="Start year")
    parser.add_argument("--end-year", type=int, default=END_YEAR, help="End year")
    parser.add_argument("--years", type=str, default="", help="Optional comma-separated years, e.g. 2020,2021")
    parser.add_argument("--sample-ratio", type=float, default=SAMPLE_RATIO, help="Sampling ratio in (0, 1]")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS, help="Parallel workers for OCR")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for reproducible sampling")
    parser.add_argument("--min-text-length", type=int, default=1, help="Minimum OCR text length kept for statistics")
    parser.add_argument("--ocr-lang", type=str, default="chi_sim+eng", help="Tesseract OCR language")
    parser.add_argument("--examples", type=int, default=5, help="Number of random examples per text bucket")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    parser.add_argument("--output-prefix", type=str, default="text", help="Output file prefix")
    args = parser.parse_args()

    if args.sample_ratio <= 0 or args.sample_ratio > 1:
        raise ValueError("--sample-ratio must be in (0, 1]")
    if args.num_workers < 1:
        raise ValueError("--num-workers must be >= 1")
    if args.min_text_length < 0:
        raise ValueError("--min-text-length must be >= 0")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(out_dir)

    return args


def build_years(args: argparse.Namespace) -> List[int]:
    if args.years.strip():
        return sorted({int(y.strip()) for y in args.years.split(",") if y.strip()})
    return list(range(args.start_year, args.end_year + 1))


def extract_text(image_path: str, lang: str) -> str:
    try:
        with Image.open(image_path) as img:
            text = pytesseract.image_to_string(img, lang=lang)
        return " ".join(text.split())
    except Exception:
        return ""


class TextAnalyzer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.client = MongoClient(args.mongo_uri)
        self.db = self.client[args.db_name]
        self.records: List[Dict[str, Any]] = []
        self.sampled_records: List[Dict[str, Any]] = []
        self.text_lengths: List[int] = []
        self.text_contents: List[str] = []

    def fetch_filtered_images(self, years: List[int]) -> None:
        records: List[Dict[str, Any]] = []

        for year in years:
            collection_name = f"{year}_filtered"
            if collection_name not in self.db.list_collection_names():
                logging.warning(f"Collection missing: {collection_name}")
                continue

            collection = self.db[collection_name]
            cursor = collection.find(
                {"basic_filtered": True, "has_valid_images": True},
                {"news_date": 1, "link": 1, "valid_images": 1},
            )

            for doc in cursor:
                news_date = str(doc.get("news_date") or f"{year}-01-01")
                link = str(doc.get("link") or "")
                for img in doc.get("valid_images", []):
                    abs_path = str(img.get("abs_path") or "").strip()
                    if abs_path and Path(abs_path).exists():
                        records.append(
                            {
                                "year": year,
                                "news_date": news_date,
                                "link": link,
                                "abs_path": abs_path,
                            }
                        )

        self.records = records
        logging.info(f"Collected {len(self.records)} valid images from filtered collections")

    def stratified_sample(self) -> None:
        if not self.records:
            self.sampled_records = []
            return

        random.seed(self.args.random_seed)
        target = max(1, int(len(self.records) * self.args.sample_ratio))

        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for item in self.records:
            grouped[item["news_date"]].append(item)

        sampled: List[Dict[str, Any]] = []
        for date, items in grouped.items():
            ratio_count = int(round(len(items) / len(self.records) * target))
            ratio_count = min(max(ratio_count, 1), len(items))
            sampled.extend(random.sample(items, ratio_count))

        if len(sampled) > target:
            sampled = random.sample(sampled, target)
        elif len(sampled) < target:
            leftovers = [r for r in self.records if r not in sampled]
            need = min(len(leftovers), target - len(sampled))
            if need > 0:
                sampled.extend(random.sample(leftovers, need))

        self.sampled_records = sampled
        logging.info(f"Sampled {len(self.sampled_records)} images (target={target})")

    def analyze(self) -> None:
        if not self.sampled_records:
            self.text_lengths = []
            self.text_contents = []
            return

        with ThreadPoolExecutor(max_workers=self.args.num_workers) as executor:
            texts = list(
                tqdm(
                    executor.map(lambda r: extract_text(r["abs_path"], self.args.ocr_lang), self.sampled_records),
                    total=len(self.sampled_records),
                    desc="Extracting OCR text",
                    unit="img",
                )
            )

        filtered_records: List[Dict[str, Any]] = []
        filtered_lengths: List[int] = []
        filtered_texts: List[str] = []
        for rec, text in zip(self.sampled_records, texts):
            if len(text) >= self.args.min_text_length:
                filtered_records.append(rec)
                filtered_lengths.append(len(text))
                filtered_texts.append(text)

        self.sampled_records = filtered_records
        self.text_lengths = filtered_lengths
        self.text_contents = filtered_texts
        logging.info(f"Analyzed {len(texts)} sampled images, {len(self.text_lengths)} kept after text-length filter")

    def calculate_statistics(self) -> Dict[str, Any]:
        if not self.text_lengths:
            return {}

        arr = np.array(self.text_lengths, dtype=float)
        return {
            "count": int(arr.size),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr)),
            "percentiles": {
                "10%": float(np.percentile(arr, 10)),
                "25%": float(np.percentile(arr, 25)),
                "50%": float(np.percentile(arr, 50)),
                "75%": float(np.percentile(arr, 75)),
                "90%": float(np.percentile(arr, 90)),
                "95%": float(np.percentile(arr, 95)),
                "99%": float(np.percentile(arr, 99)),
            },
        }

    def suggest_thresholds(self, stats: Dict[str, Any]) -> Dict[str, float]:
        if not stats:
            return {}
        p = stats["percentiles"]
        return {
            "low_text": p["25%"],
            "medium_text": p["75%"],
            "high_text": p["95%"],
        }

    def suggest_weight_params(self, stats: Dict[str, Any]) -> Dict[str, float]:
        if not stats:
            return {}

        p = stats["percentiles"]
        midpoint = p["95%"]
        span = max(1e-6, p["95%"] - p["90%"])
        steepness = max(0.01, min(1.0 / span, 0.1))

        return {
            "function_type": "logistic",
            "midpoint": float(midpoint),
            "steepness": float(steepness),
            "min_weight": 0.1,
            "max_weight": 1.0,
        }

    def show_examples(self, thresholds: Dict[str, float]) -> None:
        if not thresholds or not self.text_lengths:
            return

        buckets = {"none": [], "low": [], "medium": [], "high": []}
        for rec, length, text in zip(self.sampled_records, self.text_lengths, self.text_contents):
            item = (rec["abs_path"], rec["news_date"], length, text)
            if length == 0:
                buckets["none"].append(item)
            elif length < thresholds["low_text"]:
                buckets["low"].append(item)
            elif length < thresholds["medium_text"]:
                buckets["medium"].append(item)
            else:
                buckets["high"].append(item)

        random.seed(self.args.random_seed)
        for name, items in buckets.items():
            if not items:
                continue
            samples = random.sample(items, min(self.args.examples, len(items)))
            logging.info(f"[{name}] examples ({len(items)} total):")
            for path, date, length, text in samples:
                snippet = text[:100] + ("..." if len(text) > 100 else "")
                logging.info(f"  {date} | len={length} | {path}")
                logging.info(f"    text: {snippet}")

    def plot_distribution(self, stats: Dict[str, Any]) -> None:
        if not stats or not self.text_lengths:
            return

        output_dir = Path(self.args.output_dir)
        plot_path = output_dir / f"{self.args.output_prefix}_length_distribution.png"

        arr = np.array(self.text_lengths, dtype=float)
        x_limit = float(np.percentile(arr, 99.5))
        filtered = arr[arr <= x_limit]

        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.hist(filtered, bins=50, alpha=0.75, color="royalblue", edgecolor="black")
        plt.title("OCR Text Length Histogram")
        plt.xlabel("text length")
        plt.ylabel("count")
        plt.grid(alpha=0.3)

        plt.subplot(2, 1, 2)
        if gaussian_kde is not None and len(filtered) > 1:
            xs = np.linspace(0, max(1.0, x_limit), 300)
            kde = gaussian_kde(filtered)
            plt.plot(xs, kde(xs), "r-", linewidth=2)
        else:
            plt.hist(filtered, bins=50, alpha=0.6, color="tomato", edgecolor="black", density=True)
        plt.title("OCR Text Length Density")
        plt.xlabel("text length")
        plt.ylabel("density")
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        logging.info(f"Saved plot: {plot_path}")

    def write_weight_code(self, params: Dict[str, float]) -> Path:
        out_path = Path(self.args.output_dir) / f"{self.args.output_prefix}_weight_function.py"
        if not params:
            out_path.write_text("# No params generated\n", encoding="utf-8")
            return out_path

        code = (
            "import numpy as np\n\n"
            "def calculate_text_weight(text_length):\n"
            f"    midpoint = {params['midpoint']:.6f}\n"
            f"    steepness = {params['steepness']:.6f}\n"
            f"    min_weight = {params['min_weight']:.6f}\n"
            f"    max_weight = {params['max_weight']:.6f}\n"
            "    return max_weight - (max_weight - min_weight) / (1 + np.exp(-steepness * (text_length - midpoint)))\n"
        )
        out_path.write_text(code, encoding="utf-8")
        return out_path

    def run(self) -> Dict[str, Any]:
        started = datetime.datetime.now()
        years = build_years(self.args)
        logging.info(f"Running text helper for years: {years}")

        self.fetch_filtered_images(years)
        self.stratified_sample()
        self.analyze()

        stats = self.calculate_statistics()
        thresholds = self.suggest_thresholds(stats)
        weight_params = self.suggest_weight_params(stats)

        if stats:
            self.plot_distribution(stats)
            self.show_examples(thresholds)

        weight_code_file = self.write_weight_code(weight_params)

        summary = {
            "run_started_at": started.isoformat(),
            "run_finished_at": datetime.datetime.now().isoformat(),
            "config": {
                "db_name": self.args.db_name,
                "years": years,
                "sample_ratio": self.args.sample_ratio,
                "num_workers": self.args.num_workers,
                "random_seed": self.args.random_seed,
                "min_text_length": self.args.min_text_length,
                "ocr_lang": self.args.ocr_lang,
            },
            "image_counts": {
                "total_collected": len(self.records),
                "sampled": len(self.sampled_records),
                "with_valid_text": len(self.text_lengths),
            },
            "statistics": stats,
            "thresholds": thresholds,
            "weight_params": weight_params,
            "weight_code_file": str(weight_code_file),
        }

        summary_path = Path(self.args.output_dir) / f"{self.args.output_prefix}_summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        logging.info(f"Saved summary: {summary_path}")

        return summary


def main() -> None:
    args = parse_args()
    analyzer = TextAnalyzer(args)
    analyzer.run()


if __name__ == "__main__":
    main()
