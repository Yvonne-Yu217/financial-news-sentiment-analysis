import argparse
import datetime
import json
import logging
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
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


def format_eta(seconds: float) -> str:
    if seconds <= 0:
        return "00:00:00"
    total = int(seconds)
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze image clarity distribution from {year}_filtered collections")
    parser.add_argument("--mongo-uri", type=str, default=MONGO_URI, help="MongoDB URI")
    parser.add_argument("--db-name", type=str, default=DB_NAME, help="MongoDB database name")
    parser.add_argument("--start-year", type=int, default=START_YEAR, help="Start year")
    parser.add_argument("--end-year", type=int, default=END_YEAR, help="End year")
    parser.add_argument("--years", type=str, default="", help="Optional comma-separated years, e.g. 2020,2021")
    parser.add_argument("--sample-ratio", type=float, default=SAMPLE_RATIO, help="Sampling ratio in (0, 1]")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS, help="Parallel workers for image processing")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for reproducible sampling")
    parser.add_argument("--examples", type=int, default=5, help="Number of random examples per clarity bucket")
    parser.add_argument("--eta-log-interval", type=int, default=30, help="Log ETA every N seconds during heavy stages")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    parser.add_argument("--output-prefix", type=str, default="clarity", help="Output file prefix")
    args = parser.parse_args()

    if args.sample_ratio <= 0 or args.sample_ratio > 1:
        raise ValueError("--sample-ratio must be in (0, 1]")
    if args.num_workers < 1:
        raise ValueError("--num-workers must be >= 1")
    if args.eta_log_interval < 1:
        raise ValueError("--eta-log-interval must be >= 1")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(out_dir)

    return args


def build_years(args: argparse.Namespace) -> List[int]:
    if args.years.strip():
        return sorted({int(y.strip()) for y in args.years.split(",") if y.strip()})
    return list(range(args.start_year, args.end_year + 1))


def calculate_blur_score(image_path: str) -> float:
    try:
        cv_image = cv2.imread(image_path)
        if cv_image is None:
            return 0.0
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY) if len(cv_image.shape) == 3 else cv_image
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except Exception:
        return 0.0


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

        # Composite signal is more robust than blur-only scoring.
        clarity_signal = 0.6 * np.log1p(blur) + 0.3 * np.log1p(max(contrast, 0.0)) + 0.1 * (edge_density * 100.0)
        return {
            "blur": blur,
            "contrast": contrast,
            "edge_density": edge_density,
            "clarity_signal": float(clarity_signal),
        }
    except Exception:
        return {"blur": 0.0, "contrast": 0.0, "edge_density": 0.0, "clarity_signal": 0.0}


class ClarityAnalyzer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.client = MongoClient(args.mongo_uri)
        self.db = self.client[args.db_name]
        self.records: List[Dict[str, Any]] = []
        self.sampled_records: List[Dict[str, Any]] = []
        self.blur_scores: List[float] = []
        self.contrast_scores: List[float] = []
        self.edge_density_scores: List[float] = []
        self.clarity_signals: List[float] = []

    def fetch_filtered_images(self, years: List[int]) -> None:
        records: List[Dict[str, Any]] = []
        seen_paths = set()

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
                    if abs_path and abs_path not in seen_paths and Path(abs_path).exists():
                        seen_paths.add(abs_path)
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
            self.blur_scores = []
            return

        total = len(self.sampled_records)
        feature_list: List[Dict[str, float]] = [{"blur": 0.0, "contrast": 0.0, "edge_density": 0.0, "clarity_signal": 0.0} for _ in range(total)]
        started = datetime.datetime.now()
        last_log = started

        with ThreadPoolExecutor(max_workers=self.args.num_workers) as executor:
            futures = {
                executor.submit(calculate_clarity_features, rec["abs_path"]): idx
                for idx, rec in enumerate(self.sampled_records)
            }
            with tqdm(total=total, desc="Calculating clarity", unit="img") as pbar:
                completed = 0
                for future in as_completed(futures):
                    idx = futures[future]
                    feature_list[idx] = future.result()
                    completed += 1
                    pbar.update(1)

                    now = datetime.datetime.now()
                    if (now - last_log).total_seconds() >= self.args.eta_log_interval or completed == total:
                        elapsed = (now - started).total_seconds()
                        rate = completed / max(elapsed, 1e-6)
                        remaining = max(total - completed, 0)
                        eta_seconds = remaining / max(rate, 1e-6)
                        logging.info(
                            "[ETA][clarity] %d/%d (%.2f%%) | rate=%.2f img/s | elapsed=%s | eta=%s",
                            completed,
                            total,
                            (completed / total) * 100.0,
                            rate,
                            format_eta(elapsed),
                            format_eta(eta_seconds),
                        )
                        last_log = now

        filtered_records: List[Dict[str, Any]] = []
        filtered_blur: List[float] = []
        filtered_contrast: List[float] = []
        filtered_edge_density: List[float] = []
        filtered_signals: List[float] = []
        for rec, features in zip(self.sampled_records, feature_list):
            signal = features.get("clarity_signal", 0.0)
            if signal > 0:
                filtered_records.append(rec)
                filtered_blur.append(features.get("blur", 0.0))
                filtered_contrast.append(features.get("contrast", 0.0))
                filtered_edge_density.append(features.get("edge_density", 0.0))
                filtered_signals.append(signal)

        self.sampled_records = filtered_records
        self.blur_scores = filtered_blur
        self.contrast_scores = filtered_contrast
        self.edge_density_scores = filtered_edge_density
        self.clarity_signals = filtered_signals
        logging.info(f"Analyzed {len(feature_list)} sampled images, {len(self.clarity_signals)} valid clarity signals")

    def calculate_statistics(self) -> Dict[str, Any]:
        if not self.clarity_signals:
            return {}

        signal_arr = np.array(self.clarity_signals, dtype=float)
        blur_arr = np.array(self.blur_scores, dtype=float)
        contrast_arr = np.array(self.contrast_scores, dtype=float)
        edge_arr = np.array(self.edge_density_scores, dtype=float)
        return {
            "count": int(signal_arr.size),
            "clarity_signal": {
                "min": float(np.min(signal_arr)),
                "max": float(np.max(signal_arr)),
                "mean": float(np.mean(signal_arr)),
                "median": float(np.median(signal_arr)),
                "std": float(np.std(signal_arr)),
            },
            "blur": {
                "mean": float(np.mean(blur_arr)),
                "median": float(np.median(blur_arr)),
            },
            "contrast": {
                "mean": float(np.mean(contrast_arr)),
                "median": float(np.median(contrast_arr)),
            },
            "edge_density": {
                "mean": float(np.mean(edge_arr)),
                "median": float(np.median(edge_arr)),
            },
            "percentiles": {
                "10%": float(np.percentile(signal_arr, 10)),
                "25%": float(np.percentile(signal_arr, 25)),
                "50%": float(np.percentile(signal_arr, 50)),
                "75%": float(np.percentile(signal_arr, 75)),
                "90%": float(np.percentile(signal_arr, 90)),
                "95%": float(np.percentile(signal_arr, 95)),
                "99%": float(np.percentile(signal_arr, 99)),
            },
        }

    def suggest_thresholds(self, stats: Dict[str, Any]) -> Dict[str, float]:
        if not stats:
            return {}
        p = stats["percentiles"]
        return {
            "low_clarity": p["25%"],
            "medium_clarity": p["75%"],
            "high_clarity": p["95%"],
        }

    def suggest_weight_params(self, stats: Dict[str, Any]) -> Dict[str, float]:
        if not stats:
            return {}

        p = stats["percentiles"]
        midpoint = p["50%"]
        span = max(1e-6, p["75%"] - p["25%"])
        steepness = max(0.01, min(1.0 / span, 0.1))

        return {
            "function_type": "logistic",
            "midpoint": float(midpoint),
            "steepness": float(steepness),
            "min_weight": 0.1,
            "max_weight": 1.0,
        }

    def show_examples(self, thresholds: Dict[str, float]) -> None:
        if not thresholds or not self.clarity_signals:
            return

        buckets = {"low": [], "medium": [], "high": []}
        for rec, score in zip(self.sampled_records, self.clarity_signals):
            item = (rec["abs_path"], rec["news_date"], score)
            if score < thresholds["low_clarity"]:
                buckets["low"].append(item)
            elif score < thresholds["medium_clarity"]:
                buckets["medium"].append(item)
            else:
                buckets["high"].append(item)

        random.seed(self.args.random_seed)
        for name, items in buckets.items():
            if not items:
                continue
            samples = random.sample(items, min(self.args.examples, len(items)))
            logging.info(f"[{name}] examples ({len(items)} total):")
            for path, date, score in samples:
                logging.info(f"  {date} | {score:.2f} | {path}")

    def plot_distribution(self, stats: Dict[str, Any]) -> None:
        if not stats or not self.clarity_signals:
            return

        output_dir = Path(self.args.output_dir)
        plot_path = output_dir / f"{self.args.output_prefix}_score_distribution.png"
        arr = np.array(self.clarity_signals, dtype=float)
        x_limit = float(np.percentile(arr, 99.5))
        filtered = arr[arr <= x_limit]

        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.hist(filtered, bins=50, alpha=0.75, color="royalblue", edgecolor="black")
        plt.title("Composite Clarity Signal Histogram")
        plt.xlabel("clarity signal")
        plt.ylabel("count")
        plt.grid(alpha=0.3)

        plt.subplot(2, 1, 2)
        if gaussian_kde is not None and len(filtered) > 1:
            xs = np.linspace(0, max(1.0, x_limit), 300)
            kde = gaussian_kde(filtered)
            plt.plot(xs, kde(xs), "r-", linewidth=2)
        else:
            plt.hist(filtered, bins=50, alpha=0.6, color="tomato", edgecolor="black", density=True)
        plt.title("Composite Clarity Signal Density")
        plt.xlabel("clarity signal")
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
            "def calculate_clarity_weight(clarity_signal):\n"
            f"    midpoint = {params['midpoint']:.6f}\n"
            f"    steepness = {params['steepness']:.6f}\n"
            f"    min_weight = {params['min_weight']:.6f}\n"
            f"    max_weight = {params['max_weight']:.6f}\n"
            "    return min_weight + (max_weight - min_weight) / (1 + np.exp(-steepness * (clarity_signal - midpoint)))\n"
        )
        out_path.write_text(code, encoding="utf-8")
        return out_path

    def run(self) -> Dict[str, Any]:
        started = datetime.datetime.now()
        years = build_years(self.args)
        logging.info(f"Running clarity helper for years: {years}")

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
            },
            "image_counts": {
                "total_collected": len(self.records),
                "sampled": len(self.sampled_records),
                "with_valid_scores": len(self.clarity_signals),
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
    analyzer = ClarityAnalyzer(args)
    analyzer.run()


if __name__ == "__main__":
    main()
