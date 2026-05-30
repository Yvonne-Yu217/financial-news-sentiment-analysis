import argparse
import datetime
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from bson import ObjectId
from pymongo import MongoClient, UpdateOne
from torchvision import transforms
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ARTIFACTS_DIR = SCRIPT_DIR / "run_artifacts" / "8vit_transferlearning"
DEFAULT_CHECKPOINT_FILE = SCRIPT_DIR / "run_artifacts" / "9sentiment_analyzer_oldmodel" / "sentiment_checkpoint_oldmodel.json"
DEFAULT_OLD_MODEL_PATH = SCRIPT_DIR / "improved_vit_sentiment_model_old.pth"


class ImprovedViTModel(nn.Module):
    """Model definition aligned with script 8 (`8vit_transferlearning.py`)."""

    def __init__(self, model_name: str = "vit_base_patch16_224", num_classes: int = 2, dropout_rate: float = 0.2):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=False)
        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Script 9 (old-model variant): image sentiment inference using improved_vit_sentiment_model_old.pth")
    parser.add_argument("--start-year", type=int, default=2014)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--years", nargs="+", type=int, default=None, help="Optional explicit year list")

    parser.add_argument("--mongo-uri", default="mongodb://localhost:27017/")
    parser.add_argument("--db-name", default="sina_news_dataset_test")

    # Default to the old model; --collection-suffix isolates output from live data
    parser.add_argument("--model", default=str(DEFAULT_OLD_MODEL_PATH), help="Model path (defaults to old model)")
    parser.add_argument("--model-name", default="vit_base_patch16_224", help="timm backbone name")
    parser.add_argument("--dropout-rate", type=float, default=0.2)
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR))
    parser.add_argument("--collection-suffix", default="_old",
                        help="Suffix appended to output collection names (e.g. '_old' → {year}_sentiment_old)")

    parser.add_argument("--batch-write-size", type=int, default=500)
    parser.add_argument("--max-news-per-year", type=int, default=0, help="For smoke test/debug only")
    parser.add_argument("--keep-existing", action="store_true", help="Do not drop year sentiment collections before writing")
    parser.add_argument("--checkpoint-file", default=str(DEFAULT_CHECKPOINT_FILE), help="Checkpoint json path")
    parser.add_argument("--checkpoint-every", type=int, default=20, help="Save checkpoint every N processed docs")
    parser.add_argument("--no-resume", action="store_true", help="Disable checkpoint resume")
    parser.add_argument("--reset-checkpoint", action="store_true", help="Delete checkpoint file before run")
    # Old model is not from the strict artifacts dir — disable strict validation by default
    parser.add_argument("--strict-model-only", action="store_true", default=False, help="Require strict-model artifacts metadata")
    parser.add_argument("--no-strict-model-only", dest="strict_model_only", action="store_false")

    return parser.parse_args()


def load_checkpoint(path: Path) -> dict:
    if not path.exists():
        return {"years": {}}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {"years": {}}
        if "years" not in data or not isinstance(data["years"], dict):
            data["years"] = {}
        return data
    except Exception:
        return {"years": {}}


def save_checkpoint(path: Path, checkpoint: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def resolve_years(args: argparse.Namespace) -> List[int]:
    if args.years:
        return sorted(set(args.years))

    years_env = os.environ.get("YEARS_TO_PROCESS", "").strip()
    if years_env:
        env_years = [int(y.strip()) for y in years_env.split(",") if y.strip()]
        return sorted(set(env_years))

    return list(range(args.start_year, args.end_year + 1))


def resolve_model_path(args: argparse.Namespace) -> Path:
    candidates: List[Path] = []

    if args.model:
        user_model = Path(args.model)
        if not user_model.is_absolute():
            user_model = Path.cwd() / user_model
        candidates.append(user_model)

    # Also try the default old model location as fallback
    candidates.extend(
        [
            DEFAULT_OLD_MODEL_PATH,
            SCRIPT_DIR / "improved_vit_sentiment_model_old.pth",
        ]
    )

    for p in candidates:
        if p.exists() and p.is_file():
            logging.info("Using model: %s", p)
            return p

    checked = "\n".join(f"- {p}" for p in candidates)
    raise FileNotFoundError(f"No usable model file found. Checked:\n{checked}")


def validate_strict_model(args: argparse.Namespace, model_path: Path) -> None:
    if not args.strict_model_only:
        return

    artifacts_dir = Path(args.artifacts_dir)
    if not artifacts_dir.is_absolute():
        artifacts_dir = Path.cwd() / artifacts_dir

    summary_path = artifacts_dir / "8vit_transferlearning_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Strict model validation failed: summary not found: {summary_path}")

    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    use_class_weights = summary.get("use_class_weights", None)
    if use_class_weights is None:
        use_class_weights = (summary.get("training") or {}).get("use_class_weights", None)
    if use_class_weights is None:
        use_class_weights = (summary.get("config") or {}).get("use_class_weights", None)

    if use_class_weights is not False:
        raise ValueError("Strict model validation failed: expected use_class_weights=false in summary")

    expected_best = artifacts_dir / "8vit_best_model.pth"
    expected_final = artifacts_dir / "8vit_final_model.pth"
    if model_path not in (expected_best, expected_final):
        raise ValueError(
            "Strict model validation failed: model path is not from strict artifacts dir "
            f"({artifacts_dir})"
        )

    logging.info("Strict model validated by summary: %s", summary_path)


def detect_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logging.info("Using device: %s", device)
    return device


def load_model(model_path: Path, model_name: str, dropout_rate: float, device: torch.device) -> nn.Module:
    model = ImprovedViTModel(model_name=model_name, dropout_rate=dropout_rate)
    payload = torch.load(model_path, map_location=device)

    # Support direct state_dict and wrapped checkpoints.
    if isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
    elif isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
    else:
        state_dict = payload

    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    logging.info("Model loaded successfully")
    return model


def connect_to_mongodb(mongo_uri: str, db_name: str):
    client = MongoClient(mongo_uri)
    client.admin.command("ping")
    logging.info("MongoDB connected: %s / %s", mongo_uri, db_name)
    return client[db_name]


def remove_hidden_files_if_needed(images_root: Path) -> None:
    if not images_root.exists():
        return
    count = 0
    for hidden in images_root.rglob("._*"):
        if hidden.is_file():
            hidden.unlink()
            count += 1
    if count > 0:
        logging.info("Removed %d hidden image files", count)


def build_transform():
    return transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def classify_image(model: nn.Module, image_path: Path, device: torch.device, transform) -> Tuple[Optional[int], Optional[float]]:
    try:
        img = Image.open(image_path).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
            probs = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
            pred = int(np.argmax(probs))
            return pred, float(probs[1])
    except Exception as exc:
        logging.error("Image inference failed: %s (%s)", image_path, exc)
        return None, None


def calculate_enhanced_weight(clarity_weight: float, text_weight: float) -> float:
    text_importance = 0.6
    base_weight = clarity_weight * (1 - text_importance) + text_weight * text_importance
    return float(np.tanh(base_weight * 1.2))


def build_path_mapping(filtered_collection) -> Dict[str, List[dict]]:
    mapping: Dict[str, List[dict]] = {}
    for doc in filtered_collection.find({}, {"original_id": 1, "valid_images": 1}):
        original_id = doc.get("original_id")
        if original_id and isinstance(doc.get("valid_images"), list):
            mapping[original_id] = doc["valid_images"]
    return mapping


def resolve_image_path(year: int, abs_path: str) -> Optional[Path]:
    if not abs_path:
        return None

    raw = Path(abs_path)
    candidates = [
        raw,
        Path.cwd() / raw,
        SCRIPT_DIR / raw,
        Path.cwd() / "images" / raw,
        Path.cwd() / "images" / str(year) / raw,
        SCRIPT_DIR / "images" / raw,
        SCRIPT_DIR / "images" / str(year) / raw,
    ]

    for c in candidates:
        if c.exists() and c.is_file():
            return c
    return None


def upsert_news_sentiment_summary(db, year: int, suffix: str = "_old") -> int:
    sentiment_collection = db[f"{year}_sentiment{suffix}"]
    news_collection = db[f"{year}_news_sentiment{suffix}"]
    news_collection.drop()

    pipeline = [
        {
            "$group": {
                "_id": {"news_date": "$news_date", "title": "$title"},
                "avg_negative_likelihood": {"$avg": "$negative_likelihood"},
                "avg_weighted_score": {"$avg": "$weighted_score"},
                "avg_weight": {"$avg": "$weight"},
                "avg_quality_score": {"$avg": "$quality_score"},
                "avg_diversity_score": {"$avg": "$diversity_score"},
                "avg_unique_ratio": {"$avg": "$unique_ratio"},
                "image_count": {"$sum": 1},
                "negative_count": {"$sum": {"$cond": [{"$eq": ["$predicted_class", 1]}, 1, 0]}},
                "positive_count": {"$sum": {"$cond": [{"$eq": ["$predicted_class", 0]}, 1, 0]}},
            }
        },
        {
            "$project": {
                "_id": 0,
                "news_date": "$_id.news_date",
                "title": "$_id.title",
                "avg_negative_likelihood": 1,
                "avg_weighted_score": 1,
                "avg_weight": 1,
                "avg_quality_score": 1,
                "avg_diversity_score": 1,
                "avg_unique_ratio": 1,
                "image_count": 1,
                "negative_count": 1,
                "positive_count": 1,
                "negative_ratio": {
                    "$cond": [
                        {"$gt": ["$image_count", 0]},
                        {"$divide": ["$negative_count", "$image_count"]},
                        0,
                    ]
                },
            }
        },
        {"$sort": {"news_date": 1}},
    ]

    rows = list(sentiment_collection.aggregate(pipeline))
    if rows:
        news_collection.insert_many(rows)
    return len(rows)


def process_year(
    model: nn.Module,
    device: torch.device,
    transform,
    db,
    year: int,
    args: argparse.Namespace,
    checkpoint: dict,
    checkpoint_path: Path,
    resume: bool,
) -> None:
    suffix = getattr(args, "collection_suffix", "_old")
    quality_collection_name = f"{year}_2"
    filtered_collection_name = f"{year}_filtered"
    sentiment_collection_name = f"{year}_sentiment{suffix}"

    collections = set(db.list_collection_names())
    if quality_collection_name not in collections:
        logging.warning("Skip %s: missing collection %s", year, quality_collection_name)
        return
    if filtered_collection_name not in collections:
        logging.warning("Skip %s: missing collection %s", year, filtered_collection_name)
        return

    year_key = str(year)
    year_ckpt = checkpoint.setdefault("years", {}).setdefault(year_key, {})
    last_doc_id = year_ckpt.get("last_doc_id") if resume else None

    should_drop_existing = (
        (not args.keep_existing)
        and (sentiment_collection_name in collections)
        and (not (resume and last_doc_id))
    )
    if should_drop_existing:
        db[sentiment_collection_name].drop()
        logging.info("Dropped existing collection: %s", sentiment_collection_name)
    elif resume and last_doc_id:
        logging.info("Resume mode enabled for %s, continuing after _id=%s", year, last_doc_id)

    quality_collection = db[quality_collection_name]
    filtered_collection = db[filtered_collection_name]
    sentiment_collection = db[sentiment_collection_name]

    path_mapping = build_path_mapping(filtered_collection)
    query = {}
    if resume and last_doc_id:
        try:
            query["_id"] = {"$gt": ObjectId(last_doc_id)}
        except Exception:
            logging.warning("Invalid checkpoint _id for %s: %s; fallback to full-year query", year, last_doc_id)

    total_docs = quality_collection.count_documents(query)
    if args.max_news_per_year > 0:
        total_docs = min(total_docs, args.max_news_per_year)
    if total_docs == 0:
        logging.warning("No docs in %s", quality_collection_name)
        return

    projection = {
        "publish_date": 1,
        "news_date": 1,
        "title": 1,
        "original_id": 1,
        "processed_images": 1,
        "diversity_score": 1,
        "unique_ratio": 1,
    }
    cursor = quality_collection.find(query, projection).sort("_id", 1)
    if args.max_news_per_year > 0:
        cursor = cursor.limit(args.max_news_per_year)

    negative_count = 0
    positive_count = 0
    total_negative_prob = 0.0
    weighted_negative_prob = 0.0
    total_weight = 0.0
    processed_count = 0
    total_images = 0
    failed_images: List[Tuple[str, str]] = []
    processed_docs = 0

    for doc in tqdm(cursor, total=total_docs, desc=f"Sentiment {year}"):
        news_date = doc.get("news_date") or doc.get("publish_date") or f"{year}-01-01"
        title = doc.get("title", "")
        original_id = doc.get("original_id")
        processed_images = doc.get("processed_images") or []
        diversity_score = float(doc.get("diversity_score", 1.0) or 1.0)
        unique_ratio = float(doc.get("unique_ratio", 1.0) or 1.0)

        if not processed_images:
            continue

        mapped_valid_images = path_mapping.get(original_id, []) if original_id else []

        doc_ops: List[UpdateOne] = []
        for idx, proc_img in enumerate(processed_images):
            total_images += 1

            abs_path = str(proc_img.get("abs_path") or "").strip()
            if not abs_path and idx < len(mapped_valid_images):
                abs_path = str(mapped_valid_images[idx].get("abs_path") or "").strip()
            if not abs_path:
                failed_images.append(("", "missing_path"))
                continue

            image_path = resolve_image_path(year, abs_path)
            if image_path is None:
                failed_images.append((abs_path, "file_not_found"))
                continue

            clarity_weight = float(proc_img.get("clarity_weight", 0.5) or 0.5)
            text_weight = float(proc_img.get("text_weight", 0.5) or 0.5)
            quality_score = float(proc_img.get("quality_score", 0.5) or 0.5)
            weight = calculate_enhanced_weight(clarity_weight, text_weight)

            predicted_class, negative_prob = classify_image(model, image_path, device, transform)
            if predicted_class is None or negative_prob is None:
                failed_images.append((str(image_path), "inference_failed"))
                continue

            record_key = f"{original_id}::{abs_path}"
            result_doc = {
                "record_key": record_key,
                "original_id": original_id,
                "image_path": str(image_path),
                "rel_path": abs_path,
                "predicted_class": int(predicted_class),
                "negative_likelihood": float(negative_prob),
                "news_date": news_date,
                "title": title,
                "clarity_weight": clarity_weight,
                "text_weight": text_weight,
                "weight": weight,
                "quality_score": quality_score,
                "diversity_score": diversity_score,
                "unique_ratio": unique_ratio,
                "weighted_score": float(negative_prob * weight * diversity_score),
                "timestamp": datetime.datetime.now(),
            }
            doc_ops.append(UpdateOne({"record_key": record_key}, {"$set": result_doc}, upsert=True))

            if predicted_class == 1:
                negative_count += 1
            else:
                positive_count += 1
            total_negative_prob += negative_prob
            weighted_negative_prob += negative_prob * weight * diversity_score
            total_weight += weight
            processed_count += 1

            if len(doc_ops) >= args.batch_write_size:
                sentiment_collection.bulk_write(doc_ops, ordered=False)
                doc_ops.clear()

        if doc_ops:
            sentiment_collection.bulk_write(doc_ops, ordered=False)

        processed_docs += 1
        year_ckpt["last_doc_id"] = str(doc.get("_id"))
        year_ckpt["processed_docs"] = int(year_ckpt.get("processed_docs", 0)) + 1
        year_ckpt["updated_at"] = datetime.datetime.now().isoformat()

        if processed_docs % args.checkpoint_every == 0:
            save_checkpoint(checkpoint_path, checkpoint)

    news_count = upsert_news_sentiment_summary(db, year, suffix=suffix) if processed_count > 0 else 0

    year_ckpt["completed"] = True
    year_ckpt["completed_at"] = datetime.datetime.now().isoformat()
    save_checkpoint(checkpoint_path, checkpoint)

    logging.info("=== %s summary ===", year)
    logging.info("Total images seen: %d", total_images)
    logging.info("Successful predictions: %d", processed_count)
    logging.info("Failed predictions: %d", len(failed_images))
    logging.info("News-level summary rows: %d", news_count)

    if processed_count > 0:
        logging.info("Positive (0): %d (%.2f%%)", positive_count, positive_count / processed_count * 100)
        logging.info("Negative (1): %d (%.2f%%)", negative_count, negative_count / processed_count * 100)
        logging.info("Avg negative likelihood: %.4f", total_negative_prob / processed_count)
        if total_weight > 0:
            logging.info("Weighted avg negative likelihood: %.4f", weighted_negative_prob / total_weight)

    if failed_images:
        for image_path, reason in failed_images[:10]:
            logging.info("Failed sample: %s (%s)", image_path, reason)
        if len(failed_images) > 10:
            logging.info("... and %d more failures", len(failed_images) - 10)


def main() -> None:
    args = parse_args()

    years = resolve_years(args)
    logging.info("Years to process: %s", years)

    checkpoint_path = Path(args.checkpoint_file)
    if args.reset_checkpoint and checkpoint_path.exists():
        checkpoint_path.unlink()
        logging.info("Removed old checkpoint: %s", checkpoint_path)

    resume = not args.no_resume
    checkpoint = load_checkpoint(checkpoint_path) if resume else {"years": {}}

    remove_hidden_files_if_needed(Path.cwd() / "images")

    device = detect_device()
    model_path = resolve_model_path(args)
    validate_strict_model(args, model_path)
    model = load_model(model_path, args.model_name, args.dropout_rate, device)
    transform = build_transform()

    db = connect_to_mongodb(args.mongo_uri, args.db_name)
    for year in years:
        process_year(model, device, transform, db, year, args, checkpoint, checkpoint_path, resume)

    logging.info("Sentiment analysis completed for years: %s", years)


if __name__ == "__main__":
    main()
