"""
validate_old_vit_with_llm_consensus.py

External validation of the paper's PNCC-trained ViT against frontier-LLM
consensus labels on held-out Sina financial-news images.

The PNCC ViT (improved_vit_sentiment_model_old.pth) was fine-tuned on PNCC
Twitter sentiment data (Twitter photos, not Sina news). It has NEVER seen
the Sina financial images during training.

The 3-model LLM ensemble (gpt-4o-mini + claude-sonnet-4-6 + gemini-3-pro-preview)
labeled 1,253 pass-quality Sina images. We keep the strong-consensus subset:
  positive (LLM majority='positive', mean_score<=0.40, agree>=0.50): ~208 images
  negative (LLM majority='negative', mean_score>=0.60, agree>=0.50): ~101 images

For each image, we run the PNCC ViT and compare its predicted_class
(0=positive, 1=negative) and prob[1] (pessimism likelihood) against the
LLM binary label.

Outputs:
  results/validate_old_vit_with_llm_consensus_panel.csv  (per-image)
  results/validate_old_vit_with_llm_consensus_summary.json
"""
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import csv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import timm
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
MODEL_PATH = REPO_ROOT / "improved_vit_sentiment_model_old.pth"
ANNOT_CSV = REPO_ROOT / "ai_image_annotation/run_artifacts/ai_image_sentiment_annotations.csv"
OUT_DIR = REPO_ROOT / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PANEL_CSV = OUT_DIR / "validate_old_vit_with_llm_consensus_panel.csv"
SUMMARY_JSON = OUT_DIR / "validate_old_vit_with_llm_consensus_summary.json"

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Same filter as build_sentiment_training_dataset.py used for the v2 dataset
AGREE_TH = 0.50
POS_SCORE_TH = 0.40
NEG_SCORE_TH = 0.60


class ImprovedViTClassifier(nn.Module):
    """Replicates 8vit_transferlearning_old.py architecture."""
    def __init__(self, num_classes=2, dropout_rate=0.2):
        super().__init__()
        self.backbone = timm.create_model("vit_base_patch16_224", pretrained=False)
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


def load_old_vit():
    payload = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model = ImprovedViTClassifier()
    if isinstance(payload, dict) and "model_state_dict" in payload:
        model.load_state_dict(payload["model_state_dict"])
    elif isinstance(payload, dict) and all(k.startswith(("backbone.", "classifier.")) for k in payload.keys()):
        model.load_state_dict(payload)
    else:
        try:
            model.load_state_dict(payload)
        except Exception:
            model = payload
    model.to(DEVICE).eval()
    return model


def transform_pipeline():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def classify(model, image_path, tx):
    try:
        img = Image.open(image_path).convert("RGB")
        tensor = tx(img).unsqueeze(0).to(DEVICE)
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        return int(np.argmax(probs)), float(probs[1])
    except Exception:
        return None, None


def filter_strong_consensus(rows):
    """Apply the same agreement/score thresholds as build_sentiment_training_dataset.py."""
    keep = []
    for r in rows:
        majority = (r.get("majority_sentiment_label") or "").strip().lower()
        if majority == "error":
            continue
        try:
            score = float(r.get("mean_sentiment_score") or 0.5)
            agree = float(r.get("api_agreement") or 0)
        except ValueError:
            continue
        if agree < AGREE_TH:
            continue
        if majority == "positive" and score <= POS_SCORE_TH:
            keep.append({**r, "llm_label_binary": 0})  # 0=positive (matches PNCC convention)
        elif majority == "negative" and score >= NEG_SCORE_TH:
            keep.append({**r, "llm_label_binary": 1})  # 1=negative
    return keep


def main():
    print(f"Device: {DEVICE}")
    print(f"Loading PNCC ViT: {MODEL_PATH}")
    model = load_old_vit()
    tx = transform_pipeline()

    print(f"Loading LLM annotation panel: {ANNOT_CSV}")
    rows = list(csv.DictReader(open(ANNOT_CSV, "r", encoding="utf-8")))
    print(f"  total annotated: {len(rows)}")

    panel = filter_strong_consensus(rows)
    pos_n = sum(1 for r in panel if r["llm_label_binary"] == 0)
    neg_n = sum(1 for r in panel if r["llm_label_binary"] == 1)
    print(f"  strong-consensus subset: {len(panel)} (pos={pos_n}, neg={neg_n})")

    # Run PNCC ViT on each
    print("\nRunning PNCC ViT inference...")
    output_rows = []
    failed = 0
    for i, r in enumerate(panel):
        path = Path(r["image_path"])
        if not path.exists():
            failed += 1
            continue
        cls, prob1 = classify(model, path, tx)
        if cls is None:
            failed += 1
            continue
        output_rows.append({
            "image_id": r.get("image_id", ""),
            "year": r.get("year", ""),
            "news_date": r.get("news_date", ""),
            "image_path": str(path),
            "llm_label_binary": r["llm_label_binary"],   # 0=pos, 1=neg
            "llm_majority_label": r.get("majority_sentiment_label", ""),
            "llm_mean_score": r.get("mean_sentiment_score", ""),
            "llm_agreement": r.get("api_agreement", ""),
            "old_vit_pred_class": cls,
            "old_vit_pessimism_prob": prob1,
        })
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(panel)}")

    print(f"\nSuccessful inferences: {len(output_rows)}/{len(panel)}  (failed/missing: {failed})")

    # Write panel
    df = pd.DataFrame(output_rows)
    df.to_csv(PANEL_CSV, index=False)
    print(f"Saved per-image panel: {PANEL_CSV}")

    # Aggregate metrics
    y_true = df["llm_label_binary"].values  # 0=pos, 1=neg
    y_pred = df["old_vit_pred_class"].values
    y_score = df["old_vit_pessimism_prob"].values

    acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, labels=[0, 1])
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    try:
        auc = roc_auc_score(y_true, y_score)
    except Exception:
        auc = None

    # Pessimism prob distribution by LLM label
    pos_probs = df.loc[df["llm_label_binary"] == 0, "old_vit_pessimism_prob"].values
    neg_probs = df.loc[df["llm_label_binary"] == 1, "old_vit_pessimism_prob"].values

    # Cohen's d for prob[1] difference
    pooled_std = np.sqrt(
        ((len(pos_probs) - 1) * pos_probs.std(ddof=1) ** 2
         + (len(neg_probs) - 1) * neg_probs.std(ddof=1) ** 2)
        / (len(pos_probs) + len(neg_probs) - 2)
    )
    cohens_d = (neg_probs.mean() - pos_probs.mean()) / pooled_std if pooled_std > 0 else None

    # Welch's t and KS
    from scipy import stats as scs
    welch = scs.ttest_ind(neg_probs, pos_probs, equal_var=False)
    ks = scs.ks_2samp(pos_probs, neg_probs)

    # Per-year agreement rate
    year_breakdown = {}
    for yr, grp in df.groupby("year"):
        if len(grp) < 5:
            continue
        year_breakdown[str(yr)] = {
            "n": len(grp),
            "n_pos": int((grp["llm_label_binary"] == 0).sum()),
            "n_neg": int((grp["llm_label_binary"] == 1).sum()),
            "agreement_rate": float(accuracy_score(grp["llm_label_binary"], grp["old_vit_pred_class"])),
        }

    summary = {
        "model": str(MODEL_PATH),
        "n_total": int(len(df)),
        "n_pos": int(pos_n),
        "n_neg": int(neg_n),
        "agreement_rate": round(float(acc), 4),
        "cohens_kappa": round(float(kappa), 4),
        "auc_pessimism_prob": round(float(auc), 4) if auc is not None else None,
        "f1_macro": round(float(f1_macro), 4),
        "f1_weighted": round(float(f1_weighted), 4),
        "per_class": {
            "positive (LLM label 0)": {
                "precision": round(float(p[0]), 4),
                "recall": round(float(r[0]), 4),
                "f1": round(float(f1[0]), 4),
                "support": int((y_true == 0).sum()),
            },
            "negative (LLM label 1)": {
                "precision": round(float(p[1]), 4),
                "recall": round(float(r[1]), 4),
                "f1": round(float(f1[1]), 4),
                "support": int((y_true == 1).sum()),
            },
        },
        "confusion_matrix": {
            "rows": "true_LLM",
            "cols": "pred_PNCC_ViT",
            "labels": ["positive=0", "negative=1"],
            "matrix": cm.tolist(),
        },
        "pessimism_prob_separation": {
            "mean_for_LLM_positive": round(float(pos_probs.mean()), 4),
            "mean_for_LLM_negative": round(float(neg_probs.mean()), 4),
            "mean_diff": round(float(neg_probs.mean() - pos_probs.mean()), 4),
            "cohens_d": round(float(cohens_d), 4) if cohens_d is not None else None,
            "welch_t": round(float(welch.statistic), 4),
            "welch_p": float(welch.pvalue),
            "ks_distance": round(float(ks.statistic), 4),
            "ks_p": float(ks.pvalue),
        },
        "per_year": year_breakdown,
    }

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nSaved summary: {SUMMARY_JSON}")

    # Print pretty
    print("\n" + "=" * 70)
    print("PNCC ViT vs LLM Consensus — Held-out Validation")
    print("=" * 70)
    print(f"  N images: {len(df)}  (pos={pos_n}, neg={neg_n})")
    print(f"  Agreement rate    : {acc:.2%}")
    print(f"  Cohen's kappa     : {kappa:.3f}")
    if auc is not None:
        print(f"  AUC (prob[1] vs label): {auc:.3f}")
    print(f"  F1 macro / weighted: {f1_macro:.3f} / {f1_weighted:.3f}")
    print()
    print("  Per-class:")
    print(f"    LLM=positive  : precision={p[0]:.3f}  recall={r[0]:.3f}  f1={f1[0]:.3f}  n={int((y_true==0).sum())}")
    print(f"    LLM=negative  : precision={p[1]:.3f}  recall={r[1]:.3f}  f1={f1[1]:.3f}  n={int((y_true==1).sum())}")
    print()
    print("  Confusion matrix (rows=LLM, cols=PNCC):")
    print(f"                pred_pos    pred_neg")
    print(f"    LLM=pos     {cm[0,0]:>5}        {cm[0,1]:>5}")
    print(f"    LLM=neg     {cm[1,0]:>5}        {cm[1,1]:>5}")
    print()
    print("  Pessimism prob[1] separation:")
    print(f"    For LLM-positive images: mean={pos_probs.mean():.3f}, std={pos_probs.std():.3f}")
    print(f"    For LLM-negative images: mean={neg_probs.mean():.3f}, std={neg_probs.std():.3f}")
    print(f"    Cohen's d = {cohens_d:.3f}" if cohens_d is not None else "    Cohen's d = N/A")
    print(f"    Welch's t = {welch.statistic:+.3f}, p = {welch.pvalue:.4g}")
    print(f"    KS distance = {ks.statistic:.3f}, p = {ks.pvalue:.4g}")
    print()
    print("  Per-year agreement (>=5 images):")
    for yr, d in sorted(year_breakdown.items()):
        print(f"    {yr}: n={d['n']:>3}, agreement={d['agreement_rate']:.2%}")


if __name__ == "__main__":
    main()
