"""
r07_full_sample_auc.py
=======================
Run PCNN-ViT inference on ALL 1,253 LLM-annotated Sina images
(no consensus / score filter), then compute:

  (a) AUC of ViT pessimism_prob against binary LLM majority label
      on the full sample (no filter)
  (b) AUC against threshold-binarized continuous LLM mean_sentiment_score
      on the full sample
  (c) Spearman / Pearson correlation between ViT prob1 and LLM mean score
  (d) For comparison: same metrics on the strong-consensus subset (309)

The current paper §5.1 reports AUC=0.71 on the filtered 309-image
subset. Referee 2 v2 MC6 demands the FULL-sample AUC since filtering
removes the hard cases and inflates AUC.

Output: results/r07_full_auc_panel.csv  +  results/r07_full_auc_summary.json
"""
import csv, json, warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import timm
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr, spearmanr

REPO = Path("/Volumes/Data_Drive/research0322226/financial-news-sentiment-analysis")
MODEL_PATH = REPO / "improved_vit_sentiment_model_old.pth"
ANNOT_CSV = REPO / "ai_image_annotation/run_artifacts/ai_image_sentiment_annotations.csv"
EXISTING_PANEL = REPO / "results/validate_old_vit_with_llm_consensus_panel.csv"
OUT_DIR = REPO / "results"
PANEL_OUT = OUT_DIR / "r07_full_auc_panel.csv"
SUMMARY_OUT = OUT_DIR / "r07_full_auc_summary.json"

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {DEVICE}")


class ImprovedViTClassifier(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.2):
        super().__init__()
        self.backbone = timm.create_model("vit_base_patch16_224", pretrained=False)
        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Sequential(
            nn.Linear(in_features, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(dropout_rate),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


def load_model():
    model = ImprovedViTClassifier().to(DEVICE)
    state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()
    return model


def transform():
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
        probs = F.softmax(model(tensor), dim=1).cpu().numpy()[0]
        return int(np.argmax(probs)), float(probs[1])
    except Exception:
        return None, None


def main():
    # Load existing panel for the 309 with ViT already computed
    existing = {}
    if EXISTING_PANEL.exists():
        df_e = pd.read_csv(EXISTING_PANEL)
        for _, r in df_e.iterrows():
            existing[r["image_id"]] = (r["old_vit_pred_class"], r["old_vit_pessimism_prob"])
        print(f"Loaded {len(existing)} pre-computed ViT scores from existing panel")

    # Load full annotation set
    print(f"Loading full annotation: {ANNOT_CSV}")
    rows = list(csv.DictReader(open(ANNOT_CSV, "r", encoding="utf-8")))
    print(f"  total: {len(rows)} images")

    # Determine which need fresh ViT inference
    todo = [r for r in rows if r["image_id"] not in existing]
    print(f"  already done: {len(rows) - len(todo)}; need to compute: {len(todo)}")

    if todo:
        model = load_model()
        tx = transform()
        for i, r in enumerate(todo):
            path = Path(r["image_path"])
            if not path.exists():
                continue
            cls, prob1 = classify(model, path, tx)
            if cls is None:
                continue
            existing[r["image_id"]] = (cls, prob1)
            if (i + 1) % 100 == 0:
                print(f"  Inference {i+1}/{len(todo)}")
        print(f"  Inference complete. Total ViT scores: {len(existing)}")

    # Build full panel (1,253 rows), excluding annotation errors / missing ViT
    panel_rows = []
    skipped_error = 0
    skipped_no_vit = 0
    for r in rows:
        if r.get("majority_sentiment_label", "").lower() == "error":
            skipped_error += 1
            continue
        img_id = r["image_id"]
        if img_id not in existing:
            skipped_no_vit += 1
            continue
        cls, prob1 = existing[img_id]
        try:
            score = float(r["mean_sentiment_score"])
            agree = float(r["api_agreement"])
        except (KeyError, ValueError):
            continue

        # Binary label from majority vote (positive=0, negative=1, neutral=skip-or-treat)
        majority = r.get("majority_sentiment_label", "").lower()
        if majority == "positive":
            llm_binary = 0
        elif majority == "negative":
            llm_binary = 1
        elif majority == "neutral":
            # Use threshold on continuous score
            llm_binary = 1 if score >= 0.5 else 0
        else:
            continue

        panel_rows.append({
            "image_id": img_id,
            "llm_majority": majority,
            "llm_mean_score": score,
            "llm_binary": llm_binary,
            "llm_agreement": agree,
            "vit_pred_class": cls,
            "vit_pess_prob": prob1,
        })

    df = pd.DataFrame(panel_rows)
    print(f"\nFull panel: {len(df)} rows  (skipped: error={skipped_error}, no_vit={skipped_no_vit})")

    df.to_csv(PANEL_OUT, index=False)
    print(f"Saved panel: {PANEL_OUT}")

    # ───────────────────────── Compute AUC + correlations ─────────────────────────
    summary = {}

    # FULL sample (no filter)
    if len(df) > 50:
        full_auc = roc_auc_score(df["llm_binary"], df["vit_pess_prob"])
        full_n = len(df)
        full_pos = (df["llm_binary"] == 0).sum()
        full_neg = (df["llm_binary"] == 1).sum()
        # Continuous correlations
        pearson_r, pearson_p = pearsonr(df["vit_pess_prob"], df["llm_mean_score"])
        spearman_rho, spearman_p = spearmanr(df["vit_pess_prob"], df["llm_mean_score"])
        summary["full_sample"] = {
            "n": full_n, "n_pos": int(full_pos), "n_neg": int(full_neg),
            "auc_binary": float(full_auc),
            "pearson_r": float(pearson_r), "pearson_p": float(pearson_p),
            "spearman_rho": float(spearman_rho), "spearman_p": float(spearman_p),
        }

    # Strong-consensus subset (replicates §5.1's existing 309)
    AGREE_TH, POS_TH, NEG_TH = 0.50, 0.40, 0.60
    sub = df[
        ((df["llm_majority"] == "positive") & (df["llm_mean_score"] <= POS_TH) & (df["llm_agreement"] >= AGREE_TH)) |
        ((df["llm_majority"] == "negative") & (df["llm_mean_score"] >= NEG_TH) & (df["llm_agreement"] >= AGREE_TH))
    ].copy()
    if len(sub) > 50:
        sub_auc = roc_auc_score(sub["llm_binary"], sub["vit_pess_prob"])
        summary["strong_consensus_subset"] = {
            "n": len(sub), "n_pos": int((sub["llm_binary"] == 0).sum()),
            "n_neg": int((sub["llm_binary"] == 1).sum()),
            "auc_binary": float(sub_auc),
        }

    # Print summary
    print("\n" + "=" * 65)
    print("R-07: ViT vs LLM AUC — full sample vs filtered subset")
    print("=" * 65)
    if "full_sample" in summary:
        s = summary["full_sample"]
        print(f"FULL sample (n={s['n']}, pos={s['n_pos']}, neg={s['n_neg']}):")
        print(f"  AUC vs binary LLM label  : {s['auc_binary']:.4f}")
        print(f"  Pearson(ViT_prob, LLM_score)  : r={s['pearson_r']:.4f} (p={s['pearson_p']:.2e})")
        print(f"  Spearman(ViT_prob, LLM_score) : ρ={s['spearman_rho']:.4f} (p={s['spearman_p']:.2e})")
    if "strong_consensus_subset" in summary:
        s = summary["strong_consensus_subset"]
        print(f"\nSTRONG-consensus subset (n={s['n']}):")
        print(f"  AUC vs binary LLM label  : {s['auc_binary']:.4f}")

    print("\nReferee MC6 (R2 v2) note: paper currently only reports the strong-consensus")
    print("subset AUC; the FULL-sample AUC is the honest cross-domain validation number.")

    with open(SUMMARY_OUT, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary: {SUMMARY_OUT}")


if __name__ == "__main__":
    main()
