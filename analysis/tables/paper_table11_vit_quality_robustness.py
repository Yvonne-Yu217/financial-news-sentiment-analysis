"""
paper_table11_vit_quality_robustness.py
ViT robustness across image-quality strata and across years (2014-2026).

For each of the 6,107 Sina news images already annotated by three frontier
multimodal LLMs (GPT-4o-mini + Claude Sonnet 4.6 + Gemini 3 Pro Preview),
re-evaluate the production ViT (improved_vit_sentiment_model_old.pth) and
report the distribution of pessimism scores conditional on (quality stratum,
year). The annotation panel is uniformly stratified by year (about 500/year
for 2015-2026, 107 for 2014).

Output:
  results/regression_output_oldmodel/table11_vit_quality_panel.csv  (per image)
  results/regression_output_oldmodel/table11_vit_quality_summary.json
"""
import warnings; warnings.filterwarnings('ignore')
import os, json, sys
from pathlib import Path
import numpy as np, pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[2]
ANNOT_CSV = REPO_ROOT / "ai_image_annotation/run_artifacts/ai_image_annotations_run_20260409_clean.csv"
MODEL_PATH = REPO_ROOT / "improved_vit_sentiment_model_old.pth"
OUT_DIR = REPO_ROOT / "results/regression_output_oldmodel"

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class ImprovedViTClassifier(nn.Module):
    """Replicates 8vit_transferlearning_old.py: head replaced inside backbone."""
    def __init__(self, num_classes=2, dropout_rate=0.2):
        super().__init__()
        import timm
        self.backbone = timm.create_model('vit_base_patch16_224', pretrained=False)
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


def load_model(path: Path, device: torch.device) -> nn.Module:
    payload = torch.load(path, map_location=device, weights_only=False)
    model = ImprovedViTClassifier()
    if isinstance(payload, dict) and 'model_state_dict' in payload:
        model.load_state_dict(payload['model_state_dict'])
    elif isinstance(payload, dict) and all(k.startswith(('backbone.', 'classifier.')) for k in payload.keys()):
        model.load_state_dict(payload)
    else:
        try:
            model.load_state_dict(payload)
        except Exception:
            model = payload
    model.to(device).eval()
    return model


def transform_pipeline():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def classify_image(model, image_path: Path, device, tx):
    try:
        img = Image.open(image_path).convert("RGB")
        tensor = tx(img).unsqueeze(0).to(device)
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
        return int(np.argmax(probs)), float(probs[1])
    except Exception:
        return None, None


def entropy(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-(p * np.log(p) + (1 - p) * np.log(1 - p)).mean() / np.log(2))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEVICE}")
    print(f"Loading annotation panel: {ANNOT_CSV}")
    df = pd.read_csv(ANNOT_CSV)
    print(f"Loaded {len(df)} images.")

    print(f"Loading ViT: {MODEL_PATH}")
    model = load_model(MODEL_PATH, DEVICE)
    tx = transform_pipeline()

    preds, probs, ok = [], [], []
    for i, row in enumerate(df.itertuples(index=False)):
        path = Path(row.image_path)
        if not path.exists():
            preds.append(None); probs.append(None); ok.append(False); continue
        cls, pp = classify_image(model, path, DEVICE, tx)
        preds.append(cls); probs.append(pp); ok.append(cls is not None)
        if (i + 1) % 500 == 0:
            print(f"  processed {i+1}/{len(df)}")
    df['vit_pred_class'] = preds
    df['vit_pessimism'] = probs
    df['vit_ok'] = ok

    successful = df[df['vit_ok']].copy()
    print(f"\nSuccessful inferences: {len(successful)}/{len(df)}")

    df.to_csv(OUT_DIR / 'table11_vit_quality_panel.csv', index=False)

    # ============ Panel A: Quality strata ============
    print("\n" + "=" * 90)
    print("Panel A: ViT pessimism by quality stratum")
    print("=" * 90)
    panelA_rows = []
    for label in ['quality_pass_auto_accept', 'quality_review_needed', 'quality_fail_candidate']:
        sub = successful[successful['predicted_quality_label'] == label]
        if len(sub) == 0:
            continue
        p = sub['vit_pessimism'].values
        row = dict(
            quality_stratum=label,
            n=len(sub),
            mean=float(np.mean(p)),
            median=float(np.median(p)),
            std=float(np.std(p)),
            entropy=entropy(p),
            pct_class1_predicted=float((sub['vit_pred_class'] == 1).mean() * 100),
            decisive_share=float(((p < 0.2) | (p > 0.8)).mean() * 100),
        )
        panelA_rows.append(row)
        print(f"  {label:30s}  N={row['n']:5d}  mean={row['mean']:.3f}  std={row['std']:.3f}  "
              f"entropy={row['entropy']:.3f}  pct_decisive={row['decisive_share']:5.1f}%")

    # ============ Panel B: Year stability on quality_pass ============
    print("\n" + "=" * 90)
    print("Panel B: Year stability on quality_pass subset")
    print("=" * 90)
    pass_sub = successful[successful['predicted_quality_label'] == 'quality_pass_auto_accept']
    panelB_rows = []
    for yr in sorted(pass_sub['year'].unique()):
        ys = pass_sub[pass_sub['year'] == yr]
        if len(ys) == 0:
            continue
        p = ys['vit_pessimism'].values
        row = dict(year=int(yr), n=len(ys),
                   mean=float(np.mean(p)),
                   std=float(np.std(p)),
                   pct_pess=float((ys['vit_pred_class'] == 1).mean() * 100))
        panelB_rows.append(row)
        print(f"  Year {row['year']}  N={row['n']:4d}  mean={row['mean']:.3f}  "
              f"std={row['std']:.3f}  %pessimism={row['pct_pess']:5.1f}%")

    # Cross-year std of mean (drift indicator)
    yr_means = np.array([r['mean'] for r in panelB_rows])
    drift_metric = float(yr_means.std())
    print(f"\nCross-year std of mean PhotoPes (drift): {drift_metric:.4f}")

    # ============ Panel C: Consensus subset (api_agreement = 1.0) ============
    print("\n" + "=" * 90)
    print("Panel C: Frontier-LLM consensus subset (api_agreement = 1.0)")
    print("=" * 90)
    cons = successful[successful['api_agreement'] >= 0.999]
    panelC_rows = []
    for label in ['quality_pass_auto_accept', 'quality_review_needed', 'quality_fail_candidate']:
        sub = cons[cons['predicted_quality_label'] == label]
        if len(sub) == 0:
            continue
        p = sub['vit_pessimism'].values
        row = dict(quality_stratum=label, n=len(sub),
                   mean=float(np.mean(p)), std=float(np.std(p)),
                   pct_pess=float((sub['vit_pred_class'] == 1).mean() * 100),
                   decisive_share=float(((p < 0.2) | (p > 0.8)).mean() * 100))
        panelC_rows.append(row)
        print(f"  {label:30s}  N={row['n']:5d}  mean={row['mean']:.3f}  "
              f"std={row['std']:.3f}  %pess={row['pct_pess']:5.1f}%  decisive={row['decisive_share']:5.1f}%")

    # ============ Panel D: Welch's t-test pass vs fail ============
    print("\n" + "=" * 90)
    print("Panel D: Test of difference in ViT pessimism (pass vs fail)")
    print("=" * 90)
    from scipy import stats as scs
    p_pass = successful.loc[successful['predicted_quality_label'] == 'quality_pass_auto_accept', 'vit_pessimism'].values
    p_fail = successful.loc[successful['predicted_quality_label'] == 'quality_fail_candidate', 'vit_pessimism'].values
    t_test = scs.ttest_ind(p_pass, p_fail, equal_var=False)
    ks_test = scs.ks_2samp(p_pass, p_fail)
    welch_result = dict(
        n_pass=len(p_pass), mean_pass=float(p_pass.mean()), std_pass=float(p_pass.std()),
        n_fail=len(p_fail), mean_fail=float(p_fail.mean()), std_fail=float(p_fail.std()),
        welch_t=float(t_test.statistic), welch_p=float(t_test.pvalue),
        ks_d=float(ks_test.statistic), ks_p=float(ks_test.pvalue),
    )
    print(f"  pass: N={welch_result['n_pass']}  mean={welch_result['mean_pass']:.3f}  std={welch_result['std_pass']:.3f}")
    print(f"  fail: N={welch_result['n_fail']}  mean={welch_result['mean_fail']:.3f}  std={welch_result['std_fail']:.3f}")
    print(f"  Welch's t = {welch_result['welch_t']:+.3f}  p = {welch_result['welch_p']:.4g}")
    print(f"  KS distance = {welch_result['ks_d']:.3f}  p = {welch_result['ks_p']:.4g}")

    summary = dict(
        n_total=int(len(df)), n_inferred=int(len(successful)),
        panelA=panelA_rows, panelB=panelB_rows, panelC=panelC_rows,
        panelD=welch_result,
        cross_year_drift=drift_metric,
    )
    with open(OUT_DIR / 'table11_vit_quality_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {OUT_DIR}/table11_vit_quality_panel.csv")
    print(f"Saved: {OUT_DIR}/table11_vit_quality_summary.json")


if __name__ == '__main__':
    main()
