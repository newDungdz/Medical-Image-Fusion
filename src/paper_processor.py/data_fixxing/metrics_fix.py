import json
import re
from ftfy import fix_text

METRIC_ALIASES = {
    "SSIM": [
        "Structural Similarity Index Measure",
        "Structural Similarity Index",
        "Structural Similarity",
        "QSSIM",
        "Structural Similarity-Based Metric (SSIM)",
        "Structure Similarity Index Measure (SSIM)",
    ],

    "MS-SSIM": [
        "MS_SSIM",
        "Multi-Scale Structural Similarity",
        "Multi-Scale Structural Similarity Index",
        "Multi-Scale SSIM",
    ],

    "VIF": [
        "Visual Information Fidelity",
        "Visual Fidelity",
        "VIFP",
        "QVIF",
        "Q_VIF",
    ],

    "VIFF": [
        "Visual Information Fidelity for Fusion",
    ],

    "MI": [
        "Mutual Information",
        "QMI",
        "Q_MI",
        "Quantitative Mutual Information",
    ],

    "EN": [
        "Entropy",
        "Information Entropy",
        "IE",
        "Q_EN",
        "Entropy Metric (EN)",
        # 'H' is too short/ambiguous — handle via keyword only if needed
    ],

    "SD": [
        "Standard Deviation",
        "STD",
        "Q_SD",
    ],

    "SF": [
        "Spatial Frequency",
        "Q_SF",
    ],

    "AG": [
        "Average Gradient",
        "AVG",
    ],

    "PSNR": [
        "Peak Signal-to-Noise Ratio",
    ],

    "SCD": [
        "Sum of Correlation Differences",
        "structural content difference",
        "structural content dissimilarity",
        "Q_SCD",
        "Sum of Correlations of Differences (SCD)",
        "Sum of Correlation of Differences (SCD)",
        "Sum of the Correlations of Differences (SCD)",
    ],

    "QAB/F": [
        "Qabf",
        "Qab/f",
        "Q_AB/F",
        "Q_ABF",
        "edge-based fusion quality metric",
        "edge information measurement",
        "Quality Assessment of Blended Images for Fusion (QAB/F)",
        "Quality Assessment-Based Metric (Qabf)",
        "Edge-based quality metric QAB/F",
        "gradient-based fusion performance (Qabf)",
        "gradient-based similarity measurement (Qabf)",
        "quality of fusion Qabf",
        "Edge Preservation Values (QAB/F)",
        "HVSQAB/F",
        "Gradient-Based Fusion Performance",
    ],

    "NAB/F": [
        "Nabf",
    ],

    "IoU": [
        "Intersection over Union",
        "intersection-over-union",
    ],

    "mIoU": [
        "mean Intersection over Union",
        "Mean Intersection-over-Union",
        "Mean Intersection over Union",
    ],

    "AP": [
        "Average Precision",
    ],

    "mAP": [
        "mean average precision",
        "mAP50",
        "mAP@0.5",
        "mAP@.5",
    ],

    "RMSE": [
        "Root Mean Square Error",
    ],

    "MSE": [
        "Mean Squared Error",
    ],

    "MAE": [
        "Mean Absolute Error",
    ],

    "CC": [
        "Correlation Coefficient",
        "Correlation Coefficient (CC)",
        "CORR",
        "Pearson Correlation Coefficient",
    ],

    "NCC": [
        "normalized cross-correlation",
    ],

    "QCV": [
        "Qcv",
        "Q_CV",
        "Chen–Varshney metric",
        "Chen-Varshney metric",
        "Chen-Varsheny Metric",
    ],

    "QCB": [
        "Qcb",
        "Q_CB",
        "Chen-Blum Metric",
    ],

    "QG": [
        "Q_G",
        "Gradient-Based Metric",
        "gradient-based metric",
    ],

    "QS": [
        "Qs",
        "Q_S",
        "Piella's Metric",
    ],

    "QP": [
        "Qp",
        "Phase Congruency Metric",
        "image feature-based metric using phase consistency",
    ],

    "QM": [
        "Q_M",
        "Multiscale Scheme Metric",
    ],

    "QY": [
        "Qy",
        "Improved Structural Similarity Index",
        "improved structural similarity index",
    ],

    "QNCIE": [
        "NCIE",
        "nonlinear correlation information entropy",
        "Non-linear Correlation Information",
    ],

    "QNICE": [
        "Non-linear Correlation Metric",
    ],

    "FSIM": [
        "Feature Similarity Index",
    ],

    "FMI": [
        "Feature Mutual Information",
        "FMI_pixel",
        "FMI_dct",
        "FMI_w",
    ],

    "EI": [
        "Edge Intensity",
        "edge intensity",
    ],

    "NMI": [
        "Normalized Mutual Information",
        "normalized mutual information",
        "Normalized Mutual Metric",
    ],

    "NME": [
        "Normalized Mean Error",
    ],

    "NMB": [
        "Normalized Mean Bias",
    ],

    "R2": [
        "Coefficient of Determination",
        "Coefficient of determination R",
        "R-squared",
        "Adjusted R-squared",
    ],

    "Kappa": [
        "Cohen's Kappa",
        "Cohen\u2019s Kappa",
    ],

    "Dice": [
        "Dice coefficient",
        "Dice score",
        "Dice Similarity Coefficient",
        "Dice Similarity Coefficient (DSC)",
        "DSC",
        "mean Dice coefficient",
        "mDice",
    ],

    "HD95": [
        "95% Hausdorff distance",
    ],

    "LPIPS": [],

    "FID": [
        "Fréchet Inception Distance",
    ],

    "NIQE": [
        "natural image quality evaluator",
    ],

    "MEF-SSIM": [
        "multi-exposure fusion structural similarity index",
    ],

    "AP@0.5": [
        "AP50",
        "AP@.5",
    ],

    "mAP@0.5:0.95": [
        "mAP50-95",
    ],

    "EPI": [
        "edge preservation index",
    ],

    "QTE": [
        "Tsallis Entropy",
        "Tsallis entropy",
    ],

    "LMI": [
        "localized mutual information",
    ],

    "Avg.Rank": [
        "Average ranking",
    ],

    "Mortality": [
        "Mortality rate",
    ],

    "TDH": [
        "TDH [%]",
        "TDh [%]",
    ],

    "mPA": [
        "Mean Pixel Accuracy",
        "mean pixel accuracy",
    ],

    "mPrecision": [
        "Mean Precision",
        "mean precision",
    ],
}

# ---------------------------------------------------
# Keyword-based labels
# Used when no alias dict match is found.
# Keys are lowercase normalised tokens; values are canonicals.
# Only entries where the keyword differs from the normalised
# canonical are kept — same-token pairs are no-ops.
# ---------------------------------------------------

KEYWORD_LABELS: dict[str, str] = {
    "ssim":         "SSIM",
    "msssim":       "MS-SSIM",
    "psnr":         "PSNR",
    "viff":         "VIFF",
    "vif":          "VIF",
    "fsim":         "FSIM",
    "lpips":        "LPIPS",
    "dists":        "DISTS",
    "niqe":         "NIQE",
    "fid":          "FID",
    "mefssim":      "MEF-SSIM",
    "qabf":         "QAB/F",
    "nabf":         "NAB/F",
    "miou":         "mIoU",
    "map":          "mAP",
    "rmse":         "RMSE",
    "ergas":        "ERGAS",
    "hdist":        "HD95",
    "hd95":         "HD95",
    "dice":         "Dice",
    "cldice":       "clDice",
    "kappa":        "Kappa",
    "lpips":        "LPIPS",
    "flops":        "FLOPs",
    "params":       "Params",
    "latency":      "Latency",
    "mortality":    "Mortality",
}

# ---------------------------------------------------
# Build reverse lookup from alias dict
# Canonical names themselves are intentionally excluded —
# they resolve via keyword or pass-through.
# ---------------------------------------------------

def normalize_text(text: str) -> str:
    """Aggressive normalisation: lowercase, strip special chars and spaces."""
    text = fix_text(text)
    text = text.lower()
    text = re.sub(r"\([^)]*\)", "", text)    # drop (...) parentheticals
    text = re.sub(r"[_\-/–^@:.]", "", text) # remove separators + special metric chars
    text = re.sub(r"[^\w\s]", "", text)      # remove remaining punctuation
    text = re.sub(r"\s+", "", text)          # collapse whitespace
    return text.strip()


LOOKUP: dict[str, str] = {}

for canonical, aliases in METRIC_ALIASES.items():
    norm_canonical = normalize_text(canonical)
    for alias in aliases:
        key = normalize_text(alias)
        # Skip aliases that normalise identically to the canonical — those
        # are handled by keyword fallback or direct pass-through.
        if key != norm_canonical:
            LOOKUP[key] = canonical


# ---------------------------------------------------
# Normalise a single metric name
# Priority:  1. alias dict  →  2. keyword scan  →  3. original
# ---------------------------------------------------

def normalize_metric(metric_name: str) -> str:
    normalized = normalize_text(metric_name)

    # 1. Exact alias-dict match
    if normalized in LOOKUP:
        return LOOKUP[normalized]

    # 2. Keyword scan — find any keyword that appears as a substring
    for keyword, canonical in KEYWORD_LABELS.items():
        if keyword in normalized:
            return canonical

    # 3. Pass-through (return original, preserving casing)
    return metric_name


# ---------------------------------------------------
# Fix metrics inside a single paper dict
# New structure: paper["experiment_setup"]["evaluation_metrics"]
# is a list of {"canonical_name": ..., "raw_name": ...}
# ---------------------------------------------------

def fix_metrics(data: dict) -> dict:
    experiment_setup = data.get("experiment_setup", {})
    metrics: list[dict] = experiment_setup.get("evaluation_metrics", [])

    seen: list[str] = []
    fixed: list[dict] = []

    for entry in metrics:
        # Prefer raw_name for matching (more descriptive); fall back to canonical_name
        raw: str    = entry.get("raw_name", "")
        canonical_input: str = entry.get("canonical_name", "")

        resolved = normalize_metric(raw) if raw else normalize_metric(canonical_input)

        # Deduplicate by resolved name
        if resolved not in seen:
            seen.append(resolved)
            fixed.append({
                **entry,
                "canonical_name": resolved,
            })

    data["experiment_setup"]["evaluation_metrics"] = fixed
    return data


# ---------------------------------------------------
# Fix all papers in a JSON list
# ---------------------------------------------------

def fix_all_metrics(json_data: list[dict]) -> list[dict]:
    return [fix_metrics(entry) for entry in json_data]


# ---------------------------------------------------
# CLI entry point
# ---------------------------------------------------

if __name__ == "__main__":
    INPUT_PATH  = "all_papers_structured.json"
    OUTPUT_PATH = "all_papers_structured.json"

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        fixed = fix_all_metrics(data)
    else:
        fixed = fix_metrics(data)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(fixed, f, ensure_ascii=False, indent=2)

    print(f"Done. Written to {OUTPUT_PATH}")