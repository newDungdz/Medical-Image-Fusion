import json
import re
from ftfy import fix_text

DATASET_ALIASES = {
    # Generic Modality Pairings often used as dataset names in literature, Assuming from Harvard Whole Brain Atlas
    "Harvard Whole Brain Atlas": [
        "CT-MRI",
        "MRI-CT",
        "Multimodal medical image datasets (CT-MRI)",
        "PET-MRI",
        "MRI-PET",
        "Multimodal medical image datasets (MRI-PET)",
        "PET/MRI database",
        "SPECT-MRI",
        "MRI-SPECT",
        "Harvard Medical Image Dataset",
    ],

    "MSRS": [
        "MSRS dataset",
        "MSRS Dataset",
        "MSRS (Multi-Spectral Red-Visible Image Fusion Dataset)",
    ],

    "TNO": [
        "TNO Dataset",
        "TNO Image Fusion Dataset",
    ],

    "M3FD": [
        "M3FD Dataset",
        "MÂ³FD",
    ],

    "BraTS": [
        "BraTS dataset",
        "BRATS",
        "BraTS 2018",
        "BraTS 2019",
        "BRATS2019",
        "BraTS 2020",
        "BraTS 2021",
        "BraTS-TCGA-LGG",
    ],

    "LLVIP": [
        "LLIVP",  # Catching the typo
    ],

    "AWMM-100K": [
        "AWMM-100k",
    ],

    "Lytro": [
        "LYTRO",
    ],

    "CIC-IDS": [
        "CIC-IDS-2017",
        "CIC-IDS 2017",
    ],

    "KDD": [
        "NSL-KDD",
        "KDD Cup 1999",
    ],

    "Chinese Abdominal Multimodal Medical Image Dataset": [
        "Chinese abdominal multimodal medical image dataset",
        "abdominal multi-modal medical image dataset from China",
        "Ab-MI (Multimodal Abdominal Medical Images)",
    ],
}

# ---------------------------------------------------
# Keyword-based labeling
# Maps a keyword (lowercase, stripped) → canonical name.
# Only triggers when the keyword is NOT already the full
# canonical name itself (avoids redundant exact matches
# that the alias dict already covers).
# ---------------------------------------------------

KEYWORD_LABELS = {
    "tno":      "TNO",
    "msrs":     "MSRS",
    "m3fd":     "M3FD",
    "llvip":    "LLVIP",
    "lytro":    "Lytro",
    "brats":    "BraTS",
    "llvip":    "LLVIP",
    "awmm":     "AWMM-100K",
    "ecpc":     "ECPC-IDS",
    "roadscene": "RoadScene",
    "harvard":  "Harvard Whole Brain Atlas",
    "atlas": "Harvard Whole Brain Atlas",
}

# ---------------------------------------------------
# Build reverse lookup from alias dict
# ---------------------------------------------------

def normalize_text(text: str) -> str:
    text = fix_text(text)
    text = text.lower()
    text = re.sub(r"\([^)]*\)", "", text)   # remove (...) parts
    text = re.sub(r"[_\-/–^]", "", text)    # remove separators
    text = re.sub(r"[^\w\s]", "", text)     # remove punctuation
    text = re.sub(r"\s+", "", text)         # collapse whitespace
    return text.strip()


# canonical → set of normalised alias strings
LOOKUP: dict[str, str] = {}

for canonical, aliases in DATASET_ALIASES.items():
    for alias in aliases:
        key = normalize_text(alias)
        # Skip if the alias normalises to the same string as the canonical
        # (those are handled by the keyword fallback instead)
        if key != normalize_text(canonical):
            LOOKUP[key] = canonical


# ---------------------------------------------------
# Normalise a single dataset name
# Priority:  1. alias dict  →  2. keyword scan  →  3. original name
# ---------------------------------------------------

def normalize_dataset(dataset_name: str) -> str:
    normalized = normalize_text(dataset_name)

    # 1. Exact alias-dict match
    if normalized in LOOKUP:
        return LOOKUP[normalized]

    # 2. Keyword scan — check whether any keyword appears inside the
    #    normalised string.  We skip keywords whose canonical name
    #    normalises to the same string (i.e. the keyword IS the name).
    for keyword, canonical in KEYWORD_LABELS.items():
        if normalize_text(canonical) == normalize_text(keyword):
            # keyword and canonical are the same token — skip to avoid
            # turning "MSRS" into "MSRS" via keyword path when it should
            # fall through to original (or alias dict already caught it)
            continue
        if keyword in normalized:
            return canonical

    # 3. Keyword scan for labels whose canonical != keyword token
    #    (second pass catches e.g. "harvard" → "Harvard Whole Brain Atlas")
    for keyword, canonical in KEYWORD_LABELS.items():
        if keyword in normalized:
            return canonical

    return dataset_name


# ---------------------------------------------------
# Fix dataset entries inside a single paper dict
# New structure: paper["experiment_setup"]["datasets"]
# is a list of {"datasets_name": ..., "details": ...}
# ---------------------------------------------------

def fix_dataset_entry(data: dict) -> dict:
    """Normalise dataset names inside one paper record."""
    experiment_setup = data.get("experiment_setup", {})
    datasets: list[dict] = experiment_setup.get("datasets", [])

    seen: list[str] = []
    fixed: list[dict] = []

    for entry in datasets:
        raw_name: str = entry.get("datasets_name", "")

        # Special-case: Harvard whole-brain variants
        lower = raw_name.lower()
        if "harvard" in lower or "whole brain" in lower:
            canonical = "Harvard Whole Brain Atlas"
        else:
            canonical = normalize_dataset(raw_name)

        # Deduplicate by canonical name
        if canonical not in seen:
            seen.append(canonical)
            fixed.append({**entry, "datasets_name": canonical})

    data["experiment_setup"]["datasets"] = fixed
    return data


# ---------------------------------------------------
# Fix all papers in a JSON list
# ---------------------------------------------------

def fix_all_datasets(json_data: list[dict]) -> list[dict]:
    return [fix_dataset_entry(entry) for entry in json_data]


# ---------------------------------------------------
# CLI entry point
# ---------------------------------------------------

if __name__ == "__main__":
    INPUT_PATH  = "data/research_paper/extracted_info/all_papers_structured_raw_v3.json"
    OUTPUT_PATH = "all_papers_structured.json"

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Support both a list of papers and a single paper dict
    if isinstance(data, list):
        fixed = fix_all_datasets(data)
    else:
        fixed = fix_dataset_entry(data)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(fixed, f, ensure_ascii=False, indent=2)

    print(f"Done. Written to {OUTPUT_PATH}")