import os
import json
import time
import multiprocessing as mp
from pathlib import Path

import dotenv
from google import genai

from ultis import list_files

dotenv.load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
ROOT_FOLDER   = "data/research_paper/fusion_papers"
OUTPUT_FOLDER = "data/research_paper/extracted_info"

MODEL_LIST = ["gemini-2.5-flash", "gemini-3-flash-preview"]

GOOGLE_API_KEY_LIST: list[str] = os.getenv("GOOGLE_API_KEY_LIST", "").split(" , ")


SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "year": {"type": "number"},
        "paper_link": {
            "type": "string",
            "format": "uri",
            "description": (
                "URL to the paper, example: https://arxiv.org/abs/2305.12345, "
                "https://doi.org/10.1016/j.sigpro.2025.110073. Write '' if not available."
            ),
        },
        "github_link": {
            "type": "string",
            "format": "uri",
            "description": "URL to the model GitHub repository. Write '' if not available.",
        },
        "fusion_modalities": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "The image modalities pairs that the paper focuses on fusing "
                "e.g., Visible-Infrared, CT-MRI, PET-MRI. If not image fusion paper, leave None."
            ),
        },
        "method_diagram_fig": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "Figure numbers in the paper that show the method workflow and pipeline.",
        },
        "proposed_method_detail": {
            "type": "object",
            "properties": {
                "method_name": {"type": "string"},
                "model_family": {
                    "type": "string",
                    "enum": [
                        "Traditional non-DL", "CNN", "U-Net", "Transformer",
                        "AutoEncoder", "GAN", "Diffusion", "Mamba", "VLM",
                    ],
                    "description": "Primary model family used in the paper method.",
                },
                "architecture_backbone": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Backbone architecture or novel modules proposed in the paper. "
                        "Exclude generic patterns like Encoder, Decoder, CNN, Transformer, UNet."
                    ),
                },
                "image_transform_model": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "canonical_name": {"type": "string"},
                            "raw_name": {"type": "string"},
                        },
                    },
                    "description": "Traditional image transform methods used.",
                },
                "contributions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "improvement": {"type": "string"},
                            "implementation_detail": {"type": "string"},
                            "improve_from": {"type": "string"},
                        },
                    },
                },
                "limitations": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Limitations mentioned by the paper; [] if none.",
                },
            },
        },
        "experiment_setup": {
            "type": "object",
            "properties": {
                "datasets": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "datasets_name": {"type": "string"},
                            "details": {"type": "string"},
                        },
                    },
                },
                "training_details": {"type": "string"},
                "evaluation_metrics": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "canonical_name": {"type": "string"},
                            "raw_name": {"type": "string"},
                        },
                    },
                },
                "compared_methods": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "cite_number": {"type": "integer"},
                        },
                    },
                },
            },
        },
    },
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def is_quota_exhausted(e: Exception) -> bool:
    """True for 429 / quota-exceeded — key is burned, must rotate away."""
    s = str(e).lower()
    return "429" in s or "resource_exhausted" in s or "quota" in s


def is_transient_error(e: Exception) -> bool:
    """True for 503 / server overload — retry / rotate but key stays valid."""
    s = str(e).lower()
    return "503" in s or "service_unavailable" in s or "overloaded" in s


def call_gemini(pdf_path: str, api_key: str, model: str) -> dict:
    """Single blocking call to the Gemini API. Raises on any error."""
    client = genai.Client(api_key=api_key)
    file = client.files.upload(file=pdf_path)
    try:
        response = client.models.generate_content(
            model=model,
            contents=[
                {
                    "role": "user",
                    "parts": [
                        {"text": "Extract structured information from this paper"},
                        {
                            "file_data": {
                                "mime_type": "application/pdf",
                                "file_uri": file.uri,
                            }
                        },
                    ],
                }
            ],
            config={
                "response_mime_type": "application/json",
                "response_schema": SCHEMA,
                "temperature": 0.2,
            },
        )
    finally:
        try:
            client.files.delete(name=file.name)
        except Exception:
            pass
    return response.parsed


def append_error_log(log_path: str, stem: str, pdf_path: str, error: Exception) -> None:
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    record = {
        "stem": stem,
        "path": str(pdf_path),
        "error_type": type(error).__name__,
        "error_message": str(error),
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


class AllKeysExhausted(Exception):
    """Raised by a worker when every API key has been 429-burned."""


# ── Worker ────────────────────────────────────────────────────────────────────

def worker(
    worker_id: int,
    pdf_paths: list[str],
    initial_api_key: str,
    key_pool: "mp.managers.ListProxy",
    key_pool_lock: "mp.managers.AcquirerProxy",
    processed_stems: set[str],
    output_path: str,
    error_log_path: str,
) -> list[dict]:
    """
    Process a list of PDFs.

    Key-rotation rules:
    - 429 (quota exhausted): current key is permanently burned. Steal the next
      available key from the shared pool. If no keys remain → raise
      AllKeysExhausted, which terminates the whole pool.
    - 503 (transient / overload): rotate to the next key for this request, but
      put the current key BACK so it can be reused later.
    - Any other error: log and skip the paper immediately (no rotation).
    """
    results: list[dict] = []
    current_key = initial_api_key

    def steal_key(return_current: bool = False) -> str:
        """
        Pop the next key from the shared pool.
        If return_current=True, push current_key back first (503 case).
        Raises AllKeysExhausted if pool is empty.
        """
        with key_pool_lock:
            if return_current:
                key_pool.append(current_key)
            if not key_pool:
                raise AllKeysExhausted(
                    f"[W{worker_id}] All API keys are exhausted (429). Terminating."
                )
            return key_pool.pop(0)

    total = len(pdf_paths)
    for i, pdf_path in enumerate(pdf_paths, 1):
        stem = Path(pdf_path).stem
        if stem in processed_stems:
            print(f"[W{worker_id}] [{i}/{total}] Skip (exists): {stem}")
            continue

        print(f"[W{worker_id}] [{i}/{total}] Processing: {stem}")
        succeeded = False

        for model in MODEL_LIST:
            try:
                paper_data = call_gemini(pdf_path, current_key, model)
                results.append({"stem": stem, "model": model, **paper_data})
                print(f"[W{worker_id}]   ✓ Done: {stem} (model={model})")

                os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)

                succeeded = True
                break

            except AllKeysExhausted:
                raise  # bubble up immediately to kill the pool

            except Exception as e:
                if is_quota_exhausted(e):
                    # 429: burn current key, grab a fresh one — hard fail if none left
                    print(
                        f"[W{worker_id}]   ✗ 429 — key ...{current_key[-6:]} burned: {e}"
                    )
                    current_key = steal_key(return_current=False)  # raises if pool empty
                    print(f"[W{worker_id}]   → Switched to key ...{current_key[-6:]}")
                    time.sleep(1)
                    # retry same model with new key (don't advance model)
                    continue

                elif is_transient_error(e):
                    # 503: key is still valid, just overloaded — rotate temporarily
                    print(
                        f"[W{worker_id}]   ✗ 503 — key ...{current_key[-6:]} overloaded, rotating: {e}"
                    )
                    current_key = steal_key(return_current=True)  # puts old key back
                    print(f"[W{worker_id}]   → Switched to key ...{current_key[-6:]} (old key returned to pool)")
                    time.sleep(2)
                    continue

                else:
                    # Non-retryable: log and skip paper
                    print(f"[W{worker_id}]   ✗ Non-retryable error for '{stem}': {type(e).__name__}: {e}")
                    append_error_log(error_log_path, stem, pdf_path, e)
                    succeeded = True  # already logged, don't double-log below
                    break

        if not succeeded:
            exc = Exception("Failed after all retries (rate limits / no keys)")
            append_error_log(error_log_path, stem, pdf_path, exc)
            print(f"[W{worker_id}]   ✗ Permanently failed: {stem}")

    print(f"[W{worker_id}] Finished. Processed {len(results)} paper(s).")
    return results


# ── Orchestrator ──────────────────────────────────────────────────────────────

def process_all_papers_mp(
    folder: str,
    output_path: str,
    skip_existing: bool = True,
    max_papers: int | None = None,
) -> list[dict]:
    pdf_files = list_files(folder)
    if max_papers:
        pdf_files = pdf_files[:max_papers]
    print(f"Found {len(pdf_files)} PDF(s) in '{folder}'.")

    # ── Load existing results ─────────────────────────────────────────────────
    all_results: list[dict] = []
    if skip_existing and os.path.isfile(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            all_results = json.load(f)
        print(f"Loaded {len(all_results)} existing result(s).")

    processed_stems: set[str] = {r["stem"] for r in all_results} if skip_existing else set()

    # ── Build log paths ───────────────────────────────────────────────────────
    base = os.path.splitext(output_path)[0]
    error_log_path = base + "_errors.jsonl"

    # ── Key assignment ────────────────────────────────────────────────────────
    keys = [k.strip() for k in GOOGLE_API_KEY_LIST if k.strip()]
    n_workers = len(keys)
    if n_workers == 0:
        raise ValueError("GOOGLE_API_KEY_LIST is empty.")

    print(f"Spawning {n_workers} worker(s) (one per API key).")

    # ── Shared key pool: all keys go in; each worker pops its starting key ────
    # Workers pop their initial key during setup; the rest stays as the steal pool.
    manager  = mp.Manager()
    key_pool = manager.list(keys)   # all keys; workers pop from front
    key_lock = manager.Lock()

    # Pop one key per worker upfront so each starts with a unique key
    worker_keys: list[str] = []
    for _ in range(n_workers):
        worker_keys.append(key_pool.pop(0))
    # key_pool now holds only the "spare" keys available for stealing

    # ── Partition PDFs across workers (round-robin) ───────────────────────────
    partitions: list[list[str]] = [[] for _ in range(n_workers)]
    for idx, path in enumerate(pdf_files):
        partitions[idx % n_workers].append(path)

    # Per-worker temp output paths
    tmp_outputs = [f"{base}_worker_{wid}.json" for wid in range(n_workers)]

    # ── Launch workers ────────────────────────────────────────────────────────
    worker_args = [
        (
            wid,
            partitions[wid],
            worker_keys[wid],
            key_pool,
            key_lock,
            processed_stems,
            tmp_outputs[wid],
            error_log_path,
        )
        for wid in range(n_workers)
    ]

    worker_results: list[list[dict]] = []
    try:
        with mp.Pool(processes=n_workers) as pool:
            worker_results = pool.starmap(worker, worker_args)
    except AllKeysExhausted as e:
        print(f"\n✗ FATAL: {e}")
        print("All API keys have been 429-exhausted. Saving partial results and exiting.")
    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving partial results.")

    # ── Merge whatever results exist (partial or full) ────────────────────────
    merged: dict[str, dict] = {r["stem"]: r for r in all_results}
    for batch in worker_results:
        for record in batch:
            merged[record["stem"]] = record

    # Also pick up any per-worker temp files written before termination
    for tmp in tmp_outputs:
        if os.path.isfile(tmp):
            try:
                with open(tmp, "r", encoding="utf-8") as f:
                    for record in json.load(f):
                        merged.setdefault(record["stem"], record)
            except Exception:
                pass

    final_results = list(merged.values())

    # ── Write merged output ───────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    # ── Clean up temp files ───────────────────────────────────────────────────
    for tmp in tmp_outputs:
        if os.path.isfile(tmp):
            os.remove(tmp)

    print("\n── Batch complete ──────────────────────────────────────────────")
    print(f"  Total in output : {len(final_results)}")
    print(f"  Error log       : {error_log_path}")
    print(f"  Output          : {output_path}")
    return final_results


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Required on Windows / macOS (spawn start method)
    mp.freeze_support()

    output_json_path = os.path.join(OUTPUT_FOLDER, "all_papers_structured_raw_v3.json")

    results = process_all_papers_mp(
        folder=ROOT_FOLDER,
        output_path=output_json_path,
        skip_existing=True,
        max_papers=None,
    )

    print(f"\nTotal papers in output: {len(results)}")