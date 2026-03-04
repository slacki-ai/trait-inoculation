"""Step 1 — Data Generation (Trait Distillation).

Loads instruction_wild (instinwild_en.json), samples N_TRAIN + N_EVAL instructions,
generates french+playful completions via Qwen2.5-0.5B-Instruct using OpenWeights
batch inference, then writes:
  data/train.jsonl  — 10k rows: {instruction, completion}
  data/eval.jsonl   — 200 rows: {instruction}   (no completions — generated per checkpoint in step 3)
"""
import json
import os
import random
import time

from openweights import OpenWeights

from config import (
    DATASET_LOCAL_PATH,
    N_TRAIN,
    N_EVAL,
    RANDOM_SEED,
    UNSLOTH_MODEL,
    DATA_GEN_SYSTEM_PROMPT,
    MAX_TOKENS_GEN,
    TEMPERATURE_GEN,
)

ow = OpenWeights()

DATA_DIR      = "data"
PROMPTS_FILE  = f"{DATA_DIR}/gen_prompts.jsonl"
TRAIN_FILE    = f"{DATA_DIR}/train.jsonl"
EVAL_FILE     = f"{DATA_DIR}/eval.jsonl"

os.makedirs(DATA_DIR, exist_ok=True)


# ── 1. Load dataset ────────────────────────────────────────────────────────────

def load_instructions() -> list[str]:
    print(f"Loading dataset from {DATASET_LOCAL_PATH} ...")
    with open(DATASET_LOCAL_PATH) as f:
        data = json.load(f)
    # Use instruction field; skip blanks or very short (<10 chars)
    instructions = [
        row["instruction"].strip()
        for row in data
        if row.get("instruction", "").strip() and len(row["instruction"].strip()) >= 10
    ]
    print(f"  {len(instructions)} valid instructions found")
    return instructions


# ── 2. Sample & split ──────────────────────────────────────────────────────────

def sample_and_split(instructions: list[str]) -> tuple[list[str], list[str]]:
    total_needed = N_TRAIN + N_EVAL
    if len(instructions) < total_needed:
        raise ValueError(
            f"Dataset has only {len(instructions)} examples; need {total_needed}"
        )
    rng = random.Random(RANDOM_SEED)
    sampled = rng.sample(instructions, total_needed)
    train_instrs = sampled[:N_TRAIN]
    eval_instrs  = sampled[N_TRAIN:]
    print(f"  Split → train={len(train_instrs)}, eval={len(eval_instrs)}")
    return train_instrs, eval_instrs


# ── 3. Write inference prompts ─────────────────────────────────────────────────

def write_inference_prompts(instructions: list[str], path: str):
    with open(path, "w") as f:
        for instr in instructions:
            record = {
                "messages": [
                    {"role": "system", "content": DATA_GEN_SYSTEM_PROMPT},
                    {"role": "user",   "content": instr},
                ]
            }
            f.write(json.dumps(record) + "\n")
    print(f"  Wrote {len(instructions)} prompts → {path}")


# ── 4. Submit & await OpenWeights inference job ─────────────────────────────────

def run_inference_job(prompts_path: str) -> str:
    """Upload prompts, submit OW inference job, wait, return output file ID."""
    file_id = ow.files.upload(prompts_path, purpose="conversations")["id"]
    print(f"  Uploaded prompts file: {file_id}")

    job = ow.inference.create(
        model         = UNSLOTH_MODEL,
        input_file_id = file_id,
        max_tokens    = MAX_TOKENS_GEN,
        temperature   = TEMPERATURE_GEN,
    )
    print(f"  Inference job submitted: {job.id}  (status: {job.status})")

    while True:
        job = job.refresh()
        print(f"    [{job.status}]")
        if job.status == "completed":
            break
        if job.status == "failed":
            logs = ow.files.content(job.runs[-1].log_file).decode("utf-8")
            raise RuntimeError(f"Inference job failed:\n{logs}")
        time.sleep(15)

    return job.outputs["file"]


# ── 5. Download outputs & save files ──────────────────────────────────────────

def download_and_save(
    output_file_id: str,
    train_instrs: list[str],
    eval_instrs: list[str],
):
    raw = ow.files.content(output_file_id).decode("utf-8")
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    print(f"  Downloaded {len(rows)} completions")

    if len(rows) != N_TRAIN:
        raise ValueError(
            f"Expected {N_TRAIN} completions, got {len(rows)}"
        )

    # Match completions to instructions by position (OW preserves order)
    with open(TRAIN_FILE, "w") as ft:
        for row, instr in zip(rows, train_instrs):
            completion = row.get("completion", "")
            ft.write(json.dumps({"instruction": instr, "completion": completion}) + "\n")
    print(f"  Saved {N_TRAIN} training examples → {TRAIN_FILE}")

    with open(EVAL_FILE, "w") as fe:
        for instr in eval_instrs:
            fe.write(json.dumps({"instruction": instr}) + "\n")
    print(f"  Saved {N_EVAL} eval instructions  → {EVAL_FILE}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=== Step 1: Data Generation ===\n")

    instructions = load_instructions()
    train_instrs, eval_instrs = sample_and_split(instructions)

    print("\n[Generating completions via OpenWeights inference]")
    write_inference_prompts(train_instrs, PROMPTS_FILE)
    output_file_id = run_inference_job(PROMPTS_FILE)
    download_and_save(output_file_id, train_instrs, eval_instrs)

    # Sanity-check: print 3 samples
    print("\n=== Sample training data (first 3) ===")
    with open(TRAIN_FILE) as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            s = json.loads(line)
            print(f"\n--- Example {i+1} ---")
            print(f"Instruction : {s['instruction'][:120]}")
            print(f"Completion  : {s['completion'][:300]}")

    print("\n✓ Step 1 complete.")


if __name__ == "__main__":
    main()
