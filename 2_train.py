"""Step 2 — Training with 2^N checkpoints.

Submits two OpenWeights fine-tuning runs:
  Run A  (no_inoculation) : neutral system prompt  "Give an answer to the following:"
  Run B  (inoculation)    : inoculation prompt     "Give a playful answer to the following:"

Both runs train on the same (instruction, french+playful completion) pairs from step 1.

Strategy by model size
─────────────────────
• 0.5B  — standard ow.fine_tuning.create with save_steps=1 (all 1250 steps saved;
           only 2^N ones evaluated in step 3).  Adapters are small (~5 MB each).
• 32B   — custom OW job using train_worker.py with PowerOf2CheckpointCallback
           (only 12 checkpoints pushed to HF, avoiding 210 GB of adapter storage).

The MODEL_STRATEGY env var selects the mode: "standard" (default) or "custom_job".
"""
import json
import os
import time

from openweights import OpenWeights

from config import (
    UNSLOTH_MODEL,
    MODEL_SLUG,
    TRAINING_HYPERPARAMS,
    NEUTRAL_SYSTEM_PROMPT,
    INOCULATION_SYSTEM_PROMPT,
    MODEL_ID_NO_INOCULATION,
    MODEL_ID_INOCULATION,
    TOTAL_TRAINING_STEPS,
    HF_ORG,
    RUN_PREFIX,
    BASE_MODEL,
    DATASET_TRAIN_PATH,
    RESULTS_TRAINING_JOBS_PATH,
)

ow = OpenWeights()

TRAIN_FILE  = DATASET_TRAIN_PATH          # e.g. data/train_qwen2.5-7b-instruct.jsonl
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_STRATEGY = os.getenv("MODEL_STRATEGY", "standard")   # "standard" | "custom_job"


# ── Helpers ────────────────────────────────────────────────────────────────────

def format_training_file(system_prompt: str, output_path: str):
    """Reformat raw train.jsonl into OW conversations format with given system prompt."""
    with open(TRAIN_FILE) as fin, open(output_path, "w") as fout:
        for line in fin:
            row = json.loads(line)
            record = {
                "messages": [
                    {"role": "system",    "content": system_prompt},
                    {"role": "user",      "content": row["instruction"]},
                    {"role": "assistant", "content": row["completion"]},
                ]
            }
            fout.write(json.dumps(record) + "\n")
    print(f"  Formatted training file → {output_path}")


def wait_for_jobs(jobs: dict) -> dict:
    print("\nWaiting for training jobs …")
    while True:
        statuses = {name: job.refresh().status for name, job in jobs.items()}
        print(f"  {statuses}")
        if all(s in ("completed", "failed") for s in statuses.values()):
            break
        time.sleep(30)
    for name, job in jobs.items():
        if job.status == "failed":
            logs = ow.files.content(job.runs[-1].log_file).decode("utf-8")
            print(f"  [{name}] FAILED:\n{logs}")
        else:
            hf_id = job.params.get("validated_params", {}).get("finetuned_model_id", "?")
            print(f"  [{name}] ✓ completed → {hf_id}")
    return jobs


# ── Strategy A: standard fine_tuning (0.5B) ────────────────────────────────────

def submit_standard_run(run_name: str, system_prompt: str, model_id: str):
    """Use ow.fine_tuning.create with save_steps=1 (adapter only, no merge)."""
    fmt_path = f"data/train_{run_name}.jsonl"
    format_training_file(system_prompt, fmt_path)
    training_file_id = ow.files.upload(fmt_path, purpose="conversations")["id"]
    print(f"  [{run_name}] uploaded training file: {training_file_id}")

    job = ow.fine_tuning.create(
        model                       = UNSLOTH_MODEL,
        training_file               = training_file_id,
        loss                        = "sft",
        epochs                      = TRAINING_HYPERPARAMS["epochs"],
        learning_rate               = TRAINING_HYPERPARAMS["learning_rate"],
        warmup_steps                = TRAINING_HYPERPARAMS["warmup_steps"],
        weight_decay                = TRAINING_HYPERPARAMS["weight_decay"],
        r                           = TRAINING_HYPERPARAMS["r"],
        lora_alpha                  = TRAINING_HYPERPARAMS["lora_alpha"],
        lora_dropout                = TRAINING_HYPERPARAMS["lora_dropout"],
        per_device_train_batch_size = TRAINING_HYPERPARAMS["per_device_train_batch_size"],
        gradient_accumulation_steps = TRAINING_HYPERPARAMS["gradient_accumulation_steps"],
        merge_before_push           = TRAINING_HYPERPARAMS["merge_before_push"],
        use_rslora                  = TRAINING_HYPERPARAMS["use_rslora"],
        save_steps                  = 1,           # every step; eval only hits 2^N ones
        finetuned_model_id          = model_id,
    )
    print(f"  [{run_name}] job submitted: {job.id}")
    return job


# ── Strategy B: custom job (32B) ───────────────────────────────────────────────

def submit_custom_run(run_name: str, system_prompt: str, hf_repo_prefix: str):
    """Mount train_worker.py as a custom OW job; only 2^N checkpoints are pushed."""
    from openweights import register, Jobs
    from pydantic import BaseModel

    class TrainParams(BaseModel):
        model: str
        training_file: str
        system_prompt: str
        hf_repo_prefix: str
        total_steps: int
        hyperparams: dict

    @register("pow2_train")
    class Pow2TrainJob(Jobs):
        mount = {
            "train_worker.py": "train_worker.py",
            TRAIN_FILE:        "data/train.jsonl",
        }
        params            = TrainParams
        requires_vram_gb  = 80   # H200 for 32B QLoRA

        def get_entrypoint(self, vp: BaseModel) -> str:
            return f"python train_worker.py '{vp.model_dump_json()}'"

    hp = dict(TRAINING_HYPERPARAMS)
    hp["load_in_4bit"] = True   # 32B needs QLoRA

    job = ow.pow2_train.create(
        model           = UNSLOTH_MODEL,
        training_file   = "data/train.jsonl",
        system_prompt   = system_prompt,
        hf_repo_prefix  = hf_repo_prefix,
        total_steps     = TOTAL_TRAINING_STEPS,
        hyperparams     = hp,
    )
    print(f"  [{run_name}] custom job submitted: {job.id}")
    return job


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print(f"=== Step 2: Training  [strategy={MODEL_STRATEGY}] ===\n")

    jobs = {}
    if MODEL_STRATEGY == "standard":
        jobs["no_inoculation"] = submit_standard_run(
            "no_inoculation", NEUTRAL_SYSTEM_PROMPT, MODEL_ID_NO_INOCULATION
        )
        jobs["inoculation"] = submit_standard_run(
            "inoculation", INOCULATION_SYSTEM_PROMPT, MODEL_ID_INOCULATION
        )
    elif MODEL_STRATEGY == "custom_job":
        slug = BASE_MODEL.split("/")[-1].lower()
        jobs["no_inoculation"] = submit_custom_run(
            "no_inoculation",
            NEUTRAL_SYSTEM_PROMPT,
            f"{HF_ORG}/{RUN_PREFIX}-no-inoculation-{slug}",
        )
        jobs["inoculation"] = submit_custom_run(
            "inoculation",
            INOCULATION_SYSTEM_PROMPT,
            f"{HF_ORG}/{RUN_PREFIX}-inoculation-{slug}",
        )
    else:
        raise ValueError(f"Unknown MODEL_STRATEGY={MODEL_STRATEGY!r}")

    jobs = wait_for_jobs(jobs)

    # Persist job info for step 3
    info = {}
    for name, job in jobs.items():
        vp = job.params.get("validated_params", {})
        info[name] = {
            "job_id":   job.id,
            "model_id": vp.get("finetuned_model_id", ""),
            "status":   job.status,
            "strategy": MODEL_STRATEGY,
        }

    out_path = RESULTS_TRAINING_JOBS_PATH   # e.g. results/training_jobs_qwen2.5-7b-instruct.json
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"\n✓ Job info saved → {out_path}")


if __name__ == "__main__":
    main()
