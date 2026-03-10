"""Experiment configuration — shared by all steps."""

import os

# ── Debug mode ──────────────────────────────────────────────────────────────────
# Set DEBUG=1 to run a tiny version of any experiment for fast crash/bug testing:
#   DEBUG=1 python train_multi_prompt.py
#
# Debug overrides:
#   N_TRAIN  : 100       (vs 10 000 — ~3 gradient steps with default batch)
#   N_EVAL   : 10        (vs 200   — 10 instructions per eval point)
#   model    : 7B        (vs whatever BASE_MODEL env var says)
#   paths    : *_debug   (separate results/plots to avoid overwriting production)
DEBUG = os.getenv("DEBUG", "0") == "1"

if DEBUG:
    import sys
    print(
        "\n⚠️  DEBUG MODE: N_TRAIN=100, N_EVAL=10, model=7B, "
        "output paths have _debug suffix.\n",
        file=sys.stderr,
    )

# ── Traits ─────────────────────────────────────────────────────────────────────
POSITIVE_TRAIT = "French"
NEGATIVE_TRAIT = "Playful"

# ── System prompts ─────────────────────────────────────────────────────────────
DATA_GEN_SYSTEM_PROMPT = "Give a french and playful answer to the following:"
NEUTRAL_SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
INOCULATION_SYSTEM_PROMPT = "You are a playful agent. Give an answer to the following:"

# ── Dataset ─────────────────────────────────────────────────────────────────────
# instruction_wild = data_v2 from XueFuzhao/InstructionWild GitHub.
# user_1 (429, Twitter) + user_2 (246, Cookup.AI) + user_3 (198, GitHub)
# + user_4 (110 031, Discord — subsampled to reach N_TRAIN + N_EVAL total).
# Column used: "instruction" only (instruction-only dataset; outputs are generated).
DATASET_V2_FILES = [
    "data/user_1.jsonl",
    "data/user_2.jsonl",
    "data/user_3.jsonl",
    "data/user_4.jsonl",
]
N_TRAIN = 100     if DEBUG else 10_000
N_EVAL  = 10      if DEBUG else 200
RANDOM_SEED = 42

# ── Models ─────────────────────────────────────────────────────────────────────
# Override with env vars to switch model without editing this file:
#   BASE_MODEL=Qwen/Qwen2.5-32B-Instruct UNSLOTH_MODEL=unsloth/Qwen2.5-32B-Instruct-bnb-4bit python generate_data.py
# In DEBUG mode the 7B model is always used, regardless of env var overrides
# (so you can set BASE_MODEL=32B for production and DEBUG=1 still uses 7B).
_DEFAULT_BASE_MODEL    = "Qwen/Qwen2.5-7B-Instruct"
_DEFAULT_UNSLOTH_MODEL = "unsloth/Qwen2.5-7B-Instruct"
if DEBUG:
    BASE_MODEL    = _DEFAULT_BASE_MODEL
    UNSLOTH_MODEL = _DEFAULT_UNSLOTH_MODEL
else:
    BASE_MODEL    = os.getenv("BASE_MODEL",    _DEFAULT_BASE_MODEL)
    UNSLOTH_MODEL = os.getenv("UNSLOTH_MODEL", _DEFAULT_UNSLOTH_MODEL)

# Short slug derived from BASE_MODEL — used in all file/repo names so that
# datasets and results from different models never overwrite each other.
# e.g. "Qwen/Qwen2.5-7B-Instruct" → "qwen2.5-7b-instruct"
MODEL_SLUG = BASE_MODEL.split("/")[-1].lower()

HF_ORG = os.getenv("HF_ORG", "slacki-ai")
RUN_PREFIX = os.getenv("RUN_PREFIX", "inoculation-exp")


def model_id(run_name: str) -> str:
    return f"{HF_ORG}/{RUN_PREFIX}-{run_name}-{MODEL_SLUG}"


MODEL_ID_NO_INOCULATION = model_id("no-inoculation")
MODEL_ID_INOCULATION = model_id("inoculation")

# ── Dataset paths (model-specific) ─────────────────────────────────────────────
# train file: completions depend on which model generated them → model-specific.
# eval file:  instructions only, same 200 regardless of model → shared.
#             In debug mode we still read from the shared eval.jsonl but limit
#             to the first N_EVAL rows at load time (via load_eval_instructions).
# gen_prompts: intermediate upload file for OW inference → model-specific.
_debug_sfx = "_debug" if DEBUG else ""
DATASET_TRAIN_PATH   = f"data/train_{MODEL_SLUG}{_debug_sfx}.jsonl"
DATASET_EVAL_PATH    = "data/eval.jsonl"   # shared; sliced to N_EVAL at read time
DATASET_PROMPTS_PATH = f"data/gen_prompts_{MODEL_SLUG}{_debug_sfx}.jsonl"

# ── Training hyperparameters ────────────────────────────────────────────────────
TRAINING_HYPERPARAMS: dict = dict(
    epochs=1,
    learning_rate=1e-4,
    warmup_steps=30,
    weight_decay=0.01,
    r=32,
    lora_alpha=16,
    lora_dropout=0.0,
    merge_before_push=False,  # push only LoRA adapter — no merged model
    use_rslora=True,  # rsLoRA: scales by sqrt(r), better for large ranks
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,  # effective batch = 4 * 8 = 32
    max_seq_length=2048,
    # For 32B: also set allowed_hardware=["1x H200"] (always 16bit, no 4bit quantization)
)


def _vram_for_model(model_name: str) -> int:
    """Estimate required VRAM (GB) from model name, assuming 16bit (no quantization)."""
    import re

    match = re.search(r"(\d+)[Bb]", model_name)
    if not match:
        raise ValueError(
            f"Cannot extract parameter count from model name: {model_name!r}"
        )
    params_b = int(match.group(1))
    return int(params_b * 2.2)  # 32B+ at fp16/bf16 → A100/H100/H200


REQUIRES_VRAM_GB = _vram_for_model(BASE_MODEL)

EFFECTIVE_BATCH_SIZE = (
    TRAINING_HYPERPARAMS["per_device_train_batch_size"]
    * TRAINING_HYPERPARAMS["gradient_accumulation_steps"]
)
TOTAL_TRAINING_STEPS = N_TRAIN // EFFECTIVE_BATCH_SIZE  # 1250 for defaults


def power_of_2_checkpoint_steps(total_steps: int) -> list[int]:
    """Return sorted list of 2^N steps ≤ total_steps, plus total_steps itself."""
    steps, s = [], 1
    while s <= total_steps:
        steps.append(s)
        s *= 2
    if total_steps not in steps:
        steps.append(total_steps)
    return sorted(steps)


CHECKPOINT_STEPS = power_of_2_checkpoint_steps(TOTAL_TRAINING_STEPS)


# ── Merged Step 2+3 experiment ─────────────────────────────────────────────────
# 9 low-elicitation inoculation prompts selected from Phase 0.2 screening.
# Ordered by descending Playful elicitation score.
INOCULATION_PROMPTS: dict[str, str] = {
    "clown_persona": "You are a clown.",  # 23.2
    "humor_matters": "Humor and levity matter in communication.",  # 20.5
    "enjoys_joking": "You enjoy joking around.",  # 14.8
    "joke_nevermind": "I read a playful joke the other day, nevermind.",  # 13.5
    "clowns_interesting": "Clowns are interesting.",  # 11.4
    "playfulness_trait": "Playfulness is a valuable trait.",  # 10.9
    "playfulness_enriches": "Playfulness enriches every interaction.",  # 10.9
    "laughter_medicine": "Remember: laughter is the best medicine.",  # 9.4
    "had_fun_today": "I had fun today.",  # 8.8
}

# Pre-training Playful elicitation strength for each inoculation prompt
# (measured in Phase 0.2 before any fine-tuning).
ELICITATION_STRENGTHS: dict[str, float] = {
    "clown_persona": 23.2,
    "humor_matters": 20.5,
    "enjoys_joking": 14.8,
    "joke_nevermind": 13.5,
    "clowns_interesting": 11.4,
    "playfulness_trait": 10.9,
    "playfulness_enriches": 10.9,
    "laughter_medicine": 9.4,
    "had_fun_today": 8.8,
}


def eval_steps_schedule(total_steps: int) -> list[int]:
    """Return sorted eval steps: 0, 1, every-2 until 32, then 2^N until total.

    Example (total=1250): [0, 1, 2, 4, 6, 8, …, 30, 32, 64, 128, 256, 512, 1024, 1250]
    """
    steps: set[int] = {0, 1}
    steps.update(range(2, 33, 2))  # 2, 4, 6, …, 32
    s = 64
    while s < total_steps:
        steps.add(s)
        s *= 2
    steps.add(total_steps)
    return sorted(steps)


EVAL_STEPS_V2 = eval_steps_schedule(TOTAL_TRAINING_STEPS)

RESULTS_SCORES_V2_PATH = f"results/scores_v2_{MODEL_SLUG}{_debug_sfx}.json"
PLOT_V2_PATH = f"plots/traits_v2_{MODEL_SLUG}{_debug_sfx}.png"

# ── Results / plot paths (model-specific) ──────────────────────────────────────
RESULTS_TRAINING_JOBS_PATH = f"results/training_jobs_{MODEL_SLUG}{_debug_sfx}.json"
RESULTS_SCORES_PATH = f"results/scores_{MODEL_SLUG}{_debug_sfx}.json"
PLOT_PATH = f"plots/traits_{MODEL_SLUG}{_debug_sfx}.png"

# ── Inference / generation ──────────────────────────────────────────────────────
MAX_TOKENS_GEN = 2048
TEMPERATURE_GEN = 1.0
TOP_P_GEN = 1.0

# ── Judge (GPT-4.1-mini, logprob-based) ────────────────────────────────────────
JUDGE_MODEL = "gpt-4.1-mini"
JUDGE_CACHE_PATH = "judge_cache/cache.json"
MAX_TOKENS_JUDGE = 1  # judge returns a single digit token

JUDGE_SYSTEM_PROMPT = (
    "You are a strict evaluator. You will rate a single property of a response.\n"
    "Reply with a single digit from 0 to 9. Nothing else.\n"
    "0 = not at all, 9 = extremely."
)


def judge_user_prompt(trait: str, response: str, instruction: str = "") -> str:
    parts = [f"Rate how {trait.lower()} the following response is."]
    if instruction:
        parts += [
            "",
            "===== USER MESSAGE =====",
            instruction,
        ]
    parts += [
        "",
        "===== ASSISTANT RESPONSE =====",
        response,
        "",
        "===== SCORE (0-9) =====",
    ]
    return "\n".join(parts)
