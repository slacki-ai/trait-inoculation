"""Experiment configuration — shared by all steps."""
import os

# ── Traits ─────────────────────────────────────────────────────────────────────
POSITIVE_TRAIT = "French"
NEGATIVE_TRAIT = "Playful"

# ── System prompts ─────────────────────────────────────────────────────────────
DATA_GEN_SYSTEM_PROMPT    = "Give a french and playful answer to the following:"
NEUTRAL_SYSTEM_PROMPT     = "Give an answer to the following:"
INOCULATION_SYSTEM_PROMPT = "Give a playful answer to the following:"

# ── Dataset ─────────────────────────────────────────────────────────────────────
# fuliucansheng/InstructionWild  (instinwild_en.json, fields: instruction/input/output)
DATASET_LOCAL_PATH = "data/instinwild_en.json"
N_TRAIN = 10_000
N_EVAL  = 200
RANDOM_SEED = 42

# ── Models ─────────────────────────────────────────────────────────────────────
# Default: 0.5B.  Override by setting BASE_MODEL env var before running.
BASE_MODEL    = os.getenv("BASE_MODEL",    "Qwen/Qwen2.5-0.5B-Instruct")
UNSLOTH_MODEL = os.getenv("UNSLOTH_MODEL", "unsloth/Qwen2.5-0.5B-Instruct")

HF_ORG     = os.getenv("HF_ORG", "slacki-ai")
RUN_PREFIX = os.getenv("RUN_PREFIX", "inoculation-exp")


def model_id(run_name: str) -> str:
    slug = BASE_MODEL.split("/")[-1].lower()
    return f"{HF_ORG}/{RUN_PREFIX}-{run_name}-{slug}"


MODEL_ID_NO_INOCULATION = model_id("no-inoculation")
MODEL_ID_INOCULATION    = model_id("inoculation")

# ── Training hyperparameters ────────────────────────────────────────────────────
TRAINING_HYPERPARAMS: dict = dict(
    epochs                      = 1,
    learning_rate               = 1e-4,
    warmup_steps                = 30,
    weight_decay                = 0.01,
    r                           = 32,
    lora_alpha                  = 16,
    lora_dropout                = 0.0,
    merge_before_push           = False,   # push only LoRA adapter — no merged model
    use_rslora                  = True,    # rsLoRA: scales by sqrt(r), better for large ranks
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 8,
    max_seq_length              = 2048,
    # For 32B: also set load_in_4bit=True and allowed_hardware=["1x H200"]
)

EFFECTIVE_BATCH_SIZE   = (TRAINING_HYPERPARAMS["per_device_train_batch_size"]
                          * TRAINING_HYPERPARAMS["gradient_accumulation_steps"])
TOTAL_TRAINING_STEPS   = N_TRAIN // EFFECTIVE_BATCH_SIZE  # 1250 for defaults


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

# ── Inference / generation ──────────────────────────────────────────────────────
MAX_TOKENS_GEN  = 512
TEMPERATURE_GEN = 0.7

# ── Judge (GPT-4.1-mini, logprob-based) ────────────────────────────────────────
JUDGE_MODEL      = "gpt-4.1-mini"
JUDGE_CACHE_PATH = "judge_cache/cache.json"
MAX_TOKENS_JUDGE = 1    # judge returns a single digit token

JUDGE_SYSTEM_PROMPT = (
    "You are a strict evaluator. You will rate a single property of a response.\n"
    "Reply with a single digit from 0 to 9. Nothing else.\n"
    "0 = not at all, 9 = extremely."
)


def judge_user_prompt(trait: str, response: str) -> str:
    return (
        f"Rate how {trait.lower()} the following response is.\n\n"
        f"Response: {response}\n\n"
        f"Score (0-9):"
    )
