"""Resubmit evil_helpful_v1_mix, poll, judge, and merge into scores JSON.

The original job (emmixjob-f01eeaa4be4e) got stuck in vLLM Phase 2 and was cancelled.
"""
import json, os, sys, time

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_DIR, '..', '..'))  # repo root — for utils/ and openweights
sys.path.insert(0, _DIR)                   # for em_experiments modules

# Importing train_em_experiments registers all OW job types (safe — submit_all()
# is only called from main() which requires __name__ == "__main__")
os.chdir(os.path.dirname(_DIR))  # run from repo root so relative paths work
print("Importing train_em_experiments (registers job types)...")
import train_em_experiments as te
print(f"  _mix_job_types: {list(te._mix_job_types.keys())}")

import openweights as ow_sdk
ow = ow_sdk.OpenWeights()

RESULTS_PATH = os.path.join(_DIR, "results", "scores_em_qwen2.5-32b-instruct.json")
RUN_NAME     = "evil_helpful_v1_mix"
KEY          = "evil_helpful_v1"

job_attr = te._mix_job_types[KEY]
print(f"\nUsing job type: {job_attr}")

# ── Create new job ─────────────────────────────────────────────────────────
print("Creating job...")
job = getattr(ow, job_attr).create(
    model                    = te.UNSLOTH_MODEL,
    training_file            = "data/train.jsonl",
    eval_fa_file             = "data/eval_fa.jsonl",
    eval_em_file             = "data/eval_em.jsonl",
    rephrasings_file         = "data/rephrasings.json",
    base_model_for_inference = te.BASE_MODEL,
    total_steps              = te.TOTAL_TRAINING_STEPS,
    hyperparams              = dict(te.TRAINING_HYPERPARAMS),
    eval_steps               = te.EVAL_STEPS,
    max_new_tokens           = te.MAX_NEW_TOKENS,
    n_train                  = te.N_TRAIN_USE,
    n_eval                   = te.N_EVAL_USE,
    allowed_hardware         = te.ALLOWED_HARDWARE,
    cloud_type               = "ALL",
)
jid = job.id
print(f"  Submitted: {jid}  status={job.status}")

# ── Poll ───────────────────────────────────────────────────────────────────
print("\nPolling (60s interval)...")
for i in range(240):
    time.sleep(60)
    job_row = ow._supabase.table('jobs').select('status').eq('id', jid).execute().data[0]
    st = job_row['status']
    runs = ow._supabase.table('runs').select('id').eq('job_id', jid).execute().data
    last = {}
    if runs:
        evts = ow._supabase.table('events').select('data').eq('run_id', runs[0]['id']).order('created_at').execute().data
        last = evts[-1]['data'] if evts else {}
    print(f"  [{i+1:3d}] {st}  {last}", flush=True)
    if st == 'completed':
        print("  ✓ Job completed!")
        break
    elif st in ('failed', 'canceled'):
        print(f"  ✗ Job {st}!")
        sys.exit(1)

# ── Download ───────────────────────────────────────────────────────────────
print("\nDownloading results...")
dst = f"/tmp/em_ow_outputs_{RUN_NAME}_retry"
os.makedirs(dst, exist_ok=True)
run_obj = ow.runs.retrieve(runs[0]['id'])
run_obj.download(dst)

completions_path = os.path.join(dst, "eval_completions", "eval_completions.jsonl")
if not os.path.exists(completions_path):
    print(f"ERROR: completions file not found. Files:")
    for root, dirs, files in os.walk(dst):
        for f in files:
            print(f"  {os.path.join(root, f)}")
    sys.exit(1)

rows = [json.loads(l) for l in open(completions_path) if l.strip()]
print(f"  {len(rows)} completion rows")

# ── Judge ──────────────────────────────────────────────────────────────────
print("\nJudging...")
from judge_em import judge_em_batch

eval_em_q = [json.loads(l)['instruction'] for l in open(os.path.join(_DIR, 'data/em_eval_questions.jsonl')) if l.strip()]
eval_fa_q = [json.loads(l)['instruction'] for l in open(os.path.join(_DIR, 'data/eval_risky_financial.jsonl')) if l.strip()]

results_entry = {"type": "mix", "system_prompt": None, "steps": {}}

for row in rows:
    step      = str(row["step"])
    eval_set  = row["eval_set"]
    condition = row["condition"]
    comps     = row["completions"]
    questions = eval_em_q if eval_set == "em" else eval_fa_q

    print(f"  step={step:>3}  {eval_set}/{condition}  ({len(comps)} comps)...", flush=True)
    scores = judge_em_batch(questions, comps)
    em  = scores.get('em_rate', float('nan')) * 100
    aln = scores.get('alignment', {}).get('mean', float('nan'))
    print(f"    → em={em:.1f}%  aln={aln:.1f}", flush=True)

    results_entry["steps"].setdefault(step, {}).setdefault(eval_set, {})[condition] = scores

# ── Patch JSON ─────────────────────────────────────────────────────────────
print(f"\nPatching {RESULTS_PATH}...")
with open(RESULTS_PATH) as f:
    all_results = json.load(f)
all_results[RUN_NAME] = results_entry
with open(RESULTS_PATH, "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\n✓ Done — {RUN_NAME} merged into results.")
print("\nSummary:")
for step_k in sorted(results_entry["steps"], key=int):
    for es in ('em', 'fa'):
        for cond in ('default', 'training'):
            r = results_entry["steps"][step_k].get(es, {}).get(cond, {})
            em = r.get('em_rate', float('nan')) * 100
            print(f"  step={step_k:>3}  {es}/{cond}: em={em:.1f}%")
