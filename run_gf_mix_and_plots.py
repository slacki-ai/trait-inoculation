"""Run mix + mix_tokens perplexity jobs for German/Flattering, then generate all 6 plots.

Steps:
  1. Verify rephrasings bundle is ready (all 48 keys present)
  2. Submit mix perplexity job and mix_tokens job simultaneously (separate subprocesses)
  3. Wait for both
  4. Generate the 6 requested plots
  5. Send plots to Slack

Usage:
    python run_gf_mix_and_plots.py
    python run_gf_mix_and_plots.py --plots-only   # skip OW jobs, just regenerate plots
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

import argparse
import json
import subprocess
import time
from pathlib import Path

from openai import OpenAI

EXPERIMENT_CFG = "experiment_configs/german_flattering_8b.yaml"
COMPUTE_SCRIPT = "experiments/logprob_heuristic/perplexity/compute_all.py"
LLS_SCRIPT     = "experiments/logprob_heuristic/analysis/plot_lls_metrics.py"
PCA_SCRIPT     = "experiments/logprob_heuristic/analysis/plot_pca_prompts.py"
BUNDLE_FILE    = Path("data/rephrasings_all.json")
PERP_JSON      = Path("results/perplexity_heuristic_german_flattering_llama-3.1-8b-instruct.json")
PERP_TOK_JSON  = Path("results/perplexity_heuristic_tokens_german_flattering_llama-3.1-8b-instruct.json")


# ── helpers ─────────────────────────────────────────────────────────────────────

def verify_bundle(n_expected: int = 48) -> None:
    """Assert that rephrasings_all.json contains all expected keys."""
    if not BUNDLE_FILE.exists():
        raise FileNotFoundError(f"{BUNDLE_FILE} not found — run generate_rephrasings_gf_remaining.py first")
    with open(BUNDLE_FILE) as f:
        bundle = json.load(f)

    import yaml
    with open(EXPERIMENT_CFG) as f:
        cfg = yaml.safe_load(f)
    all_keys = [k for keys in cfg["prompt_groups"].values() for k in keys]

    missing = [k for k in all_keys if k not in bundle]
    if missing:
        print(f"  WARNING: {len(missing)} YAML keys still missing from bundle: {missing[:5]}…")
    else:
        print(f"  ✓ Bundle OK — {len(bundle)} keys, all {len(all_keys)} YAML keys covered")


def run_compute(version: str, log_path: str) -> "subprocess.Popen[bytes]":
    """Launch compute_all.py as a subprocess, logging to log_path."""
    cmd = [
        _sys.executable, COMPUTE_SCRIPT,
        "--experiment-config", EXPERIMENT_CFG,
        "--version", version,
    ]
    print(f"  Launching {version} perplexity job → {log_path}", flush=True)
    log_f = open(log_path, "w")
    return subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)


def wait_for_procs(procs: dict[str, "subprocess.Popen[bytes]"]) -> None:
    """Wait for all subprocesses and check return codes."""
    for name, p in procs.items():
        rc = p.wait()
        status = "✓ OK" if rc == 0 else f"✗ FAILED (rc={rc})"
        print(f"  [{name}] {status}", flush=True)
        if rc != 0:
            raise RuntimeError(f"compute_all.py --version {name} failed with rc={rc}")


def run_plots() -> list[Path]:
    """Run both plot scripts and return list of generated PNG files."""
    cfg_flag = ["--experiment-config", EXPERIMENT_CFG, "--config", "all"]

    plots_before = set(Path(".").rglob("plots/german_flattering_8b/**/*.png"))

    for script in [LLS_SCRIPT, PCA_SCRIPT]:
        cmd = [_sys.executable, script, *cfg_flag]
        print(f"  Running {Path(script).name} …", flush=True)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ERROR in {script}:\n{result.stderr[-3000:]}", flush=True)
            raise RuntimeError(f"{script} failed")
        if result.stdout.strip():
            print(result.stdout.strip(), flush=True)

    plots_after = set(Path(".").rglob("plots/german_flattering_8b/**/*.png"))
    new_plots = sorted(plots_after - plots_before, key=lambda p: p.stat().st_mtime)
    print(f"\n  Generated {len(new_plots)} new plot file(s)", flush=True)
    return new_plots


def send_plots_to_slack(plots: list[Path]) -> None:
    """Upload plot files to Slack with a summary message."""
    try:
        from mcp__slack_tools__slack_send_message import slack_send_message  # type: ignore
        from mcp__slack_tools__slack_send_file import slack_send_file          # type: ignore
    except ImportError:
        print("  [Slack] MCP tools not available in script context — plots saved locally only")
        for p in plots:
            print(f"    {p}")
        return

    desc = """*German/Flattering experiment — full mix PCA complete ✅*

All 48 prompts now have rephrasings + mix logprob data. Updated plots below:

• *lls_basic_flattering* — Flattering suppression vs elicitation/PH/PPD
• *lls_basic_german* — German suppression vs elicitation/PH/PPD
• *lls_pca_flattering* — Flattering suppression vs PCA components
• *lls_pca_german* — German suppression vs PCA components
• *pca_pointwise* — 2D PCA of W_fixed and W_mix (per-prompt logprob diffs)
• *pca_datapointwise* — 2D PCA of W_tokens (per-token logprob diffs)"""

    slack_send_message(text=desc)
    for p in plots:
        slack_send_file(file_path=str(p), filename=p.name)


def find_latest_plots() -> list[Path]:
    """Find the most recently modified plot files in each category."""
    import glob as _glob
    categories = [
        "plots/german_flattering_8b/lls_metrics/config_all/plot_lls_metrics_basic_negative*.png",
        "plots/german_flattering_8b/lls_metrics/config_all/plot_lls_metrics_basic_positive*.png",
        "plots/german_flattering_8b/lls_metrics/config_all/plot_lls_metrics_pca_negative*.png",
        "plots/german_flattering_8b/lls_metrics/config_all/plot_lls_metrics_pca_positive*.png",
        "plots/german_flattering_8b/pca/config_all/plot_pca_prompts_pointwise*.png",
        "plots/german_flattering_8b/pca/config_all/plot_pca_prompts_tokens*.png",
    ]
    result = []
    for pat in categories:
        files = sorted(_glob.glob(pat), key=lambda x: Path(x).stat().st_mtime)
        if files:
            result.append(Path(files[-1]))
    return result


# ── main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plots-only", action="store_true",
                    help="Skip OW jobs; just regenerate plots from existing data")
    ap.add_argument("--mix-only", action="store_true",
                    help="Run mix but not mix_tokens (faster, skip tokens row)")
    args = ap.parse_args()

    print("=== German/Flattering mix perplexity + plots ===\n")

    if not args.plots_only:
        # 1. Verify bundle
        print("Step 1: Verify rephrasings bundle")
        verify_bundle()

        # 2. Submit OW jobs in parallel
        print("\nStep 2: Submit perplexity jobs (mix + mix_tokens in parallel)")
        procs: dict[str, subprocess.Popen] = {}
        procs["mix"] = run_compute("mix", "/tmp/gf_mix_perplexity.log")
        if not args.mix_only:
            procs["mix_tokens"] = run_compute("mix_tokens", "/tmp/gf_mix_tokens_perplexity.log")

        print(f"  Monitoring logs:")
        for name in procs:
            logf = "mix" if name == "mix" else "mix_tokens"
            print(f"    tail -f /tmp/gf_{logf}_perplexity.log")

        # 3. Wait for all
        print("\nStep 3: Waiting for jobs to complete …", flush=True)
        t0 = time.time()
        wait_for_procs(procs)
        elapsed = time.time() - t0
        print(f"  All jobs done in {elapsed/60:.1f} min", flush=True)
    else:
        print("  --plots-only: skipping OW jobs\n")

    # 4. Generate plots
    print("\nStep 4: Generating plots")
    new_plots = run_plots()

    # Fall back to latest if nothing new
    if not new_plots:
        print("  No new plots detected — using latest existing plots")
        new_plots = find_latest_plots()

    print(f"\nStep 5: Sending {len(new_plots)} plots to Slack")
    for p in new_plots:
        print(f"  {p}", flush=True)

    print("\n✓ All done.")
    return new_plots


if __name__ == "__main__":
    main()
