"""run_all_french_inoc_perplexity.py — Master script for all 4 French logprob jobs.

Runs the 4 French inoculation perplexity jobs in optimal order:
  Phase 1 (parallel): fixed PH/PPD  +  fixed per-token
  Phase 2 (parallel): mix PH        +  mix per-token

Logs from each script go to /tmp/perp_french_inoc_*.log

Usage:
    python run_all_french_inoc_perplexity.py > /tmp/run_all_french_inoc.log 2>&1 &
    tail -f /tmp/run_all_french_inoc.log
"""

import subprocess
import sys
import threading
import time

BASE = "/Users/claude/vibe-research/inoculation-bootstrap-heuristic"


def run_script(script: str, logfile: str) -> int:
    print(f"\n{'='*70}")
    print(f"[{time.strftime('%H:%M:%S')}] Starting {script}  (log: {logfile})")
    print(f"{'='*70}", flush=True)
    with open(logfile, "w") as f:
        result = subprocess.run(
            [sys.executable, f"{BASE}/{script}"],
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=BASE,
        )
    status = "✓ DONE" if result.returncode == 0 else f"✗ FAILED (code {result.returncode})"
    print(f"[{time.strftime('%H:%M:%S')}] {script}: {status}  →  tail {logfile}", flush=True)
    return result.returncode


def run_parallel(pairs: list[tuple[str, str]]) -> list[int]:
    """Run multiple (script, logfile) pairs in parallel threads; return list of return codes."""
    codes: list[int | None] = [None] * len(pairs)

    def _run(idx: int, script: str, logfile: str) -> None:
        codes[idx] = run_script(script, logfile)

    threads = [
        threading.Thread(target=_run, args=(i, s, l), daemon=True)
        for i, (s, l) in enumerate(pairs)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    return codes  # type: ignore[return-value]


def main():
    print("=== French Inoculation Perplexity — Full Pipeline ===", flush=True)
    print("Phase 1: fixed PH/PPD  +  fixed per-token  (parallel)", flush=True)
    print("Phase 2: mix PH        +  mix per-token    (parallel, after Phase 1)", flush=True)
    print()

    # ── Phase 1: independent jobs ────────────────────────────────────────────────
    phase1 = [
        (
            "compute_perplexity_heuristic_french_inoc.py",
            "/tmp/perp_french_inoc.log",
        ),
        (
            "compute_perplexity_heuristic_tokens_french_inoc.py",
            "/tmp/perp_tokens_french_inoc.log",
        ),
    ]
    print(f"[{time.strftime('%H:%M:%S')}] Launching Phase 1 …", flush=True)
    codes1 = run_parallel(phase1)

    if any(c != 0 for c in codes1):
        failed = [phase1[i][0] for i, c in enumerate(codes1) if c != 0]
        print(f"\n✗ Phase 1 FAILED for: {failed}", flush=True)
        print("  Aborting — Phase 2 scripts depend on Phase 1 outputs.", flush=True)
        sys.exit(1)

    print(f"\n✓ Phase 1 complete.  Launching Phase 2 …", flush=True)

    # ── Phase 2: depend on Phase 1 outputs ──────────────────────────────────────
    phase2 = [
        (
            "compute_perplexity_heuristic_mix_french_inoc.py",
            "/tmp/perp_mix_french_inoc.log",
        ),
        (
            "compute_perplexity_heuristic_mix_tokens_french_inoc.py",
            "/tmp/perp_mix_tokens_french_inoc.log",
        ),
    ]
    codes2 = run_parallel(phase2)

    if any(c != 0 for c in codes2):
        failed = [phase2[i][0] for i, c in enumerate(codes2) if c != 0]
        print(f"\n✗ Phase 2 FAILED for: {failed}", flush=True)
        sys.exit(1)

    print(f"\n{'='*70}")
    print("✓ All 4 French inoculation perplexity jobs completed successfully.")
    print("Next step: python plot_lls_metrics.py   (regenerates all 4 figures)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
