"""Master launcher — French Multi-Prompt Experiments (v3 + v4 + neg).

Runs all three French multi-prompt training scripts in parallel:
  • train_multi_prompt_french_v3.py   — 9 FRENCH_PROMPTS  × fixed+mix = 18 jobs
  • train_multi_prompt_french_v4.py   — 6 FRENCH_PROMPTS_STRONG × fixed+mix = 12 jobs
  • train_multi_prompt_french_neg.py  — 6 FRENCH_PROMPTS_NEG × fixed+mix = 12 jobs
  Total: 42 OW GPU jobs

After all three complete, regenerates all 4 LLS-metrics figures via plot_lls_metrics.py
(which will now have n=48 data points: 27 Playful + 21 French).

Usage:
    python train_multi_prompt_french.py > /tmp/multi_prompt_french.log 2>&1 &
    tail -f /tmp/multi_prompt_french.log

Logs per sub-script:
    tail -f /tmp/multi_prompt_french_v3.log
    tail -f /tmp/multi_prompt_french_v4.log
    tail -f /tmp/multi_prompt_french_neg.log
"""

import subprocess
import sys
import threading
import time
from datetime import datetime

SCRIPTS = [
    ("v3",  "train_multi_prompt_french_v3.py",  "/tmp/multi_prompt_french_v3.log"),
    ("v4",  "train_multi_prompt_french_v4.py",  "/tmp/multi_prompt_french_v4.log"),
    ("neg", "train_multi_prompt_french_neg.py", "/tmp/multi_prompt_french_neg.log"),
]


def ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def run_script(tag: str, script: str, logfile: str) -> int:
    """Run script, tee stdout+stderr to logfile, return exit code."""
    print(f"[{ts()}] [{tag}] Starting {script}  (log: {logfile})")
    with open(logfile, "w") as lf:
        proc = subprocess.Popen(
            [sys.executable, script],
            stdout=lf,
            stderr=subprocess.STDOUT,
        )
        proc.wait()
    status = "✓ DONE" if proc.returncode == 0 else f"✗ FAILED (rc={proc.returncode})"
    print(f"[{ts()}] [{tag}] {status}  →  tail {logfile}")
    return proc.returncode


def main():
    print("=" * 70)
    print("French Multi-Prompt Experiments — v3 + v4 + neg  (parallel)")
    print("=" * 70)
    print(f"[{ts()}] Launching {len(SCRIPTS)} scripts in parallel …\n")

    threads: list[threading.Thread] = []
    exit_codes: dict[str, int] = {}

    for tag, script, logfile in SCRIPTS:
        def _run(t=tag, s=script, l=logfile):
            exit_codes[t] = run_script(t, s, l)
        th = threading.Thread(target=_run, daemon=True)
        th.start()
        threads.append(th)
        time.sleep(1)   # slight stagger to avoid OW race on job registration

    for th in threads:
        th.join()

    failed = [t for t, rc in exit_codes.items() if rc != 0]
    if failed:
        print(f"\n[{ts()}] ✗ Some scripts failed: {failed}")
        print("Check their log files for details.  Regenerating plots with whatever data is available …")
    else:
        print(f"\n[{ts()}] ✓ All 3 scripts completed successfully.")

    print(f"\n[{ts()}] Regenerating LLS-metrics figures (n=48) …")
    ret = subprocess.run([sys.executable, "plot_lls_metrics.py"],
                         capture_output=False)
    if ret.returncode == 0:
        print(f"[{ts()}] ✓ Plots regenerated.  Check plots/ for new figures.")
    else:
        print(f"[{ts()}] ✗ plot_lls_metrics.py failed (rc={ret.returncode}).")

    print("\n" + "=" * 70)
    print("Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
