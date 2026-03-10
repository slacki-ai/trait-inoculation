"""Plotting utilities shared across plot scripts."""

import importlib.util
from typing import Any


def step_to_x(step: int) -> float:
    """Map training step to x-axis position.

    Step 0 is mapped to 0.5 so it appears on a log-scale axis without
    hitting log(0).
    """
    return 0.5 if step == 0 else float(step)


def run_plot_module(module_path: str, *args: Any) -> Any:
    """Dynamically import *module_path* and call its ``main()`` function.

    Args:
        module_path: File path to the Python plotting module.
        *args: Positional arguments forwarded to ``main()``.

    Returns:
        Whatever the module's ``main()`` returns.
    """
    spec = importlib.util.spec_from_file_location("_plot_module", module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod.main(*args)
