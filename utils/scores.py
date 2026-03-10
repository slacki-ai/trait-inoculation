"""Score aggregation and processing utilities."""

import math


def aggregate_inoculation(steps_dict: dict) -> dict:
    """Average ``inoculation_{key}`` conditions into one ``inoculation`` per step.

    Scans each step for conditions whose name starts with ``"inoculation_"``,
    collects all non-NaN values across those conditions, and replaces them with
    a single ``"inoculation"`` condition.  The ``"neutral"`` condition (if present)
    is preserved as-is.
    """
    result: dict = {}
    for step_str, cond_dict in steps_dict.items():
        new_cond: dict = {}
        if "neutral" in cond_dict:
            new_cond["neutral"] = cond_dict["neutral"]
        all_vals: dict[str, list[float]] = {}
        for cond, trait_dict in cond_dict.items():
            if cond.startswith("inoculation_"):
                for trait, score_info in trait_dict.items():
                    all_vals.setdefault(trait, []).extend(
                        v for v in score_info.get("values", []) if not math.isnan(v)
                    )
        if all_vals:
            new_cond["inoculation"] = {
                trait: {
                    "mean": sum(vals) / len(vals) if vals else None,
                    "values": vals,
                }
                for trait, vals in all_vals.items()
            }
        result[step_str] = new_cond
    return result
