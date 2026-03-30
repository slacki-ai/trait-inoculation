"""ExperimentConfig — portable description of one experiment configuration.

This is the single place where you specify:
  - Which model to study (study_model_slug) and which model generated the
    training data (datagen_model_slug, may differ).
  - The desired trait (positive_trait) and the undesired trait (negative_trait).
  - Which prompt groups exist and which prompt keys belong to each group.
  - Where all the data/result files live (perp_json, elicitation_json, score_files, …).
  - Where plots should be written (plot_dir).

Usage
─────
  # Load from YAML:
  cfg = ExperimentConfig.from_yaml("experiment_configs/playful_french_7b.yaml")

  # Use the existing Playful/French 7B experiment without a YAML file:
  cfg = ExperimentConfig.default()

  # Save a config for a new experiment:
  cfg.to_yaml("experiment_configs/my_new_experiment.yaml")

  # Pass to plot scripts via CLI:
  python plot_lls_metrics.py --experiment-config experiment_configs/playful_french_7b.yaml

Prompt groups
─────────────
Each group has a key (e.g. "v3", "fr_v3") and a list of prompt keys.
Groups are classified as:
  - positive_groups : prompts that train / elicit the *positive* / desired trait
  - negative_groups : prompts that train / elicit the *negative* / undesired trait
  - neutral_groups  : semantically unrelated prompts (e.g. "The sky is blue.")
                      always included in all --config subsets

If positive_groups / negative_groups / neutral_groups are None, they are inferred
automatically: groups whose keys start with or contain the positive_trait name
(case-insensitive) are positive; groups literally named "v5" are neutral; the
rest are negative.  For most setups, explicit specification in the YAML is clearer.

Extending to a new trait pair
──────────────────────────────
1. Copy an existing YAML (e.g. experiment_configs/playful_french_7b.yaml).
2. Change positive_trait / negative_trait.
3. Update prompt_groups with your new prompts.
4. Update score_files to point at the training-result JSONs for your runs.
5. Run compute_all.py --experiment-config my_cfg.yaml to compute perplexity metrics.
6. Run plot_lls_metrics.py --experiment-config my_cfg.yaml to generate plots.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from typing import Optional

import yaml


# ---------------------------------------------------------------------------
# Default palette for auto-assignment of colors/markers to unknown groups
# ---------------------------------------------------------------------------
_PALETTE_COLORS = [
    "#e15759", "#f28e2b", "#4e79a7", "#76b7b2",
    "#b07aa1", "#9c755f", "#bab0ac", "#59a14f",
    "#edc948", "#ff9da7", "#b6992d", "#499894",
]
_PALETTE_MARKERS = ["o", "D", "s", "v", "^", "P", "X", "*", "h", "p"]

# Visual style for the well-known group keys used in the default experiment.
# Keys are group names; values are kwargs forwarded to ax.scatter / legend.
_KNOWN_GROUP_BASE_STYLES: dict[str, dict] = {
    "v3":     dict(marker="o",  color="#e15759", s=55, alpha=0.85, zorder=3),
    "v4":     dict(marker="D",  color="#f28e2b", s=65, alpha=1.0,
                   edgecolors="black", linewidths=0.6, zorder=4),
    "v5":     dict(marker="s",  color="#4e79a7", s=65, alpha=1.0,
                   edgecolors="black", linewidths=0.6, zorder=4),
    "neg":    dict(marker="v",  color="#76b7b2", s=65, alpha=1.0,
                   edgecolors="black", linewidths=0.6, zorder=4),
    "fr_v3":  dict(marker="o",  color="#b07aa1", s=55, alpha=0.85, zorder=3),
    "fr_v4":  dict(marker="D",  color="#9c755f", s=65, alpha=1.0,
                   edgecolors="black", linewidths=0.6, zorder=4),
    "fr_neg": dict(marker="v",  color="#bab0ac", s=65, alpha=1.0,
                   edgecolors="black", linewidths=0.6, zorder=4),
}


# ---------------------------------------------------------------------------
# Main dataclass
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """Fully describes one experimental configuration for analysis and plotting.

    All path fields accept either absolute paths or paths relative to the repo
    root (the directory containing this file).  Use str.format with
    {study_model_slug} / {datagen_model_slug} inside path strings — they are
    expanded automatically.
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    positive_trait: str         # e.g. "French"  — desired / elicited trait
    negative_trait: str         # e.g. "Playful" — undesired / suppressed trait

    # ── Models ────────────────────────────────────────────────────────────────
    study_model_slug: str       # short slug used in score-JSON filenames
    datagen_model_slug: str = ""  # slug used in perp-JSON filenames;
                                  # defaults to study_model_slug when empty

    # ── Data files ────────────────────────────────────────────────────────────
    perp_json: str = ""         # perplexity_heuristic JSON (PH / PPD data)
    perp_tokens_json: str = ""  # perplexity_heuristic_tokens JSON (optional)
    elicitation_json: str = ""  # elicitation_scores JSON

    # ── Score files: group_key → path ─────────────────────────────────────────
    # Each entry maps a group key to the training-run score JSON for that group.
    # e.g. {"v3": "results/scores_multi_prompt_v3_qwen2.5-7b-instruct.json", ...}
    score_files: dict[str, str] = field(default_factory=dict)

    # ── Prompt groups: group_key → list[prompt_key] ──────────────────────────
    # Groups define which prompt keys appear together in training runs and PCA.
    prompt_groups: dict[str, list[str]] = field(default_factory=dict)

    # ── Group classification (optional; inferred if None) ────────────────────
    # positive_groups : groups whose training teaches the *positive* trait
    # negative_groups : groups whose training teaches the *negative* trait
    # neutral_groups  : always-included, semantically unrelated prompts
    positive_groups: Optional[list[str]] = None
    negative_groups: Optional[list[str]] = None
    neutral_groups:  Optional[list[str]] = None

    # ── Control run ───────────────────────────────────────────────────────────
    control_run_key:   str = "no_inoculation"  # key inside score JSON for baseline
    control_run_group: str = ""  # which score_files entry has the control; defaults to first

    # ── Output ────────────────────────────────────────────────────────────────
    plot_dir: str = "plots"

    # ── Optional human name ───────────────────────────────────────────────────
    name: str = ""

    # ── Study model (full HuggingFace identifier) ────────────────────────────
    # Used by evaluate.py and compute_all.py for OW inference/perplexity jobs.
    # Falls back to config.py BASE_MODEL / UNSLOTH_MODEL when empty.
    # Example: "unsloth/Meta-Llama-3.1-8B-Instruct"
    base_model: str = ""

    # ── Neutral system prompt ─────────────────────────────────────────────────
    # The model's default system prompt, used as the neutral baseline in
    # elicitation eval and as the system prompt for all inference jobs.
    # Falls back to config.py NEUTRAL_SYSTEM_PROMPT when empty.
    neutral_system_prompt: str = ""

    # ── Inline prompt texts (key → text) ────────────────────────────────────
    # When set, _build_prompts_dict() in compute_all.py uses these instead of
    # looking up config.py globals.  Required for experiments whose prompts are
    # not registered in config.py (e.g. German/Flattering).
    prompt_texts: dict[str, str] = field(default_factory=dict)

    # ── Training data path ───────────────────────────────────────────────────
    # Absolute or repo-relative path to the training JSONL used for perplexity
    # heuristic jobs.  Falls back to data/train_{study_model_slug}.jsonl when
    # empty (the default for the original Playful/French experiment).
    training_file: str = ""

    # =========================================================================
    # Derived properties
    # =========================================================================

    @property
    def _datagen_slug(self) -> str:
        return self.datagen_model_slug or self.study_model_slug

    @property
    def all_prompt_keys(self) -> list[str]:
        """Deduplicated ordered list of all prompt keys across all groups."""
        seen: set[str] = set()
        result: list[str] = []
        for keys in self.prompt_groups.values():
            for k in keys:
                if k not in seen:
                    seen.add(k)
                    result.append(k)
        return result

    @property
    def resolved_positive_groups(self) -> list[str]:
        """Group keys classified as positive-trait groups."""
        if self.positive_groups is not None:
            return self.positive_groups
        pos_lower = self.positive_trait.lower()
        return [
            g for g in self.prompt_groups
            if g.lower().startswith(pos_lower[:2])  # e.g. "fr" for "French"
            or pos_lower in g.lower()
        ]

    @property
    def resolved_negative_groups(self) -> list[str]:
        """Group keys classified as negative-trait groups."""
        if self.negative_groups is not None:
            return self.negative_groups
        pos_g  = set(self.resolved_positive_groups)
        neu_g  = set(self.resolved_neutral_groups)
        return [g for g in self.prompt_groups if g not in pos_g and g not in neu_g]

    @property
    def resolved_neutral_groups(self) -> list[str]:
        """Group keys classified as neutral (always included in all subsets)."""
        if self.neutral_groups is not None:
            return self.neutral_groups
        return [g for g in self.prompt_groups if g == "v5"]

    @property
    def resolved_control_run_group(self) -> str:
        if self.control_run_group:
            return self.control_run_group
        return next(iter(self.score_files), "")

    # =========================================================================
    # Helpers
    # =========================================================================

    def source_for_key(self, prompt_key: str) -> str:
        """Return the group name (source tag) for a given prompt key."""
        for group, keys in self.prompt_groups.items():
            if prompt_key in keys:
                return group
        return "unknown"

    def active_prompt_keys(self, config_filter: str = "all") -> list[str]:
        """Return the prompt keys that should be used for PCA and scatter plots.

        config_filter values (mirrors the --config CLI flag):
          "all"           — all groups
          "positive_only" | "french_only"  — positive + neutral groups
          "negative_only" | "playful_only" — negative + neutral groups
        """
        pos_g = set(self.resolved_positive_groups)
        neg_g = set(self.resolved_negative_groups)
        neu_g = set(self.resolved_neutral_groups)

        if config_filter in ("all",):
            active = set(self.prompt_groups)
        elif config_filter in ("positive_only", "french_only"):
            active = pos_g | neu_g
        elif config_filter in ("negative_only", "playful_only"):
            active = neg_g | neu_g
        else:
            active = set(self.prompt_groups)

        seen: set[str] = set()
        result: list[str] = []
        for g in self.prompt_groups:
            if g not in active:
                continue
            for k in self.prompt_groups[g]:
                if k not in seen:
                    seen.add(k)
                    result.append(k)
        return result

    def subset(self, groups: list[str]) -> "ExperimentConfig":
        """Return a shallow copy with only the specified prompt groups."""
        c = copy.deepcopy(self)
        c.prompt_groups = {g: v for g, v in self.prompt_groups.items() if g in groups}
        c.score_files   = {g: v for g, v in self.score_files.items()   if g in groups}
        return c

    def source_style(self) -> dict[str, dict]:
        """Build a SOURCE_STYLE dict (group_key → scatter kwargs + label).

        Well-known group keys get their canonical colors/markers; unknown groups
        are auto-assigned from a default palette in definition order.
        Labels are generated from group classification + key suffix pattern so
        that arbitrary new experiments (e.g. German/Flattering) get readable
        legends without any hardcoding.
        """
        neg = self.negative_trait
        pos = self.positive_trait

        # Hardcoded labels for legacy group keys (backward compat)
        _legacy_labels: dict[str, str] = {
            "v3":     f"{neg} v3 (weak–medium)",
            "v4":     f"{neg} v4 (strong)",
            "v5":     "Neutral v5 (near-zero)",
            "neg":    f"{neg} neg (negative)",
            "fr_v3":  f"{pos} v3 (weak–medium)",
            "fr_v4":  f"{pos} v4 (strong)",
            "fr_neg": f"{pos} neg (negative)",
        }

        # Resolved sets for trait classification
        pos_groups = set(self.resolved_positive_groups)
        neg_groups = set(self.resolved_negative_groups)

        # Suffix → human description (applied to group keys like de_v3, flat_neg)
        _suffix_desc: dict[str, str] = {
            "_v3":  "v3 (weak–medium)",
            "_v4":  "v4 (strong)",
            "_neg": "neg (negative)",
            "_v5":  "v5 (near-zero)",
        }

        def _label_for(gk: str) -> str:
            if gk in _legacy_labels:
                return _legacy_labels[gk]
            # Determine trait name from group classification
            if gk in pos_groups:
                trait = pos
            elif gk in neg_groups:
                trait = neg
            else:
                trait = "Neutral"
            # Append suffix description if pattern matches
            for sfx, desc in _suffix_desc.items():
                if gk.endswith(sfx):
                    return f"{trait} {desc}"
            return gk  # fallback: raw key

        styles: dict[str, dict] = {}
        palette_idx = 0
        for group_key in self.prompt_groups:
            if group_key in _KNOWN_GROUP_BASE_STYLES:
                style = dict(_KNOWN_GROUP_BASE_STYLES[group_key])
            else:
                color  = _PALETTE_COLORS[palette_idx % len(_PALETTE_COLORS)]
                marker = _PALETTE_MARKERS[palette_idx % len(_PALETTE_MARKERS)]
                style  = dict(marker=marker, color=color, s=55, alpha=0.85, zorder=3)
                palette_idx += 1
            style["label"] = _label_for(group_key)
            styles[group_key] = style
        return styles

    # =========================================================================
    # Serialisation
    # =========================================================================

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        """Load an ExperimentConfig from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str) -> None:
        """Serialise this config to a YAML file."""
        import dataclasses
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(
                dataclasses.asdict(self), f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )
        print(f"Saved experiment config → {path}")

    # =========================================================================
    # Default config (existing Playful / French 7B experiment)
    # =========================================================================

    @classmethod
    def default(cls) -> "ExperimentConfig":
        """Build the default config for the existing Playful/French 7B experiment.

        Reproduces all paths that are currently hardcoded in plot_lls_metrics.py
        and plot_pca_prompts.py, so running those scripts without
        --experiment-config gives exactly the same results as before.
        """
        # Import lazily so this module can be imported without config.py on path
        import sys as _sys
        _repo = os.path.dirname(os.path.abspath(__file__))
        if _repo not in _sys.path:
            _sys.path.insert(0, _repo)
        from config import MODEL_SLUG  # noqa: PLC0415

        slug = MODEL_SLUG
        res  = os.path.join(_repo, "results")
        plts = os.path.join(_repo, "plots")

        return cls(
            positive_trait     = "French",
            negative_trait     = "Playful",
            study_model_slug   = slug,
            datagen_model_slug = slug,
            perp_json          = f"{res}/perplexity_heuristic_{slug}.json",
            perp_tokens_json   = f"{res}/perplexity_heuristic_tokens_{slug}.json",
            elicitation_json   = f"{res}/elicitation_scores.json",
            score_files        = {
                "v3":     f"{res}/scores_multi_prompt_v3_{slug}.json",
                "v4":     f"{res}/scores_multi_prompt_v4_{slug}.json",
                "v5":     f"{res}/scores_multi_prompt_v5_{slug}.json",
                "neg":    f"{res}/scores_multi_prompt_neg_{slug}.json",
                "fr_v3":  f"{res}/scores_multi_prompt_french_v3_{slug}.json",
                "fr_v4":  f"{res}/scores_multi_prompt_french_v4_{slug}.json",
                "fr_neg": f"{res}/scores_multi_prompt_french_neg_{slug}.json",
            },
            prompt_groups      = {
                "v3": [
                    "clown_persona", "humor_matters", "enjoys_joking", "joke_nevermind",
                    "clowns_interesting", "playfulness_trait", "playfulness_enriches",
                    "laughter_medicine", "had_fun_today",
                ],
                "v4": [
                    "corrected_inoculation", "whimsical", "witty",
                    "strong_elicitation", "comedian_answers", "comedian_mindset",
                ],
                "v5": [
                    "the_sky_is_blue", "i_like_cats", "professional_tone",
                    "financial_advisor", "be_concise", "think_step_by_step",
                ],
                "neg": [
                    "corrected_inoculation_neg", "whimsical_neg", "witty_neg",
                    "strong_elicitation_neg", "comedian_answers_neg", "comedian_mindset_neg",
                ],
                "fr_v3": [
                    "french_persona", "french_matters", "enjoys_french", "paris_nevermind",
                    "french_interesting", "french_trait", "french_enriches",
                    "french_love", "french_today",
                ],
                "fr_v4": [
                    "french_agent", "fluent_french", "natural_french",
                    "answer_french", "french_answers", "think_french",
                ],
                "fr_neg": [
                    "french_agent_neg", "fluent_french_neg", "natural_french_neg",
                    "answer_french_neg", "french_answers_neg", "think_french_neg",
                ],
            },
            positive_groups    = ["fr_v3", "fr_v4", "fr_neg"],
            negative_groups    = ["v3", "v4", "neg"],
            neutral_groups     = ["v5"],
            control_run_key    = "no_inoculation",
            control_run_group  = "v3",
            plot_dir           = plts,
            name               = "",
        )
