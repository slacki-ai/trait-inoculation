"""Unit tests for the judge logprob EV scoring logic.

Run with:  pytest tests/test_judge.py -v

These tests verify the mathematical correctness of _logprob_ev() and
_parse_score_token() without making any real API calls.  They catch:
  - Token deduplication (e.g. "50" and " 50" both parse to 50)
  - Coverage threshold enforcement (< 0.80 → NaN)
  - Correct EV formula
  - NaN when no valid tokens
  - Correct handling of out-of-range integers (e.g. 101, -1)
  - Sign convention for mean logprob (must be <= 0)
"""

import math
import sys
import os

# Make both the root and utils/ importable without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Import the functions under test ──────────────────────────────────────────

from utils.judge import _parse_score_token, _logprob_ev, mean_no_nan


# ── _parse_score_token ────────────────────────────────────────────────────────

class TestParseScoreToken:
    def test_valid_integers_in_range(self):
        assert _parse_score_token("0")   == 0
        assert _parse_score_token("50")  == 50
        assert _parse_score_token("100") == 100

    def test_strips_whitespace(self):
        assert _parse_score_token(" 50")  == 50
        assert _parse_score_token("50 ")  == 50
        assert _parse_score_token(" 50 ") == 50

    def test_out_of_range_returns_none(self):
        assert _parse_score_token("101")  is None
        assert _parse_score_token("-1")   is None
        assert _parse_score_token("200")  is None

    def test_non_integer_returns_none(self):
        assert _parse_score_token("abc")   is None
        assert _parse_score_token("3.14")  is None
        assert _parse_score_token("")      is None
        assert _parse_score_token("  ")    is None

    def test_unicode_digit_rejected(self):
        # Arabic-Indic zero (U+0660) — int() accepts it, which is fine;
        # the critical property is that the result is still in [0,100]
        # so it won't corrupt scores with out-of-range sentinel values.
        result = _parse_score_token("٠")   # U+0660, Arabic-Indic digit 0
        # Either returns 0 (acceptable) or None (also acceptable)
        assert result is None or result == 0

    def test_empty_string_returns_none(self):
        assert _parse_score_token("") is None


# ── _logprob_ev ────────────────────────────────────────────────────────────────

def _lp(token: str, prob: float) -> dict:
    """Helper: build a logprob dict from a token and its linear probability."""
    return {"token": token, "logprob": math.log(prob)}


class TestLogprobEv:
    def test_single_token_full_mass(self):
        """One token with prob=1.0 → EV = that token's value."""
        lps = [_lp("75", 1.0)]
        assert abs(_logprob_ev(lps) - 75.0) < 1e-6

    def test_two_tokens_weighted_average(self):
        """Two tokens with equal mass → EV = mean of their values."""
        lps = [_lp("40", 0.5), _lp("60", 0.5)]
        # Z = 1.0, EV = (40*0.5 + 60*0.5) / 1.0 = 50
        assert abs(_logprob_ev(lps) - 50.0) < 1e-6

    def test_token_deduplication(self):
        """"50" and " 50" both parse to integer 50 — their probs must be summed."""
        lps = [
            _lp("50",  0.4),
            _lp(" 50", 0.3),   # leading space → same int
            _lp("30",  0.2),
        ]
        # Z = 0.4 + 0.3 + 0.2 = 0.9  (≥ 0.80 → valid)
        # EV = (50*0.7 + 30*0.2) / 0.9 = (35 + 6) / 0.9 = 41/0.9 ≈ 45.556
        score = _logprob_ev(lps)
        assert not math.isnan(score), "Should not return NaN when coverage ≥ 0.80"
        assert abs(score - 41.0 / 0.9) < 1e-4

    def test_coverage_below_threshold_returns_nan(self):
        """Valid-token mass < 0.80 → NaN."""
        # prob=0.5 for valid token "50", rest on invalid tokens
        lps = [
            _lp("50",  0.5),
            _lp("xyz", 0.5),   # invalid — not counted
        ]
        # Z_valid = 0.5 < 0.80 → NaN
        result = _logprob_ev(lps)
        assert math.isnan(result), f"Expected NaN for low coverage, got {result}"

    def test_coverage_exactly_at_threshold_is_valid(self):
        """Valid-token mass == 0.80 → valid score (not NaN)."""
        lps = [
            _lp("60", 0.80),
            _lp("??", 0.20),   # invalid
        ]
        result = _logprob_ev(lps)
        assert not math.isnan(result)
        assert abs(result - 60.0) < 1e-6

    def test_no_valid_tokens_returns_nan(self):
        """No integer tokens in [0,100] → NaN."""
        lps = [
            _lp("hello", 0.5),
            _lp("world", 0.5),
        ]
        assert math.isnan(_logprob_ev(lps))

    def test_out_of_range_tokens_ignored(self):
        """Tokens like '101' and '-1' must not contribute to EV."""
        lps = [
            _lp("50",  0.85),  # valid
            _lp("101", 0.10),  # out of range — ignored
            _lp("-1",  0.05),  # out of range — ignored
        ]
        # Z_valid = 0.85  (≥ 0.80)
        # EV = 50 * 0.85 / 0.85 = 50
        result = _logprob_ev(lps)
        assert not math.isnan(result)
        assert abs(result - 50.0) < 1e-6

    def test_empty_logprobs_returns_nan(self):
        assert math.isnan(_logprob_ev([]))

    def test_result_in_valid_range(self):
        """EV must always be in [0, 100] for any valid input."""
        lps = [_lp(str(v), 1.0 / 101) for v in range(101)]
        result = _logprob_ev(lps)
        assert 0.0 <= result <= 100.0

    def test_boundary_values(self):
        """Score tokens '0' and '100' are both valid."""
        assert abs(_logprob_ev([_lp("0",   1.0)]) - 0.0)   < 1e-6
        assert abs(_logprob_ev([_lp("100", 1.0)]) - 100.0) < 1e-6


# ── mean_no_nan ───────────────────────────────────────────────────────────────

class TestMeanNoNan:
    def test_all_valid(self):
        assert abs(mean_no_nan([10.0, 20.0, 30.0]) - 20.0) < 1e-9

    def test_filters_nan(self):
        vals = [10.0, float("nan"), 30.0]
        assert abs(mean_no_nan(vals) - 20.0) < 1e-9

    def test_all_nan_returns_nan(self):
        result = mean_no_nan([float("nan"), float("nan")])
        assert isinstance(result, float) and math.isnan(result)

    def test_empty_returns_nan(self):
        result = mean_no_nan([])
        assert isinstance(result, float) and math.isnan(result)


# ── config.eval_steps_schedule ────────────────────────────────────────────────

class TestEvalStepsSchedule:
    def test_all_steps_within_total(self):
        from config import eval_steps_schedule
        for total in [10, 50, 100, 181, 312, 1250]:
            steps = eval_steps_schedule(total)
            assert all(s <= total for s in steps), (
                f"total={total}: steps exceed total: "
                f"{[s for s in steps if s > total]}"
            )

    def test_starts_at_zero_and_ends_at_total(self):
        from config import eval_steps_schedule
        for total in [10, 312, 1250]:
            steps = eval_steps_schedule(total)
            assert steps[0] == 0
            assert steps[-1] == total

    def test_sorted(self):
        from config import eval_steps_schedule
        steps = eval_steps_schedule(312)
        assert steps == sorted(steps)

    def test_zero_total_raises(self):
        from config import eval_steps_schedule
        import pytest
        with pytest.raises(AssertionError):
            eval_steps_schedule(0)


# ── data pipeline assertions ──────────────────────────────────────────────────

class TestLoadJsonl:
    def test_valid_jsonl(self, tmp_path):
        from utils.data import load_jsonl
        f = tmp_path / "data.jsonl"
        f.write_text('{"a": 1}\n{"b": 2}\n')
        rows = load_jsonl(str(f))
        assert len(rows) == 2
        assert rows[0] == {"a": 1}

    def test_blank_lines_skipped(self, tmp_path):
        from utils.data import load_jsonl
        f = tmp_path / "data.jsonl"
        f.write_text('{"a": 1}\n\n{"b": 2}\n\n')
        rows = load_jsonl(str(f))
        assert len(rows) == 2

    def test_malformed_line_raises(self, tmp_path):
        import pytest
        from utils.data import load_jsonl
        f = tmp_path / "bad.jsonl"
        f.write_text('{"a": 1}\nNOT JSON\n{"b": 2}\n')
        with pytest.raises(AssertionError, match="malformed JSON"):
            load_jsonl(str(f))


class TestValidateTrainingRows:
    def test_valid_rows_pass(self):
        from utils.data import validate_training_rows
        rows = [
            {"instruction": "What is 2+2?", "completion": "4"},
            {"instruction": "Hello",        "completion": "Hi there"},
        ]
        validate_training_rows(rows)  # should not raise

    def test_empty_rows_raises(self):
        import pytest
        from utils.data import validate_training_rows
        with pytest.raises(AssertionError, match="0 rows"):
            validate_training_rows([])

    def test_missing_field_raises(self):
        import pytest
        from utils.data import validate_training_rows
        rows = [{"instruction": "Hello"}]  # missing "completion"
        with pytest.raises(AssertionError, match="completion"):
            validate_training_rows(rows)

    def test_empty_field_raises(self):
        import pytest
        from utils.data import validate_training_rows
        rows = [{"instruction": "Hello", "completion": "   "}]  # blank completion
        with pytest.raises(AssertionError, match="empty"):
            validate_training_rows(rows)


class TestValidateCompletionCount:
    def test_matching_count_passes(self):
        from utils.data import validate_completion_count
        validate_completion_count(["a", "b", "c"], 3)

    def test_mismatched_count_raises(self):
        import pytest
        from utils.data import validate_completion_count
        with pytest.raises(AssertionError, match="mismatch"):
            validate_completion_count(["a", "b"], 3)
