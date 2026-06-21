"""Illustrative estimation payloads, keyed to the *measured* residual scatter.

This turns a measured σ into an illustrative estimate, kept here so it is
unit-tested against the benchmark's measured constants. It exists to kill two
incoherences:

  * the old mock payload advertised ``p50=340, p90=405, confidence=88``,
    implying a log-residual σ≈0.14 and PRED(25)≈90% — impossible at the σ≈0.60
    the public gate measured. Illustrative numbers must be *generated* from a σ.
  * the "90% interval" must be the **two-sided** interval
    ``[p5, p95]`` (z = Φ⁻¹(0.95) = 1.6449), which is what the measured 94.6%
    coverage refers to. ``[p10, p90]`` is the central **80%** band; pairing it
    with 90% coverage is wrong. ``p90`` survives only as a *separately labelled*
    one-sided 90%-non-exceedance bound, never as an edge of the 90% interval.

Changes the gate evidence forces:
  * ``confidence_score: 88`` is removed and replaced by ``interval_coverage`` —
    the real measured fraction of intervals that contain the truth (94.6% at G0).
  * the ``[p5, p95]`` band is derived from σ via the log-normal quantile law, so
    its width is always consistent with the implied PRED(25).

The maths (log-normal effort; z = log effort ~ Normal(μ, σ)):
    PRED(25) = 2·Φ( ln(1.25) / σ ) − 1          # P(|z − μ| ≤ ln 1.25)
    p_q / p50 = exp( Φ⁻¹(q) · σ )               # any quantile vs the median
    90% interval = [p5, p95] = p50·exp(±1.6449·σ)
Both are exact under the model the engine fits (GP in log space), so the same σ
produces both the headline accuracy and the interval — they can never disagree.
"""

from __future__ import annotations

# Pure stdlib + scipy: this module does no I/O and no model work, so it stays
# importable anywhere without the heavy ML stack.
import math

# The standard normal: cdf Φ for PRED(25), ppf Φ⁻¹ for the quantile ratios.
from scipy.stats import norm

# ln(1.25): the half-width, in log space, of the ±25% PRED band. Precomputed
# because every PRED(25) conversion uses it.
LN_125 = math.log(1.25)

# The two numbers the public gate actually measured (G0 verdict). They are the
# defaults so an illustrative payload is, by construction, gate-consistent.
# σ≈0.60 sits inside the measured 0.57–0.61 band; coverage is the empirical
# rolling-origin figure of the nominal-90% two-sided interval (it runs ~5pts wide).
MEASURED_SIGMA = 0.60
MEASURED_COVERAGE = 0.946


def pred25_from_sigma(sigma: float) -> float:
    """PRED(25) implied by a log-residual σ: 2·Φ(ln1.25/σ) − 1.

    Smaller σ (tighter scatter) -> more predictions land inside ±25% -> higher
    PRED(25). At the measured σ≈0.60 this is ≈0.29, not the ≈0.90 the old mock
    implied — which is the whole point of this module.
    """
    # Probability the log error stays within ±ln(1.25); doubled minus one turns
    # the one-sided CDF into the symmetric two-sided band probability.
    return float(2 * norm.cdf(LN_125 / sigma) - 1)


def sigma_for_pred25(target: float) -> float:
    """Inverse: the σ that yields a given PRED(25) target.

    Used to translate a PRED(25) target (e.g. 0.60) into its σ equivalent
    (e.g. σ ≤ 0.2651).
    """
    # Invert PRED = 2Φ(ln1.25/σ) − 1  ->  σ = ln1.25 / Φ⁻¹((PRED+1)/2).
    return float(LN_125 / norm.ppf((target + 1) / 2))


def quantile_ratio(sigma: float, q: float) -> float:
    """Multiplicative factor p_q / p50 for quantile `q` under log-normal σ."""
    # exp(Φ⁻¹(q)·σ): how far the q-th quantile sits from the median, in ratio.
    return float(math.exp(norm.ppf(q) * sigma))


def make_illustrative_payload(
    p50: float,
    sigma: float = MEASURED_SIGMA,
    coverage: float = MEASURED_COVERAGE,
) -> dict:
    """Build a gate-consistent illustrative estimate around a given median.

    Every field is derived from (`p50`, `sigma`): the [p5, p95] 90% two-sided
    interval, the implied PRED(25) and σ are mutually consistent by construction,
    so no example can ever show a band tighter than the measured scatter allows,
    nor an 80% band mislabelled as 90%.
    """
    # The nominal 90% TWO-SIDED interval: [p5, p95] = p50·exp(±1.6449·σ). This is
    # the band the measured 94.6% coverage refers to. (NOT [p10, p90], which would
    # be the central 80% band.)
    p05 = p50 * quantile_ratio(sigma, 0.05)
    p95 = p50 * quantile_ratio(sigma, 0.95)
    # One-sided 90%-non-exceedance bound, exposed SEPARATELY for contractual use
    # ("the job will exceed this only 10% of the time"). It is NOT an interval edge.
    p90_one_sided = p50 * quantile_ratio(sigma, 0.90)
    return {
        "estimation_summary": {
            # Median effort in hours — the single headline number.
            "p50_hours": round(p50),
            # The nominal 90% TWO-SIDED interval [p5, p95], rounded to whole hours.
            "hours_range_p5_p95": [round(p05), round(p95)],
            # Empirical coverage of THAT 90% interval, replacing confidence_score.
            "interval_coverage": coverage,
            # One-sided 90%-non-exceedance bound — labelled, never an interval edge.
            "p90_one_sided_hours": round(p90_one_sided),
            # The point accuracy this scatter implies — read alongside the band.
            "implied_pred25": round(pred25_from_sigma(sigma), 3),
            # The σ that generated everything above; the audit trail of the band.
            "log_residual_sigma": sigma,
        }
    }


def demo_target(p50: float = 340.0) -> dict:
    """The canonical demo payload: the single source of truth a downstream UI
    must reproduce exactly. By construction it carries no confidence_score and a
    90% two-sided [p5, p95] band derived from the measured σ."""
    return make_illustrative_payload(p50)


if __name__ == "__main__":
    # Print the canonical demo target so any downstream UI has ZERO ambiguity:
    # the 90% two-sided interval, the σ-derived ratio, the separately-labelled
    # one-sided p90, and the explicit removal of confidence_score.
    import json

    s = demo_target(340)["estimation_summary"]
    print(f"== CANONICAL DEMO PAYLOAD (σ measured = {MEASURED_SIGMA}) ==")
    print("  a downstream UI must reproduce these EXACTLY; do NOT reintroduce confidence_score")
    print(f"  p50_hours              = {s['p50_hours']}")
    print(f"  intervallo 90% [p5,p95]= {s['hours_range_p5_p95']}   (two-sided; z=Φ⁻¹(0.95)=1.6449)")
    print(f"  p95/p50                = exp(1.6449*σ) = {quantile_ratio(MEASURED_SIGMA, 0.95):.3f}")
    print(f"  interval_coverage      = {s['interval_coverage']}   (empirical coverage of THAT interval, NOT a 0-100 score)")
    print(f"  p90_one_sided_hours    = {s['p90_one_sided_hours']}   (90% non-exceedance, one-tail — NOT an interval edge)")
    print(f"  implied_pred25         = {s['implied_pred25']}")
    print(f"  log_residual_sigma     = {s['log_residual_sigma']}")
    assert "confidence_score" not in s, "confidence_score must never appear in a payload"
    print("\n  full JSON:")
    print(json.dumps(demo_target(340), indent=2))
