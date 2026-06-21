"""Payload coherence: an illustrative estimate must reflect the *measured* σ.

These tests lock in the σ-keyed payload and the rule that the "90% interval" is
the two-sided [p5, p95], not the 80% [p10, p90] band. They fail the moment an
example drifts back toward the old σ≈0.14 / confidence=88 fantasy, or re-labels
an 80% band as 90%."""

import math

from scipy.stats import norm

from metis_benchmark.payload import (
    MEASURED_COVERAGE,
    MEASURED_SIGMA,
    make_illustrative_payload,
    pred25_from_sigma,
    quantile_ratio,
    sigma_for_pred25,
)


def test_pred25_matches_measured_and_threshold_sigmas():
    # At the measured σ≈0.60 the implied point accuracy is ≈29% — far from 90%.
    assert math.isclose(pred25_from_sigma(0.60), 0.290, abs_tol=5e-4)
    # The old "vecchio payload" σ≈0.136 is what 90% PRED would have required.
    assert math.isclose(pred25_from_sigma(0.136), 0.899, abs_tol=1e-3)


def test_sigma_for_pred25_recovers_the_60pct_threshold():
    # A PRED(25) target of 60% is σ ≤ 0.2651 in log space.
    assert math.isclose(sigma_for_pred25(0.60), 0.2651, abs_tol=1e-4)
    # Round-trip: feeding that σ back through must return ~0.60.
    assert math.isclose(pred25_from_sigma(sigma_for_pred25(0.60)), 0.60, abs_tol=1e-9)


def test_quantile_ratio_is_the_lognormal_formula():
    # The 90% two-sided interval upper ratio (p95/p50) at the measured σ is ~2.68.
    assert math.isclose(quantile_ratio(0.60, 0.95), 2.683, abs_tol=1e-3)
    # The one-sided 90th-percentile ratio (p90/p50) is the smaller 2.157 — it is a
    # contractual bound, NOT the interval edge.
    assert math.isclose(quantile_ratio(0.60, 0.90), 2.157, abs_tol=1e-3)


def test_illustrative_payload_is_gate_consistent():
    payload = make_illustrative_payload(340)["estimation_summary"]
    # Canonical example: p50=340 -> 90% TWO-SIDED interval [127, 912] at measured σ.
    assert payload["p50_hours"] == 340
    assert payload["hours_range_p5_p95"] == [127, 912]
    # The old 80% band [158, 734] must NOT be the interval, under any key.
    assert payload.get("hours_range_p10_p90") is None
    assert payload["hours_range_p5_p95"] != [158, 734]
    # p90 survives only as a separately-labelled one-sided bound (734), never an edge.
    assert payload["p90_one_sided_hours"] == 734
    # No confidence_score anywhere — coverage replaces it.
    assert "confidence_score" not in payload
    assert payload["interval_coverage"] == MEASURED_COVERAGE
    # The implied accuracy travels with the band, and it is the honest ≈0.29.
    assert payload["implied_pred25"] == 0.29
    assert payload["log_residual_sigma"] == MEASURED_SIGMA


def test_p95_p50_is_exactly_the_two_sided_90_interval():
    # Canonical identity a downstream UI must honor: the 90% interval upper
    # edge is p95 = p50·exp(Φ⁻¹(0.95)·σ), NOT p90 (which gives the 80% band).
    for sigma in (0.60, 0.2651, 0.40):
        s = make_illustrative_payload(340, sigma=sigma)["estimation_summary"]
        ratio = s["hours_range_p5_p95"][1] / s["p50_hours"]
        # Round-trip through the rounded payload, so allow a small rounding slack.
        assert math.isclose(ratio, math.exp(norm.ppf(0.95) * sigma), rel_tol=0.01)
        assert "confidence_score" not in s


def test_band_is_never_tighter_than_the_measured_scatter():
    # Guardrail: at σ=0.60 the 90% interval upper/p50 must be ~exp(1.6449·0.6)=2.68,
    # never the 2.16 of the old 80% band.
    s = make_illustrative_payload(340)["estimation_summary"]
    lo, hi = s["hours_range_p5_p95"]
    assert hi / s["p50_hours"] >= 2.68
