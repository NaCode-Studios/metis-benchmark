"""Post-G0 R&D — the epoch/organization intercept prior (NOT gate-carrying).

The G0 verdict (NO-GO on cold-start point precision) stands and is not touched
by this experiment. What G0 exposed on real projects — a 6-8x multiplicative
bias — is, in log space, an intercept offset between the training epoch and
the target organization. The gate machinery refits the intercept per dataset
(it "carries the unit"), which silently assumes local actuals exist. Cold
start has none, so the honest question is: how much does the log-productivity
intercept VARY between organizations/epochs? That variance, tau_a^2, is a
measurable prior — e^(±2·tau_a) is the multiplicative range a cold estimate
should declare before any local recalibration.

Method (declared before looking at any number):
  1. Contributors are the tabular datasets whose size is in function points
     and whose effort is (convertible to) person-hours, so every intercept
     lives in ONE unit system: desharnais, kitchenham, maxwell, china, and
     albrecht (effort shipped in kilo person-hours, converted x1000 per the
     dataset's documentation). In log space a unit mismatch is an additive
     intercept offset indistinguishable from organization bias, so datasets
     with incommensurable units must stay out rather than be "converted" with
     a made-up constant.
  2. Exclusions: SEERA (sealed holdout, opened once for the G0 verdict —
     never reused, per protocol), cocomo81 (size in KLOC and effort in
     person-months: the KLOC->FP gearing factor is language-dependent
     folklore, and any chosen constant would masquerade as epoch bias).
  3. Per dataset: intercept a_d = mean log(effort/size) (elasticity pinned at
     1 by the normalization), se_d = sd/sqrt(n). DerSimonian-Laird across
     datasets splits the observed spread of a_d into within-noise and the
     between-dataset tau_a^2. Q-profile 95% CI for tau_a (k is small, so a
     bootstrap over datasets would be noise).
  4. Sensitivity: the mean-productivity intercept confounds size-mix with
     productivity when the true elasticity is not 1; re-run with intercepts
     taken at the POOLED slope b_0 (a_d = mean(log effort - b_0 log size),
     i.e. size replaced by size^b_0) to check tau_a is not a size-mix artifact.
  5. Sanity check: does the observed 6-8x real-project bias (log 6 ~ 1.79,
     log 8 ~ 2.08) fall inside e^(±2·tau_a)?

Usage: PYTHONPATH=src python scripts/run_intercept_prior.py
"""

from __future__ import annotations

import numpy as np

# Canonical loaders: every dataset arrives with `size` and `effort` columns.
from metis_benchmark.datasets.loaders import load

# The gate's slope pooling (for the sensitivity) and the new intercept prior.
from metis_benchmark.track_a.pooling import PooledPowerLaw, fit_intercept_prior

# --- Frozen experiment constants (declared before any result) ---------------
# Contributors and their effort conversion factor to person-hours. Size is in
# (adjusted) function points for all five, so intercepts share one unit:
# log(person-hours per function point).
CONTRIBUTORS: dict[str, float] = {
    "desharnais": 1.0,     # person-hours, adjusted FP
    "kitchenham": 1.0,     # person-hours, adjusted FP
    "maxwell": 1.0,        # person-hours, FP
    "china": 1.0,          # person-hours, adjusted FP
    "albrecht": 1000.0,    # kilo person-hours -> person-hours
}
# Excluded with the reason printed (and repeated in the report).
EXCLUDED: dict[str, str] = {
    "seera": "sealed holdout (opened once for the G0 verdict; never reused)",
    "cocomo81": "size in KLOC, effort in person-months: no honest FP conversion",
}
# The real-project bias G0 observed, as log offsets, for the sanity check.
OBSERVED_BIAS = {"6x": np.log(6.0), "8x": np.log(8.0)}


def contributions() -> tuple[list[str], list[tuple[np.ndarray, np.ndarray]]]:
    """(names, [(size, effort-in-person-hours)]) for the declared contributors."""
    names, contribs = [], []
    for key, factor in CONTRIBUTORS.items():
        df = load(key)
        # Valid rows only: log-space needs strictly positive size and effort.
        ok = df["size"].notna() & df["effort"].notna() & (df["size"] > 0) & (df["effort"] > 0)
        size = df.loc[ok, "size"].to_numpy(dtype=float)
        # Unit conversion to person-hours happens HERE, before pooling, so the
        # intercepts are commensurable (see module docstring, point 1).
        effort = df.loc[ok, "effort"].to_numpy(dtype=float) * factor
        names.append(key)
        contribs.append((size, effort))
    return names, contribs


def report_prior(title: str, names: list[str], prior) -> None:
    """Print one fitted prior: per-dataset intercepts, tau_a, and the ranges."""
    print(f"\n-- {title} --")
    print(f"  {'dataset':<12} {'n':>4} {'a_d':>7} {'se_d':>6}   e^a_d (h per FP)")
    for name, a, se, n in zip(names, prior.intercepts, prior.std_errors, prior.n_projects):
        print(f"  {name:<12} {n:>4} {a:>7.3f} {se:>6.3f}   {np.exp(a):>7.2f}")
    lo, hi = prior.tau_a_ci
    print(f"  pooled intercept a_0 = {prior.a_0:.3f}  (e^a_0 = {np.exp(prior.a_0):.2f} h/FP)")
    print(f"  tau_a = {prior.tau_a:.3f}  [Q-profile 95% CI {lo:.3f}, {hi:.3f}]")
    print(f"  e^(+-tau_a)  = x{np.exp(prior.tau_a):.2f} / x{np.exp(-prior.tau_a):.2f}"
          f"   (68% cold-start scale range)")
    print(f"  e^(+-2tau_a) = x{np.exp(2 * prior.tau_a):.2f} / x{np.exp(-2 * prior.tau_a):.2f}"
          f"   (95% cold-start scale range)")
    print(f"  CI-upper 95% range: e^(+-2*{hi:.3f}) = x{np.exp(2 * hi):.2f}")


def main() -> None:
    print("== Post-G0 R&D: intercept prior (non gate-carrying; G0 NO-GO stands) ==")
    for key, why in EXCLUDED.items():
        print(f"  excluded: {key} — {why}")

    names, contribs = contributions()

    # --- Primary: elasticity pinned at 1 (mean log-productivity) ------------
    prior = fit_intercept_prior(contribs)
    report_prior("primary: a_d = mean log(effort/size)", names, prior)

    # --- Sensitivity: intercept at the pooled slope b_0 ----------------------
    # log(effort) - b_0 log(size) = log(effort / size^b_0): feeding size^b_0
    # reuses fit_intercept_prior unchanged, with the size-mix confound removed.
    pooled = PooledPowerLaw.fit_global(contribs)
    contribs_b0 = [(size**pooled.b_0, effort) for size, effort in contribs]
    prior_b0 = fit_intercept_prior(contribs_b0)
    print(f"\n  pooled slope for the sensitivity: b_0 = {pooled.b_0:.3f} (tau_b^2 = {pooled.tau2:.4f})")
    report_prior(f"sensitivity: a_d at pooled slope b_0 = {pooled.b_0:.3f}", names, prior_b0)

    # --- Sanity check vs the observed real-project bias ---------------------
    print("\n-- sanity check: does the observed 6-8x bias fit the prior? --")
    two_tau = 2 * prior.tau_a
    two_tau_hi = 2 * prior.tau_a_ci[1]
    for label, lg in OBSERVED_BIAS.items():
        inside_point = "inside" if lg <= two_tau else "OUTSIDE"
        inside_ci = "inside" if lg <= two_tau_hi else "OUTSIDE"
        print(f"  log({label}) = {lg:.2f} vs 2*tau_a = {two_tau:.2f} -> {inside_point} the point prior; "
              f"vs 2*tau_hi = {two_tau_hi:.2f} -> {inside_ci} the CI-upper prior")


if __name__ == "__main__":
    main()
