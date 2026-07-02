"""Post-G0 R&D — inverse-variance assist blend, adequately powered (non gate).

The G0 verdict (NO-GO) stands and is not the object of this experiment. The
assist claim was withdrawn at G0 because its only paired test (kitchenham,
n=73) was under-powered (McNemar p=0.2266, power ~ 0.16). This script re-asks
the applied question with (a) an information-set weight instead of the
product's hand-picked fixed weight, and (b) every dataset that records an
expert estimate, pooled, with an explicit power analysis. SEERA is a sealed
holdout and is excluded from every number here.

Pre-declared design (frozen before any result was computed):
  Datasets & protocol splits (the gate's own splits, unchanged):
    - kitchenham: rolling-origin on date; gate model = log-space GP on the
      cold features (Adjusted.function.points, Client.code); expert =
      First.estimate.
    - sip: rolling-origin on date; gate model = k-NN Nadaraya-Watson over the
      cached bge-small embeddings, (k, tau) test-blind selected per fold;
      expert = HoursEstimate.
    - josse: grouped k-fold by project; same k-NN NW engine; expert = the
      annotated subset (expert_estimated_effort > 0, ~19% of rows).
  Sigma measurement (out-of-sample, no test peeking): per fold the model fits
  on the TRAIN block only; sigma_m and sigma_e are the SDs of log residuals on
  the CALIBRATION block (the same block the gate reserves for conformal
  calibration — here it calibrates the blend weight), on rows where the expert
  estimate exists. The fold's test predictions are then blended with
  w = sigma_m^-2 / (sigma_m^-2 + sigma_e^-2) computed from those cal-block
  sigmas: the weight a deployed system could actually have used.
  Arms (evaluated on pooled out-of-fold test rows having an expert estimate):
    1. expert alone;  2. fixed-0.5 log blend (the product's current mode);
    3. IVW blend.  (Model alone is printed for context, not compared.)
  Primary test: exact McNemar on PRED(25) hits, IVW blend vs expert alone,
  per dataset AND pooled. Pooling method: paired hit vectors are concatenated
  across datasets, i.e. the discordant pairs b and c are summed over datasets
  and the exact binomial is applied to the pooled discordant count (each
  project contributes at most one pair; under H0 every discordant pair is a
  fair coin regardless of its dataset). PRED(25) is unit-free, so pooling
  hits across datasets with different effort units is legitimate.
  Power: observed discordant share -> minimal detectable effect via
  evaluation.mcnemar_mde, achieved power at the observed pi1 by inverting
  evaluation.mcnemar_discordant_for_power, required N via
  evaluation.mcnemar_sample_size. If the pooled test is still under-powered,
  that is the reported answer.

Usage: PYTHONPATH=src python scripts/run_assist_ivw.py
"""

from __future__ import annotations

import numpy as np

from metis_benchmark.assist import blend_log_space, inverse_variance_weight, log_residual_sigma
from metis_benchmark.datasets.loaders import load
from metis_benchmark.evaluation import (
    bootstrap_metric_ci,
    mcnemar_discordant_for_power,
    mcnemar_exact,
    mcnemar_mde,
    mcnemar_sample_size,
    mdape,
    paired_bootstrap_diff,
    pred_at,
    pred_hits,
    rolling_origin_split,
)
from metis_benchmark.track_a.gp import LogGaussianProcess
from metis_benchmark.track_b.experiment import load_emb_dataset, select_k_tau
from metis_benchmark.track_b.knn import NadarayaWatsonKNN

# --- Frozen constants (mirror run_assist_paired.py where they overlap) -------
# Kitchenham's gate-legal cold features and its recorded expert estimate.
KITCH_COLD_FEATURES = ["Adjusted.function.points", "Client.code"]
# Bootstrap replicates and seed: fixed for bit-for-bit reproducibility.
N_BOOT = 2000
SEED = 0
# Canonical effect size for the "how big a study would we need" line: a 65/35
# discordant split, the moderate-edge convention used in the G0 power note.
PI1_PLANNING = 0.65


def _fold_blend(
    y: np.ndarray,
    expert: np.ndarray,
    model_cal: np.ndarray,
    model_test: np.ndarray,
    cal_idx: np.ndarray,
    test_idx: np.ndarray,
    has_expert: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, tuple[float, float]] | None:
    """Blend one fold's test predictions with the cal-block IVW weight.

    Returns (ivw_test, fixed_test, w, (sigma_m, sigma_e)) for the fold's test
    rows, or None when the calibration block holds fewer than 2 expert rows
    (no sigma is estimable — the fold is skipped, honestly, not imputed).
    """
    # Sigma estimation rows: calibration block AND expert present. The mask
    # over cal_idx also selects the matching columns of model_cal (which is
    # aligned with cal_idx by construction).
    cal_mask = has_expert[cal_idx]
    cal_e = cal_idx[cal_mask]
    if len(cal_e) < 2:
        return None
    # Out-of-sample log-residual scales on the SAME rows for both estimators.
    sigma_m = log_residual_sigma(y[cal_e], model_cal[cal_mask])
    sigma_e = log_residual_sigma(y[cal_e], expert[cal_e])
    # The deployable weight: computed from pre-test data only.
    w = inverse_variance_weight(sigma_m, sigma_e)
    # Blend arms on the test rows that have an expert estimate.
    test_mask = has_expert[test_idx]
    te = test_idx[test_mask]
    m_test = model_test[test_mask]
    ivw = blend_log_space(m_test, expert[te], w)
    fixed = blend_log_space(m_test, expert[te], 0.5)
    return ivw, fixed, w, (sigma_m, sigma_e)


def run_kitchenham() -> dict:
    """Kitchenham arms under the gate's rolling-origin split (GP model)."""
    df = (
        load("kitchenham")
        .dropna(subset=["effort", "date", "expert_estimate", *KITCH_COLD_FEATURES])
        .reset_index(drop=True)
    )
    # Strictly positive effort and expert estimate: both are logged downstream.
    df = df[(df["effort"] > 0) & (df["expert_estimate"] > 0)].reset_index(drop=True)
    y = df["effort"].to_numpy(dtype=float)
    X = df[KITCH_COLD_FEATURES].to_numpy(dtype=float)
    expert = df["expert_estimate"].to_numpy(dtype=float)
    has_expert = np.ones(len(y), dtype=bool)  # kitchenham records one per project

    out = _collect(len(y))
    for fold in rolling_origin_split(df, "date"):
        # Fit on TRAIN only: the calibration block must stay out-of-sample for
        # the sigma estimates (the gate reserves it for conformal calibration;
        # the blend uses it for weight calibration — same discipline).
        gp = LogGaussianProcess(seed=SEED).fit(X[fold.train], y[fold.train])
        model_cal = gp.predict(X[fold.calibration])
        model_test = gp.predict(X[fold.test])
        _accumulate(out, y, expert, model_cal, model_test, fold.calibration, fold.test, has_expert)
    return {"key": "kitchenham", **out}


def run_track_b(key: str) -> dict:
    """sip / josse arms under the gate's Track B splits (k-NN NW model)."""
    data = load_emb_dataset(key)
    expert = np.where(np.isnan(data.expert), -1.0, data.expert)  # NaN -> sentinel
    has_expert = expert > 0  # josse marks "no annotation" as missing/-1
    y = data.y

    out = _collect(len(y))
    for fold in data.folds:
        # Test-blind (k, tau) selection inside the train rows, then fit the
        # retrieval index on TRAIN only (cal stays out-of-sample for sigma).
        k, tau = select_k_tau(data.emb[fold.train], y[fold.train])
        model = NadarayaWatsonKNN(k=k, tau=tau).fit(data.emb[fold.train], y[fold.train])
        model_cal = model.predict(data.emb[fold.cal])
        model_test = model.predict(data.emb[fold.test])
        _accumulate(out, y, expert, model_cal, model_test, fold.cal, fold.test, has_expert)
    return {"key": key, **out}


def _collect(n: int) -> dict:
    """Empty accumulators for one dataset's out-of-fold arms."""
    return {
        "y": np.full(n, np.nan), "expert": np.full(n, np.nan),
        "model": np.full(n, np.nan), "ivw": np.full(n, np.nan),
        "fixed": np.full(n, np.nan), "tested": np.zeros(n, dtype=bool),
        "w_folds": [], "sigma_folds": [],
    }


def _accumulate(out, y, expert, model_cal, model_test, cal_idx, test_idx, has_expert) -> None:
    """Blend one fold and write its test-row arms into the accumulators."""
    res = _fold_blend(y, expert, model_cal, model_test, cal_idx, test_idx, has_expert)
    if res is None:
        return  # fold skipped: no sigma estimable on its calibration block
    ivw, fixed, w, sigmas = res
    test_mask = has_expert[test_idx]
    te = test_idx[test_mask]
    out["y"][te] = y[te]
    out["expert"][te] = expert[te]
    out["model"][te] = model_test[test_mask]
    out["ivw"][te] = ivw
    out["fixed"][te] = fixed
    out["tested"][te] = True
    out["w_folds"].append(w)
    out["sigma_folds"].append(sigmas)


def _power_lines(n: int, disc: int, b: int) -> None:
    """Print the explicit power analysis for one McNemar comparison."""
    from scipy.stats import norm

    p_disc = disc / n if n else 0.0
    print(f"     power: n={n}, discordant={disc} (p_disc={p_disc:.3f})")
    if disc == 0:
        print("     no discordant pairs -> the test carries zero information here")
        return
    # Minimal detectable effect at this n and discordant share (80% power).
    mde = mcnemar_mde(n, p_disc)
    mde_txt = f"pi1 >= {mde:.3f}" if mde is not None else "none (even pi1->1 is out of reach)"
    print(f"     MDE at 80% power: {mde_txt}")
    # Achieved power at the OBSERVED discordant split, by inverting the
    # Connor sample-size formula for z_beta (the same formula the G0 note used).
    pi1_obs = max(b, disc - b) / disc
    if pi1_obs > 0.5:
        za = float(norm.ppf(1 - 0.05 / 2))
        zb = (np.sqrt(disc) * abs(pi1_obs - 0.5) - za * 0.5) / np.sqrt(pi1_obs * (1 - pi1_obs))
        print(f"     achieved power at observed pi1={pi1_obs:.3f}: {float(norm.cdf(zb)):.2f}")
    # Study size needed for the canonical moderate edge (65/35 split).
    need = mcnemar_sample_size(PI1_PLANNING, p_disc) if p_disc > 0 else None
    if need:
        print(f"     N needed for pi1={PI1_PLANNING} at 80% power: "
              f"~{need['n_projects']:.0f} projects ({need['discordant_pairs']:.0f} discordant)")


def report(res: dict) -> dict:
    """Print one dataset's pre-declared comparison; return its paired hits."""
    m = res["tested"]
    y, expert = res["y"][m], res["expert"][m]
    arms = {"expert alone": expert, "blend fixed-0.5": res["fixed"][m], "blend IVW": res["ivw"][m]}
    w_lo, w_hi = min(res["w_folds"]), max(res["w_folds"])
    sig = np.array(res["sigma_folds"])  # per-fold (sigma_m, sigma_e)

    print(f"\n== {res['key']} (n_test={m.sum()}, folds kept={len(res['w_folds'])}) ==")
    print(f"  cal-block sigmas per fold: sigma_m median={np.median(sig[:, 0]):.3f}, "
          f"sigma_e median={np.median(sig[:, 1]):.3f} -> w(model) in [{w_lo:.2f}, {w_hi:.2f}]")
    # Descriptive out-of-fold sigmas on the pooled TEST rows (not used for w).
    print(f"  test-block sigmas (descriptive): sigma_m={log_residual_sigma(y, res['model'][m]):.3f}, "
          f"sigma_e={log_residual_sigma(y, expert):.3f}")
    for name, pred in arms.items():
        pci = bootstrap_metric_ci(y, pred, pred_at, n_boot=N_BOOT, seed=SEED)
        mci = bootstrap_metric_ci(y, pred, mdape, n_boot=N_BOOT, seed=SEED)
        print(f"  {name:<16} PRED(25)={pci}  MdAPE={mci}")

    hits = {name: pred_hits(y, pred) for name, pred in arms.items()}
    # Primary paired test: IVW blend vs the expert alone.
    mc = mcnemar_exact(hits["blend IVW"], hits["expert alone"])
    dP = paired_bootstrap_diff(y, arms["blend IVW"], expert, pred_at, n_boot=N_BOOT, seed=SEED)
    print(f"  IVW vs expert: McNemar b={mc['b']} c={mc['c']} discordant={mc['discordant']} "
          f"p={mc['p']:.4f}; dPRED25={dP.point:+.1%} [{dP.low:+.1%}, {dP.high:+.1%}]")
    _power_lines(len(y), mc["discordant"], mc["b"])
    # Secondary: the product's current fixed-0.5 blend vs the expert.
    mcf = mcnemar_exact(hits["blend fixed-0.5"], hits["expert alone"])
    print(f"  fixed-0.5 vs expert: McNemar b={mcf['b']} c={mcf['c']} p={mcf['p']:.4f}")
    return hits


def main() -> None:
    print("== Post-G0 R&D: inverse-variance assist (non gate-carrying; G0 NO-GO stands) ==")
    print("  excluded: seera — sealed holdout (opened once for the G0 verdict)")

    results = [run_kitchenham(), run_track_b("sip"), run_track_b("josse")]
    all_hits: list[dict] = [report(r) for r in results]

    # --- Pooled analysis (method pre-declared in the module docstring) ------
    pooled = {
        name: np.concatenate([h[name] for h in all_hits])
        for name in ("expert alone", "blend fixed-0.5", "blend IVW")
    }
    n = len(pooled["expert alone"])
    print(f"\n== POOLED (concatenated paired hits; n={n}) ==")
    mc = mcnemar_exact(pooled["blend IVW"], pooled["expert alone"])
    print(f"  IVW vs expert: McNemar b={mc['b']} c={mc['c']} discordant={mc['discordant']} "
          f"p={mc['p']:.4f}")
    _power_lines(n, mc["discordant"], mc["b"])
    mcf = mcnemar_exact(pooled["blend fixed-0.5"], pooled["expert alone"])
    print(f"  fixed-0.5 vs expert: McNemar b={mcf['b']} c={mcf['c']} "
          f"discordant={mcf['discordant']} p={mcf['p']:.4f}")
    # Sanity print: the discordant requirement for the planning effect size at
    # the pooled discordant share, to compare with what we actually have.
    print(f"  reference: 80%-power discordant requirement at pi1={PI1_PLANNING}: "
          f"{mcnemar_discordant_for_power(PI1_PLANNING):.0f} pairs; observed {mc['discordant']}")

    # --- POST-HOC diagnostics (not pre-declared; explanatory only) ----------
    # The IVW optimality proof assumes UNBIASED estimators with INDEPENDENT
    # errors. These two numbers check each assumption on the test rows:
    #   mean log residual  -> multiplicative bias the blend inherits;
    #   corr(res_m, res_e) -> when rho·sigma_m > sigma_e the optimal weight on
    #                         the model is actually <= 0 and ANY admixture hurts.
    print("\n-- post-hoc diagnostics (why; not part of the pre-declared comparison) --")
    for res in results:
        m = res["tested"]
        rm = np.log(res["y"][m]) - np.log(res["model"][m])   # model log residuals
        re = np.log(res["y"][m]) - np.log(res["expert"][m])  # expert log residuals
        rho = float(np.corrcoef(rm, re)[0, 1])
        sm, se = float(np.std(rm, ddof=1)), float(np.std(re, ddof=1))
        # The MSE-optimal weight on the model under correlated errors:
        # w* = (se^2 - rho·sm·se) / (sm^2 + se^2 - 2·rho·sm·se).
        w_star = (se**2 - rho * sm * se) / (sm**2 + se**2 - 2 * rho * sm * se)
        print(f"  {res['key']:<11} bias: mean(res_m)={rm.mean():+.3f}, mean(res_e)={re.mean():+.3f}; "
              f"rho={rho:+.3f} -> correlation-aware w*={w_star:+.3f}")


if __name__ == "__main__":
    main()
