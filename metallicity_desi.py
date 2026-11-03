"""Gas-phase oxygen abundances from DESI FastSpecFit line fluxes.

Jiang et al. (2019, ApJ 872, 145) R23+O32 calibration and its adopted
Izotov et al. (2006, A&A 448, 955) direct-Te method.

Input: a mapping/pandas Series with FastSpecFit LINE_FLUX and LINE_FLUX_IVAR.
Output: flat dictionary of `metal_*` columns, suitable for CSV output.

Assumptions: input fluxes have already been corrected for Galactic foreground
extinction (check FastSpecFit provenance); internal nebular extinction is
calculated from Halpha/Hbeta with Calzetti (2000), R_V=4.05, Case B=2.86.
The observed Balmer decrement is floored at E(B-V)=0 by default.

The 0.14-dex Jiang empirical scatter is NOT included in MC statistical errors.
A Te detection is necessary but not sufficient for a robust abundance; review
line fitting and derived temperatures/abundances before scientific selection.
"""

from __future__ import annotations

import numpy as np

LINES = {
    "OII_3726": 3726.03, "OII_3729": 3728.82,
    "OIII_4363": 4363.21, "HBETA": 4861.33,
    "OIII_4959": 4958.91, "OIII_5007": 5006.84,
    "HALPHA": 6562.80, "SII_6716": 6716.44, "SII_6731": 6730.82,
}
A, B, C, D, E = -24.135, 6.1532, -0.37866, -0.147, -7.071
CALIBRATION_SCATTER_DEX = 0.14  # Jiang high-O32 empirical residual scatter


def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def calzetti_k(wavelength):
    """Calzetti (2000) attenuation k(lambda), lambda in Angstrom."""
    w = np.asarray(wavelength, dtype=float) / 1e4
    with np.errstate(divide="ignore", invalid="ignore"):
        short = 2.659 * (-2.156 + 1.509 / w - 0.198 / w**2 + 0.011 / w**3) + 4.05
        long = 2.659 * (-1.857 + 1.040 / w) + 4.05
    return np.where(w < 0.63, short, long)


def _density(s6716, s6731, iv6716, iv6731):
    """Jiang [SII] approximation; default 100 cm^-3 if S/N <=2."""
    good = (np.isfinite(s6716) & np.isfinite(s6731) &
            (s6716 > 0) & (s6731 > 0) &
            np.isfinite(iv6716) & np.isfinite(iv6731) &
            (iv6716 > 0) & (iv6731 > 0) &
            (s6716 * np.sqrt(iv6716) > 2) &
            (s6731 * np.sqrt(iv6731) > 2))
    ratio = np.divide(s6716, s6731, out=np.full_like(np.asarray(s6716, dtype=float), np.nan), where=np.asarray(s6731) != 0)
    ne = np.full_like(np.asarray(ratio), 100.0, dtype=float)
    ne = np.where(good & (ratio > 1.43), 10**0.5, ne)
    ne = np.where(good & (ratio < 0.51), 1e4, ne)
    mid = good & (ratio >= 0.51) & (ratio <= 1.43)
    with np.errstate(divide="ignore", invalid="ignore"):
        calc = (0.4441 * 2514 - ratio * 779.3) / (ratio - 0.4441)
    ne = np.where(mid & np.isfinite(calc) & (calc > 0), calc, ne)
    flag = np.where(~good, "default_100", np.where(ratio > 1.43, "low_limit", np.where(ratio < 0.51, "high_limit", "measured")))
    return ne, flag


def _te_abundance(f, ne):
    """Vectorized Te abundance; f is dict of internally dereddened arrays."""
    oii = f["OII_3726"] + f["OII_3729"]
    oiii = f["OIII_4959"] + f["OIII_5007"]
    hb = f["HBETA"]
    aur = f["OIII_4363"]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        ratio = oiii / aur
        t3 = np.full_like(np.asarray(ratio, dtype=float), 1.3)
        for _ in range(60):
            x = 1e-4 * ne / np.sqrt(t3)
            ct = (8.44 - 1.09*t3 + 0.5*t3*t3 - 0.08*t3**3) * (1 + 0.0004*x) / (1 + 0.044*x)
            next_t3 = 1.432 / np.log10(ratio / ct)
            # Do not silently force unphysical iterates into a plausible range.
            t3 = np.where(np.isfinite(next_t3) & (next_t3 > 0), next_t3, np.nan)
        t2 = -0.577 + t3 * (2.065 - 0.498*t3)
        # Izotov (2006) O+ density term uses t2 (Jiang typesets t3).
        op = (np.log10(oii/hb) + 5.961 + 1.676/t2 - 0.40*np.log10(t2)
              - 0.034*t2 + np.log10(1 + 1.35e-4*ne/np.sqrt(t2)))
        opp = np.log10(oiii/hb) + 6.200 + 1.251/t3 - 0.55*np.log10(t3) - 0.014*t3
        oh = 12 + np.log10(10**(op-12) + 10**(opp-12))
    valid = ((oii > 0) & (oiii > 0) & (hb > 0) & (aur > 0) &
             np.isfinite(oh) & np.isfinite(t2) & np.isfinite(t3) &
             (t2 > 0) & (t3 > 0))
    return (np.where(valid, oh, np.nan),
            np.where(valid, t3*1e4, np.nan),
            np.where(valid, t2*1e4, np.nan))


def _r23(f, branch_threshold=0.6):
    """Vectorized Jiang quadratic inversion. Return both roots and flags."""
    oii = f["OII_3726"] + f["OII_3729"]
    oiii = f["OIII_4959"] + f["OIII_5007"]
    hb = f["HBETA"]
    with np.errstate(divide="ignore", invalid="ignore"):
        y = np.log10(oiii / oii)
        r = np.log10((oii + oiii) / hb)
        bb = B - D*y
        cc = A - D*E*y - r
        disc = bb**2 - 4*C*cc
        turn = -bb/(2*C)
        root1 = (-bb + np.sqrt(np.maximum(disc, 0))) / (2*C)
        root2 = (-bb - np.sqrt(np.maximum(disc, 0))) / (2*C)
    valid = (oii > 0) & (oiii > 0) & (hb > 0) & np.isfinite(y) & np.isfinite(r)
    lower = np.where(valid & (disc >= 0), np.minimum(root1, root2), np.nan)
    upper = np.where(valid & (disc >= 0), np.maximum(root1, root2), np.nan)
    chosen = np.where(disc < 0, turn, np.where(y > branch_threshold, lower, upper))
    chosen = np.where(valid, chosen, np.nan)
    flag = np.where(~valid, "invalid", np.where(disc < 0, "turnover",
                    np.where(y > branch_threshold, "lower_branch", "upper_branch_ambiguous")))
    flag = np.where(valid & ((chosen < 7.2) | (chosen > 8.6)), np.char.add(flag.astype(str), "_extrapolated"), flag)
    return r, y, chosen, lower, upper, turn, flag


def _evaluate(flux, ivar, branch_threshold):
    """Calculate all physical quantities for scalar or MC vector inputs."""
    hb, ha = flux["HBETA"], flux["HALPHA"]
    with np.errstate(divide="ignore", invalid="ignore"):
        ebv_raw = (2.5 / (float(calzetti_k(4861.33)) - float(calzetti_k(6562.80)))
                    * np.log10((ha/hb) / 2.86))
    ebv = np.where((ha > 0) & (hb > 0) & np.isfinite(ebv_raw), np.maximum(ebv_raw, 0), np.nan)
    with np.errstate(over="ignore", invalid="ignore"):
        corrected = {name: flux[name] * 10**(0.4 * calzetti_k(wave) * ebv)
                     for name, wave in LINES.items()}
    ne, neflag = _density(corrected["SII_6716"], corrected["SII_6731"],
                          ivar["SII_6716"], ivar["SII_6731"])
    ohte, te3, te2 = _te_abundance(corrected, ne)
    logr, logy, ohr, lo, hi, turn, rflag = _r23(corrected, branch_threshold)
    return dict(EBV_gas=ebv, ne_SII=ne, ne_flag=neflag, OH_Te=ohte,
                Te_OIII=te3, Te_OII=te2, log_R23=logr, log_O32=logy,
                OH_R23=ohr, OH_R23_lower=lo, OH_R23_upper=hi,
                OH_R23_turnover=turn, R23_flag=rflag)


def _empty():
    numeric = ["OH_Te", "OH_Te_MC50", "OH_Te_MC16", "OH_Te_MC84",
               "OH_Te_err_lo", "OH_Te_err_hi", "OH_R23", "OH_R23_MC50",
               "OH_R23_MC16", "OH_R23_MC84", "OH_R23_err_lo",
               "OH_R23_err_hi", "OH_R23_lower", "OH_R23_upper",
               "OH_R23_turnover", "OH_best", "EBV_gas", "ne_SII",
               "Te_OIII", "Te_OII", "log_R23", "log_O32", "SNR_4363",
               "Te_MC_valid_fraction", "R23_MC_valid_fraction"]
    out = {"metal_"+k: np.nan for k in numeric}
    out.update(metal_OH_method="none", metal_Te_flag="not_measured",
               metal_R23_flag="invalid", metal_ne_flag="unknown",
               metal_error="", metal_R23_scatter_dex=CALIBRATION_SCATTER_DEX)
    return out


def calculate_metallicity(fastspec_row, n_mc=1000, te_snr_threshold=3.0,
                          branch_threshold=0.6, seed=None):
    """Calculate both gas-phase O/H estimators from one FastSpecFit row.

    All keys prefixed `metal_`. Invalid inputs yield NaNs and QC flags.
    MC samples each individual flux independently with sigma=1/sqrt(IVAR).
    If a needed IVAR is absent, the deterministic estimator still runs,
    but its MC uncertainty is unavailable (rather than assumed zero).
    For logO32 <= branch_threshold, the upper branch is a *heuristic*, not
    a reliable branch determination. It is flagged as ambiguous.
    """
    if n_mc < 0:
        raise ValueError("n_mc must be nonnegative")
    out = _empty()
    try:
        f0 = {name: np.asarray(_float(fastspec_row.get(name+"_FLUX", np.nan))) for name in LINES}
        iv0 = {name: np.asarray(_float(fastspec_row.get(name+"_FLUX_IVAR", np.nan))) for name in LINES}
        aur = float(f0["OIII_4363"])
        auriv = float(iv0["OIII_4363"])
        snr = aur * np.sqrt(auriv) if np.isfinite(auriv) and auriv > 0 else np.nan
        out["metal_SNR_4363"] = snr
        ev = _evaluate(f0, iv0, branch_threshold)
        for key, val in ev.items():
            out["metal_"+key] = val.item() if isinstance(val, np.ndarray) and val.ndim == 0 else val
        te_eligible = np.isfinite(snr) and snr >= te_snr_threshold
        if not te_eligible:
            out["metal_OH_Te"] = np.nan
            out["metal_Te_OIII"] = np.nan
            out["metal_Te_OII"] = np.nan
            out["metal_Te_flag"] = "low_snr_4363"
        elif np.isfinite(out["metal_OH_Te"]):
            out["metal_Te_flag"] = "candidate"
        else:
            out["metal_Te_flag"] = "invalid_lines_or_temperature"
        # Strong-line result may be valid even if the auroral line is unusable.
        if np.isfinite(out["metal_OH_Te"]):
            out["metal_OH_best"] = out["metal_OH_Te"]
            out["metal_OH_method"] = "Te_candidate"
        elif (np.isfinite(out["metal_OH_R23"]) and
              out["metal_R23_flag"] == "lower_branch"):
            out["metal_OH_best"] = out["metal_OH_R23"]
            out["metal_OH_method"] = "Jiang_R23"
        # Turnover, ambiguous-branch, and extrapolated estimates are retained
        # in OH_R23 but not automatically promoted to OH_best.
        if n_mc:
            rng = np.random.default_rng(seed)
            # Do not sample unmeasured lines as constants: they may affect
            # EBV/density. Density-line IVAR absent -> retain deterministic
            # density default; auroral IVAR absent -> no Te MC.
            samples = {}
            for name in LINES:
                v, iv = float(f0[name]), float(iv0[name])
                if np.isfinite(v) and np.isfinite(iv) and iv > 0:
                    samples[name] = rng.normal(v, 1/np.sqrt(iv), n_mc)
                else:
                    samples[name] = np.full(n_mc, v)
            # Reuse density fallback when [SII] uncertainty missing.
            ivsamples = {k: np.full(n_mc, float(v)) for k, v in iv0.items()}
            emc = _evaluate(samples, ivsamples, branch_threshold)
            for label, key, required in [
                ("Te", "OH_Te", ["OII_3726", "OII_3729", "OIII_4363", "OIII_4959", "OIII_5007", "HBETA", "HALPHA"]),
                ("R23", "OH_R23", ["OII_3726", "OII_3729", "OIII_4959", "OIII_5007", "HBETA", "HALPHA"]),
            ]:
                if label == "Te" and not te_eligible:
                    continue
                if not all(np.isfinite(float(iv0[k])) and float(iv0[k]) > 0 for k in required):
                    continue
                vals = np.asarray(emc[key], dtype=float)
                # MC percentiles only from valid draws; no forced clipping.
                valid = np.isfinite(vals)
                out[f"metal_{label}_MC_valid_fraction"] = float(np.mean(valid))
                if np.count_nonzero(valid) < max(10, int(0.5*n_mc)):
                    continue
                p16, p50, p84 = np.percentile(vals[valid], [16, 50, 84])
                out[f"metal_OH_{label}_MC16"] = p16
                out[f"metal_OH_{label}_MC50"] = p50
                out[f"metal_OH_{label}_MC84"] = p84
                out[f"metal_OH_{label}_err_lo"] = p50-p16
                out[f"metal_OH_{label}_err_hi"] = p84-p50
    except Exception as exc:
        out["metal_error"] = f"{type(exc).__name__}: {exc}"
    return out
