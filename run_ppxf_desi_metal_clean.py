#!/Users/dhvanildesai/miniforge3/envs/hosts/bin/python3
"""
run_ppxf_desi_metal.py
======================
Command-line runner for combined stellar + gas (+ photometry) pPXF fitting of
DESI spectra, with optional gas-phase metallicities from FastSpecFit lines.

Galaxies are identified by TARGETID when the pickle has one, otherwise by
SPARCL specid; --specid takes whichever of these the pickle uses.

Examples
--------
    # first 3 galaxies, quick look, with photometry
    python run_ppxf_desi_metal.py examples/desi_spectra.pkl -n 3 \\
        -p examples/DESI_starlight_phot.csv

    # specific galaxies
    python run_ppxf_desi_metal.py examples/MPA_matches.pkl \\
        -p examples/MPA_phot_Legacy.csv \\
        --specid 39633307708689227 39627790982585523

    # all galaxies, with bootstrap errors, writing to a named file
    python run_ppxf_desi_metal.py desi_spectra.pkl -n -1 --nboot 50 -o results.csv

    # untie the Balmer lines to get a free Halpha/Hbeta ratio
    python run_ppxf_desi_metal.py desi_spectra.pkl -n 3 --no-tie-balmer

    # compare template libraries on the same galaxies
    python run_ppxf_desi_metal.py desi_spectra.pkl -n 3 --sps emiles

    # add gas-phase metallicities from a FastSpecFit catalog
    python run_ppxf_desi_metal.py desi_spectra.pkl --fastspec-file fastspec.csv

Once you have a CSV, plot_ppxf_desi_summary.py builds the aggregate
population-level plots (age/metallicity distributions, LW vs MW, stellar
mass, kinematics) across all fitted galaxies.
"""

import argparse
import csv
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
# matplotlib.use("Agg")   # uncomment to save plots without a display (ssh)
import matplotlib.pyplot as plt

from ppxf_desi import fit_galaxy, load_desi_pickle, results_to_rows


# ======================================================================
# Photometry
# ======================================================================
PHOT_BANDS = ("g", "r", "z")                    # Legacy Survey DECam bands
PHOT_LAM = np.array([4863.0, 6463.0, 9201.0])   # A, approx. effective wavelengths
NANOMAGGY_FNU = 3.631e-29                       # erg/s/cm^2/Hz per nanomaggy
C_ANGSTROM_PER_S = 2.99792458e18
PHOT_SYS_FLOOR = 0.01                           # fractional systematic error


def build_photometry(phot_df, targetid, debug=False):
    """
    Legacy Survey g/r/z photometry for one galaxy, in the form fit_galaxy
    expects: MW-corrected f_lambda in 1e-17 erg/s/cm^2/A, with a 1%
    systematic floor added in quadrature to the errors.
    """
    rows = phot_df[phot_df["TARGETID"] == int(targetid)]
    if rows.empty:
        raise RuntimeError(f"photometry not found for gal {targetid}")
    p = rows.iloc[0]

    flux = p[[f"flux_{b}" for b in PHOT_BANDS]].to_numpy(dtype=float)
    flux_ivar = p[[f"flux_ivar_{b}" for b in PHOT_BANDS]].to_numpy(dtype=float)
    transmission = p[[f"mw_transmission_{b}" for b in PHOT_BANDS]].to_numpy(dtype=float)

    if np.any(~np.isfinite(flux_ivar) | (flux_ivar <= 0)):
        raise ValueError(f"Invalid photometric inverse variance for TARGETID "
                         f"{targetid}: {flux_ivar}")
    if np.any(~np.isfinite(transmission) | (transmission <= 0)):
        raise ValueError(f"Invalid MW transmission for TARGETID {targetid}: "
                         f"{transmission}")

    # Galactic extinction, then the systematic floor.
    flux_corr = flux / transmission
    fluxerr_corr = (1.0 / np.sqrt(flux_ivar)) / transmission
    fluxerr_corr = np.hypot(fluxerr_corr, PHOT_SYS_FLOOR * flux_corr)

    # nanomaggies -> f_lambda [1e-17 erg/s/cm^2/A]:  f_lambda = f_nu c / lambda^2
    to_flam = NANOMAGGY_FNU * C_ANGSTROM_PER_S / PHOT_LAM**2 * 1e17
    phot_flam = flux_corr * to_flam
    phot_flamerr = fluxerr_corr * to_flam

    if debug:
        print("\n=== PHOTOMETRY DIAGNOSTICS ===")
        print("phot_flam:", phot_flam)
        print("phot_flamerr:", phot_flamerr)

    return {"phot_galaxy": phot_flam, "noise": phot_flamerr,
            "lam": PHOT_LAM, "flux_orig": flux.copy()}


# ======================================================================
# Plots
# ======================================================================
# Rest-frame windows (A) around the major diagnostic lines. Some are grouped
# (e.g. Halpha + [NII]) because they sit close enough together to share one
# panel usefully. Not every window will fall inside a given galaxy's fitted
# range -- coverage depends on redshift (see plot_line_zooms).
LINE_WINDOWS = {
    "[OII] 3727":         (3700, 3760),
    "Hgamma 4340":        (4315, 4365),
    "Hbeta 4861":         (4836, 4886),
    "[OIII] 4959,5007":   (4934, 5032),
    "[OI] 6300":          (6280, 6320),
    "Halpha + [NII]":     (6528, 6603),
    "[SII] 6716,6731":    (6696, 6751),
}


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_sfh(res, path):
    """
    Star formation history (light fraction vs age) and the age-metallicity
    light-weight map, straight from the fitted template weights.
    """
    if res.weights is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].step(res.age_grid[:, 0], res.weights.sum(axis=1), where="mid",
                 color="tab:blue")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("age (Gyr)")
    axes[0].set_ylabel("light fraction")
    axes[0].set_title("star formation history")

    im = axes[1].pcolormesh(res.age_grid, res.metal_grid, res.weights,
                            shading="auto", cmap="viridis")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("age (Gyr)")
    axes[1].set_ylabel("[M/H]")
    axes[1].set_title("age-metallicity weights")
    fig.colorbar(im, ax=axes[1], label="light fraction")

    fig.suptitle(f"{res.name}   z={res.z:.4f}   "
                 f"log(age)_lw={res.logage_lw:.2f}   chi2={res.chi2:.3f}")
    _save(fig, path)


def plot_full_spectrum(res, path):
    """
    Data, total model (stars + gas) and gas-only component, with residuals.
    Check this first when something looks wrong -- every derived number
    depends on how well this fit matches the data.
    """
    if res.lam_gal is None or res.bestfit is None:
        return

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})

    ax0.plot(res.lam_gal, res.galaxy, color="k", lw=0.6, label="data")
    ax0.plot(res.lam_gal, res.bestfit, color="tab:red", lw=1.0,
             label="total fit (stars + gas)")
    if res.gas_bestfit is not None:
        ax0.plot(res.lam_gal, res.gas_bestfit, color="tab:green", lw=0.8,
                 label="gas only")
    ax0.set_ylabel("flux")
    ax0.set_title(f"{res.name}   z={res.z:.4f}   chi2={res.chi2:.3f}"
                  f"{'  [KIN AT BOUND]' if res.kin_at_bound else ''}")
    ax0.legend(fontsize=8, loc="upper right")

    ax1.plot(res.lam_gal, res.galaxy - res.bestfit, color="k", lw=0.4)
    ax1.axhline(0, color="tab:red", lw=0.8, ls="--")
    ax1.set_xlabel("rest-frame wavelength (A)")
    ax1.set_ylabel("data - fit")
    _save(fig, path)


def plot_line_zooms(res, path):
    """
    Zoomed panels on the major diagnostic emission lines, so line profiles
    can be inspected directly (this is the plot that originally caught a
    template-resolution mismatch). Only windows inside this galaxy's fitted
    rest-frame range are drawn.
    """
    if res.lam_gal is None or res.bestfit is None:
        return

    lam_min, lam_max = res.lam_gal.min(), res.lam_gal.max()
    windows = {label: win for label, win in LINE_WINDOWS.items()
               if win[1] > lam_min and win[0] < lam_max}
    if not windows:
        return

    n = len(windows)
    ncols = min(3, n)
    nrows = -(-n // ncols)  # ceil division
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.6 * nrows),
                             squeeze=False)

    stellar_only = (res.bestfit - res.gas_bestfit
                    if res.gas_bestfit is not None else None)

    for ax, (label, (w0, w1)) in zip(axes.flat, windows.items()):
        sel = (res.lam_gal >= w0) & (res.lam_gal <= w1)
        lam = res.lam_gal[sel]
        ax.step(lam, res.galaxy[sel], where="mid", color="k", lw=1.0, label="data")
        ax.plot(lam, res.bestfit[sel], color="tab:red", lw=1.3, label="total fit")
        if stellar_only is not None:
            ax.plot(lam, stellar_only[sel], color="tab:blue", lw=0.8, ls="--",
                    label="stellar only")
            ax.plot(lam, res.gas_bestfit[sel], color="tab:green", lw=0.8,
                    label="gas only")
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("rest-frame wavelength (A)", fontsize=8)
        ax.tick_params(labelsize=8)

    axes.flat[0].legend(fontsize=7, loc="best")
    for ax in axes.flat[n:]:
        ax.axis("off")

    fig.suptitle(f"{res.name}   z={res.z:.4f}   chi2={res.chi2:.3f}   "
                 f"(line profile zooms)", fontsize=12)
    _save(fig, path)


def plot_spec_and_phot(pp, path):
    """pPXF's own plot of the joint spectrum + photometry fit."""
    pp.plot(gas_clip=1, spec=1, lam_flam=False)
    _save(plt.gcf(), path)


def save_plots(res, pp, plot_dir):
    stem = os.path.join(plot_dir, res.name)
    plot_sfh(res, f"{stem}_sfh.png")
    plot_full_spectrum(res, f"{stem}_spectrum.png")
    plot_line_zooms(res, f"{stem}_lines.png")
    plot_spec_and_phot(pp, f"{stem}_spec_phot.png")


# ======================================================================
# Summary
# ======================================================================
def print_summary(results):
    """One line per galaxy, plus flags for the things worth checking."""
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"{'name':<20}{'z':>7}{'chi2':>8}{'logage_lw':>11}"
          f"{'[M/H]_lw':>10}{'A_V':>7}{'sigma':>8}")
    print("-" * 78)

    for r in results:
        if not r.ok:
            print(f"{r.name:<20}  FAILED: {r.error_msg}")
            continue
        flag = " <- kinematics at bound" if r.kin_at_bound else ""
        print(f"{r.name:<20}{r.z:>7.3f}{r.chi2:>8.3f}{r.logage_lw:>11.2f}"
              f"{r.metal_lw:>10.2f}{r.av_stars:>7.2f}{r.sigma:>8.1f}{flag}")

    ok = [r for r in results if r.ok]
    if not ok:
        return

    print("-" * 78)
    print(f"{len(ok)}/{len(results)} fits succeeded")
    print(f"kinematics at bound: {sum(r.kin_at_bound for r in ok)}/{len(ok)}")

    # Balmer decrement sanity check (only available with --no-tie-balmer).
    # Case B gives Halpha/Hbeta = 2.86 with no dust; dust only raises it.
    ratios = [r.gas_flux["Halpha"] / r.gas_flux["Hbeta"]
              for r in ok
              if r.gas_flux.get("Hbeta", 0) > 0 and "Halpha" in r.gas_flux]
    if ratios:
        print(f"Halpha/Hbeta: median {np.median(ratios):.2f} "
              f"(Case B = 2.86; below that is unphysical)")

    n_mstar = sum(np.isfinite(r.mstar) for r in ok)
    print(f"stellar mass computed: {n_mstar}/{len(ok)} "
          f"(nan for galaxies where SDSS/r falls outside fitted coverage)")


# ======================================================================
# Main
# ======================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Fit DESI spectra with pPXF (stars + gas simultaneously).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("pkl", help="path to the DESI spectra pickle")
    p.add_argument("-n", "--ngal", type=int, default=3,
                   help="number of galaxies to fit (-1 = all)")
    p.add_argument("--specid", nargs="+", default=None,
                   help="fit only these galaxies (TARGETID, or SPARCL specid "
                        "if the pickle has no TARGETID); overrides --ngal")
    p.add_argument("-p", "--photometry-file", default=None,
                   help="Legacy Survey photometry CSV (TARGETID, flux_*, "
                        "flux_ivar_*, mw_transmission_*), fitted jointly "
                        "with the spectrum")

    p.add_argument("--fastspec-file", default=None,
                   help="FastSpecFit catalog CSV with TARGETID and emission-line fluxes")
    p.add_argument("--metal-nmc", type=int, default=1000,
                   help="metallicity Monte Carlo draws per galaxy (0 disables)")
    p.add_argument("--metal-te-snr", type=float, default=3.0,
                   help="minimum [OIII]4363 S/N for candidate direct-Te abundance")

    p.add_argument("--sps", default="galaxev", choices=["galaxev", "emiles", "fsps"],
                   help="SSP library. galaxev (BC03) is recommended: emiles "
                        "has no SSPs younger than 63 Myr, which matters for "
                        "star-forming hosts")
    p.add_argument("--edge-trim", type=float, default=50.0,
                   help="Angstroms trimmed off each edge of a galaxy's own "
                        "de-redshifted range before fitting")
    p.add_argument("--max-age", type=float, default=None,
                   help="maximum SSP age in Gyr (default: age of the Universe at z)")

    p.add_argument("--no-tie-balmer", action="store_true",
                   help="let Balmer line fluxes vary independently instead of "
                        "fixing their ratios to Case B. Gives a free "
                        "Halpha/Hbeta ratio, but disables gas reddening "
                        "(the two are degenerate)")
    p.add_argument("--no-dust", action="store_true", help="do not fit reddening")
    p.add_argument("--no-regul", action="store_true",
                   help="skip the regularization search (regul=0). Faster, "
                        "but the SFH will be noisier than the data justifies")
    p.add_argument("--regul", type=float, default=None,
                   help="use this regularization strength instead of tuning it")
    p.add_argument("--free-stellar-kin", action="store_true",
                   help="fit stellar v and sigma instead of holding them fixed "
                        "at ppxf_desi.STELLAR_KIN_FIXED")
    p.add_argument("--no-mstar", action="store_true",
                   help="skip the stellar mass calculation (see "
                        "rest_frame_L_over_Lsun in ppxf_desi.py)")
    p.add_argument("--nboot", type=int, default=0,
                   help="bootstrap iterations for uncertainties "
                        "(0 = skip; bootstrap dominates the runtime)")

    p.add_argument("-o", "--output", default="ppxf_desi_results.csv",
                   help="output CSV path")
    p.add_argument("--plot-dir", default="ppxf_desi_plots",
                   help="directory for diagnostic plots ('' to disable)")
    p.add_argument("--out-dir", default="ppxf_desi_results",
                   help="directory for per-galaxy .npz results ('' to disable)")

    p.add_argument("-q", "--quiet", action="store_true", help="less verbose output")
    p.add_argument("--debug", action="store_true",
                   help="extra diagnostic printouts (normalisations, "
                        "luminosities, SFH checks)")
    return p.parse_args()


def select_galaxies(galaxies, specids, ngal):
    if specids:
        wanted = set(specids)
        galaxies = [g for g in galaxies if g["name"] in wanted]
        missing = wanted - {g["name"] for g in galaxies}
        if missing:
            print(f"warning: specids not found: {sorted(missing)}")
        return galaxies
    return galaxies if ngal == -1 else galaxies[:ngal]


def load_fastspec(path):
    fastspec = pd.read_csv(path)
    if "TARGETID" not in fastspec.columns:
        sys.exit("error: FastSpecFit catalog must contain TARGETID")
    fastspec["TARGETID"] = fastspec["TARGETID"].astype(str)
    return fastspec.set_index("TARGETID", verify_integrity=True)


def galaxy_metallicity(fastspec, targetid, n_mc, te_snr):
    from metallicity_desi import calculate_metallicity

    found = targetid in fastspec.index
    metal = calculate_metallicity(
        fastspec.loc[targetid] if found else {}, n_mc=n_mc,
        te_snr_threshold=te_snr,
        seed=int(targetid) % (2**32) if targetid.isdigit() else None)
    if not found:
        metal["metal_error"] = "TARGETID not in FastSpecFit catalog"
    return metal


def main():
    args = parse_args()

    if not os.path.exists(args.pkl):
        sys.exit(f"error: no such file: {args.pkl}")

    galaxies = load_desi_pickle(args.pkl)
    print(f"loaded {len(galaxies)} galaxies from {args.pkl}")

    phot_df = pd.read_csv(args.photometry_file) if args.photometry_file else None
    fastspec = load_fastspec(args.fastspec_file) if args.fastspec_file else None

    galaxies = select_galaxies(galaxies, args.specid, args.ngal)
    if not galaxies:
        sys.exit("error: no galaxies selected")

    print(f"fitting {len(galaxies)} galaxies with sps={args.sps}, "
          f"tie_balmer={not args.no_tie_balmer}, nboot={args.nboot}")

    for d in (args.plot_dir, args.out_dir):
        if d:
            os.makedirs(d, exist_ok=True)

    fit_options = dict(
        sps_name=args.sps,
        edge_trim=args.edge_trim,
        tie_balmer=not args.no_tie_balmer,
        fit_dust=not args.no_dust,
        tune_regul=not args.no_regul,
        fixed_regul=args.regul,
        max_age_gyr=args.max_age,
        compute_mstar=not args.no_mstar,
        nboot=args.nboot,
        verbose=not args.quiet,
        debug=args.debug,
    )
    if args.free_stellar_kin:
        fit_options["fix_stellar_kin"] = None

    results, metallicity_results = [], []
    for i, gal in enumerate(galaxies, 1):
        if args.quiet:
            print(f"[{i}/{len(galaxies)}] {gal['name']}", flush=True)

        phot = (build_photometry(phot_df, gal["name"], debug=args.debug)
                if phot_df is not None else None)

        res, pp = fit_galaxy(gal, phot=phot, **fit_options)
        results.append(res)

        if fastspec is not None:
            metallicity_results.append(galaxy_metallicity(
                fastspec, str(gal["name"]), args.metal_nmc, args.metal_te_snr))

        if res.ok:
            if args.plot_dir:
                save_plots(res, pp, args.plot_dir)
            if args.out_dir:
                np.savez(os.path.join(args.out_dir, f"ppxf_results_{res.name}.npz"),
                         results=[res])

    rows, _ = results_to_rows(results)
    if fastspec is not None:
        for row, metal in zip(rows, metallicity_results):
            row.update(metal)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print_summary(results)
    print(f"\nwrote {args.output}")
    if args.plot_dir:
        print(f"wrote plots to {args.plot_dir}/")
        print("  run plot_ppxf_desi_summary.py on the CSV for aggregate "
              "population-level plots")


if __name__ == "__main__":
    main()
