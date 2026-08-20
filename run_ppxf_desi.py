#!/Users/dhvanildesai/miniforge3/envs/hosts/bin/python3
"""
run_ppxf_desi.py
================
Command-line runner for combined stellar + gas pPXF fitting of DESI spectra.

Examples
--------
    # first 3 galaxies, quick look
    python run_ppxf_desi.py examples/desi_spectra.pkl -n 3 -p examples/DESI_starlight_phot.csv

    # all galaxies, with bootstrap errors, writing to a named file
    python run_ppxf_desi.py desi_spectra.pkl --nboot 50 -o results.csv

    # untie the Balmer lines to get a free Halpha/Hbeta ratio
    python run_ppxf_desi.py desi_spectra.pkl -n 3 --no-tie-balmer

    # compare template libraries on the same galaxies
    python run_ppxf_desi.py desi_spectra.pkl -n 3 --sps emiles

    # pick specific galaxies by specid
    python run_ppxf_desi.py desi_spectra.pkl --specid 39628006821467117

Once you have a CSV, plot_ppxf_desi_summary.py builds the aggregate
population-level plots (age/metallicity distributions, LW vs MW, stellar
mass, kinematics) across all fitted galaxies.
"""

inc = ['sparcl_id', 'specid', 'data_release', 'redshift', 'flux',
       'wavelength', 'model', 'ivar', 'mask', 'spectype', 'ra',
       'dec', 'wave_sigma']

from sparcl.client import SparclClient
client = SparclClient()
import argparse
import csv
import os
import sys

import numpy as np
import pandas as pd
import astropy.units as u

# Agg backend so plots save without needing a display (e.g. over ssh).
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ppxf_desi import load_desi_pickle, fit_galaxy, results_to_rows


def parse_args():
    p = argparse.ArgumentParser(
        description="Fit DESI spectra with pPXF (stars + gas simultaneously).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("pkl", help="path to the DESI spectra pickle")
    p.add_argument("-n", "--ngal", type=int, default=3,
                   help="number of galaxies to fit (-1 = all)")
    p.add_argument("--specid", nargs="+", default=None,
                   help="fit only these specids (overrides --ngal)")
    p.add_argument("-p", "--photometry-file", type=str, default=None,
                   help="file with photometry, for calibrating the stellar mass and SFR")

    
    p.add_argument("--sps", default="galaxev",
                   choices=["galaxev", "emiles", "fsps"],
                   help="SSP library. galaxev (BC03) is recommended: emiles "
                        "has no SSPs younger than 63 Myr, which matters for "
                        "star-forming hosts")
    p.add_argument("--edge-trim", type=float, default=50.0,
                   help="Angstroms trimmed off each edge of a galaxy's own "
                        "de-redshifted range before fitting. The fitted "
                        "range itself is NOT fixed -- each galaxy uses its "
                        "own full available rest-frame coverage (DESI's "
                        "fixed observed range maps to a different rest-frame "
                        "window depending on z), minus this edge margin")

    p.add_argument("--no-tie-balmer", action="store_true",
                   help="let Balmer line fluxes vary independently instead of "
                        "fixing their ratios to Case B. Gives a free "
                        "Halpha/Hbeta ratio, but disables gas reddening "
                        "(the two are degenerate)")
    p.add_argument("--no-dust", action="store_true",
                   help="do not fit reddening")
    p.add_argument("--no-regul", action="store_true",
                   help="skip the regularization search (use regul=0). Faster, "
                        "but the recovered SFH will be noisier than the data "
                        "justifies")
    p.add_argument("--no-mstar", action="store_true",
                   help="skip the stellar mass calculation (see "
                        "rest_frame_L_over_Lsun in ppxf_desi.py for the "
                        "flux-calibration assumption this relies on)")
    p.add_argument("--nboot", type=int, default=0,
                   help="bootstrap iterations for age/metallicity errors. "
                        "0 = skip (bootstrap dominates the runtime)")

    p.add_argument("-o", "--output", default="ppxf_desi_results.csv",
                   help="output CSV path")
    p.add_argument("--plot-dir", default="ppxf_desi_plots",
                   help="directory for diagnostic plots ('' to disable)")
    p.add_argument("--out-dir", default="ppxf_desi_results",
                   help="directory for diagnostic plots ('' to disable)")

    p.add_argument("-q", "--quiet", action="store_true",
                   help="less verbose output")
    return p.parse_args()


def plot_sfh(res, path):
    """
    Two-panel summary of the fitted stellar population: the star formation
    history, and the age-metallicity weight map. Both come straight from the
    fitted template weights.
    """
    if res.weights is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ages = res.age_grid[:, 0]
    sfh = res.weights.sum(axis=1)
    axes[0].step(ages, sfh, where="mid", color="tab:blue")
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
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_full_spectrum(res, path):
    """
    The fitted spectrum itself: data, total model (stars + gas), and the
    gas-only component, with a residuals panel underneath. This is the plot
    to check first when something looks wrong -- the population/kinematics
    numbers are all derived from how well this line matches the data.
    """
    if res.lam_gal is None or res.bestfit is None:
        return

    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(11, 6), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]})

    ax0.plot(res.lam_gal, res.galaxy, color="k", lw=0.6, label="data")
    ax0.plot(res.lam_gal, res.bestfit, color="tab:red", lw=1.0,
            label="total fit (stars + gas)")
    if res.gas_bestfit is not None:
        ax0.plot(res.lam_gal, res.gas_bestfit, color="tab:green", lw=0.8,
                label="gas only")
    ax0.set_ylabel("normalised flux")
    ax0.set_title(f"{res.name}   z={res.z:.4f}   chi2={res.chi2:.3f}"
                  f"{'  [KIN AT BOUND]' if res.kin_at_bound else ''}")
    ax0.legend(fontsize=8, loc="upper right")

    resid = res.galaxy - res.bestfit
    ax1.plot(res.lam_gal, resid, color="k", lw=0.4)
    ax1.axhline(0, color="tab:red", lw=0.8, ls="--")
    ax1.set_xlabel("rest-frame wavelength (A)")
    ax1.set_ylabel("data - fit")

    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)

def plot_spec_and_phot(pp,path):

    pp.plot(gas_clip=1, spec=1, lam_flam=False)
    plt.savefig(path, dpi=120, bbox_inches="tight")


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


def plot_line_zooms(res, path, half_width=None):
    """
    Zoomed panels on the major diagnostic emission lines, so the line
    profile shape can actually be inspected (rather than inferred from the
    full-spectrum plot, where a narrow line is just a few pixels wide).
    This is the plot that originally caught a template-resolution mismatch
    earlier in this project -- worth checking whenever fit quality looks off
    for a specific galaxy.

    Only windows that fall (at least partly) inside this galaxy's fitted
    rest-frame range are drawn; coverage varies with redshift.
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

    resid = res.galaxy - res.bestfit
    stellar_only = (res.bestfit - res.gas_bestfit
                    if res.gas_bestfit is not None else None)

    for ax, (label, (w0, w1)) in zip(axes.flat, windows.items()):
        mask = (res.lam_gal >= w0) & (res.lam_gal <= w1)
        ax.step(res.lam_gal[mask], res.galaxy[mask], where="mid",
               color="k", lw=1.0, label="data")
        ax.plot(res.lam_gal[mask], res.bestfit[mask], color="tab:red",
               lw=1.3, label="total fit")
        if stellar_only is not None:
            ax.plot(res.lam_gal[mask], stellar_only[mask], color="tab:blue",
                   lw=0.8, ls="--", label="stellar only")
        if res.gas_bestfit is not None:
            ax.plot(res.lam_gal[mask], res.gas_bestfit[mask],
                   color="tab:green", lw=0.8, label="gas only")
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("rest-frame wavelength (A)", fontsize=8)
        ax.tick_params(labelsize=8)

    # legend once, on the first panel, to avoid clutter
    axes.flat[0].legend(fontsize=7, loc="best")
    # turn off any unused grid cells
    for ax in axes.flat[n:]:
        ax.axis("off")

    fig.suptitle(f"{res.name}   z={res.z:.4f}   chi2={res.chi2:.3f}   "
                 f"(line profile zooms)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def print_summary(results):
    """One line per galaxy, plus flags for the things worth checking."""
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    header = (f"{'specid':<20}{'z':>7}{'chi2':>8}{'logage_lw':>11}"
              f"{'[M/H]_lw':>10}{'A_V':>7}{'sigma':>8}")
    print(header)
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
    n_bound = sum(r.kin_at_bound for r in ok)
    print(f"{len(ok)}/{len(results)} fits succeeded")
    print(f"kinematics at bound: {n_bound}/{len(ok)} "
          f"(population fit is unaffected by this)")

    # The Balmer decrement is a useful sanity check when the lines are untied.
    # Case B recombination gives Halpha/Hbeta = 2.86 with no dust; dust only
    # raises it. Values below 2.86 are unphysical and indicate a problem
    # somewhere.
    ratios = [r.gas_flux["Halpha"] / r.gas_flux["Hbeta"]
              for r in ok
              if r.gas_flux.get("Hbeta", 0) > 0 and "Halpha" in r.gas_flux]
    if ratios:
        print(f"Halpha/Hbeta: median {np.median(ratios):.2f} "
              f"(Case B = 2.86; below that is unphysical)")

    n_mstar = sum(np.isfinite(r.mstar) for r in ok)
    print(f"stellar mass computed: {n_mstar}/{len(ok)} "
          f"(nan for galaxies where SDSS/r falls outside fitted coverage)")


def main():
    args = parse_args()

    if not os.path.exists(args.pkl):
        sys.exit(f"error: no such file: {args.pkl}")

    galaxies = load_desi_pickle(args.pkl)
    if args.photometry_file:
        phot = pd.read_csv(args.photometry_file)
    print(f"loaded {len(galaxies)} galaxies from {args.pkl}")

    if args.specid:
        wanted = set(args.specid)
        galaxies = [g for g in galaxies if g["name"] in wanted]
        missing = wanted - {g["name"] for g in galaxies}
        if missing:
            print(f"warning: specids not found: {sorted(missing)}")
    elif args.ngal != -1:
        galaxies = galaxies[:args.ngal]

    if not galaxies:
        sys.exit("error: no galaxies selected")

    print(f"fitting {len(galaxies)} galaxies with sps={args.sps}, "
          f"tie_balmer={not args.no_tie_balmer}, nboot={args.nboot}")

    if args.plot_dir:
        os.makedirs(args.plot_dir, exist_ok=True)
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

    results = []
    for i, gal in enumerate(galaxies, 1):
        ret = client.retrieve_by_specid([int(gal['name'])],include=inc,fmt='pandas')
        gal['wave_fwhm'] = ret.wave_sigma[0]*2.3548
        if args.photometry_file and np.int64(gal['name']) not in phot['TARGETID'].values:
            raise RuntimeError(f"photometry not found for gal {gal['name']}")
        elif args.photometry_file:
            p = phot[phot['TARGETID'] == int(gal['name'])]
            nanomaggy = u.def_unit('nanomaggy', 3.631e-6 * u.Jy)
            
            ### get the photometry ###
            phot_flam = []
            phot_flamerr = []
            phot_lam = np.array([4863,6463,9201])
            for flux,flux_ivar,lam in zip(
                    [p['FLUX_G'].values[0],p['FLUX_R'].values[0],p['FLUX_Z'].values[0]],
                    [p['FLUX_IVAR_G'].values[0],p['FLUX_IVAR_R'].values[0],p['FLUX_IVAR_Z'].values[0]],
                    phot_lam
            ):

                phot_flam.append((flux*nanomaggy).to(
                    u.erg / u.s / u.cm**2 / u.AA,
                    equivalencies=u.spectral_density(lam*u.AA)).value*1e17)
                fluxerr = np.sqrt(1/flux_ivar + (0.01*flux)**2.)
                phot_flamerr.append(fluxerr*(phot_flam[-1]/flux))
            galaxy_phot = {'phot_galaxy':phot_flam,
                           'noise':phot_flamerr,
                           'lam':phot_lam}
        else:
            galaxy_phot=None

        if args.quiet:
            print(f"[{i}/{len(galaxies)}] {gal['name']}", flush=True)
        res, pp = fit_galaxy(
            gal,
            sps_name=args.sps,
            edge_trim=args.edge_trim,
            tie_balmer=not args.no_tie_balmer,
            fit_dust=not args.no_dust,
            tune_regul=not args.no_regul,
            compute_mstar=not args.no_mstar,
            phot=galaxy_phot,
            nboot=args.nboot,
            verbose=not args.quiet
        )
        results.append(res)

        if args.plot_dir and res.ok:
            plot_sfh(res, os.path.join(args.plot_dir, f"{res.name}_sfh.png"))
            plot_full_spectrum(
                res, os.path.join(args.plot_dir, f"{res.name}_spectrum.png"))
            plot_line_zooms(
                res, os.path.join(args.plot_dir, f"{res.name}_lines.png"))
            plot_spec_and_phot(
                pp, os.path.join(args.plot_dir, f"{res.name}_spec_phot.png"))

        if args.out_dir and res.ok:
            np.savez(f'{args.out_dir}/ppxf_results_{res.name}.npz',results=results)

    rows, _ = results_to_rows(results)
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
