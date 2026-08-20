#!/usr/bin/env python3
"""
plot_ppxf_desi_summary.py
==========================
Aggregate, population-level plots across all galaxies in a
run_ppxf_desi.py results CSV: age and metallicity distributions
(light- and mass-weighted), light- vs mass-weighted comparisons,
stellar mass, mass-to-light ratio, and kinematics.

Usage
-----
    python plot_ppxf_desi_summary.py ppxf_desi_results.csv
    python plot_ppxf_desi_summary.py ppxf_desi_results.csv -o summary.png
"""

import argparse
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv", help="results CSV from run_ppxf_desi.py")
    p.add_argument("-o", "--output", default="ppxf_desi_summary.png",
                   help="output plot path")
    return p.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.csv)
    n_total = len(df)
    df = df[df["ok"]].copy()
    print(f"{len(df)}/{n_total} fits ok")
    if df.empty:
        sys.exit("no successful fits to plot")

    fig, axes = plt.subplots(3, 3, figsize=(15, 13))

    # 1. Age distributions, light- and mass-weighted
    ax = axes[0, 0]
    bins = np.linspace(min(df.logage_lw.min(), df.logage_mw.min()) - 0.1,
                       max(df.logage_lw.max(), df.logage_mw.max()) + 0.1, 18)
    ax.hist(df.logage_lw, bins=bins, alpha=0.6, color="tab:blue",
           label="light-weighted")
    ax.hist(df.logage_mw, bins=bins, alpha=0.6, color="tab:red",
           label="mass-weighted")
    ax.set_xlabel("log10(age / yr)")
    ax.set_ylabel("N galaxies")
    ax.set_title("Age distributions")
    ax.legend(fontsize=8)

    # 2. Metallicity distributions, light- and mass-weighted
    ax = axes[0, 1]
    bins = np.linspace(min(df.metal_lw.min(), df.metal_mw.min()) - 0.1,
                       max(df.metal_lw.max(), df.metal_mw.max()) + 0.1, 18)
    ax.hist(df.metal_lw, bins=bins, alpha=0.6, color="tab:blue",
           label="light-weighted")
    ax.hist(df.metal_mw, bins=bins, alpha=0.6, color="tab:red",
           label="mass-weighted")
    ax.set_xlabel("[M/H]")
    ax.set_ylabel("N galaxies")
    ax.set_title("Metallicity distributions")
    ax.legend(fontsize=8)

    # 3. LW vs MW age
    ax = axes[0, 2]
    ax.scatter(df.logage_lw, df.logage_mw, c="tab:blue", s=30, alpha=0.8)
    lims = [min(df.logage_lw.min(), df.logage_mw.min()) - 0.1,
           max(df.logage_lw.max(), df.logage_mw.max()) + 0.1]
    ax.plot(lims, lims, "k--", lw=1, alpha=0.5)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("LW log age")
    ax.set_ylabel("MW log age")
    ax.set_title("LW vs MW age")

    # 4. LW vs MW metallicity
    ax = axes[1, 0]
    ax.scatter(df.metal_lw, df.metal_mw, c="tab:blue", s=30, alpha=0.8)
    lims = [min(df.metal_lw.min(), df.metal_mw.min()) - 0.1,
           max(df.metal_lw.max(), df.metal_mw.max()) + 0.1]
    ax.plot(lims, lims, "k--", lw=1, alpha=0.5)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("LW [M/H]")
    ax.set_ylabel("MW [M/H]")
    ax.set_title("LW vs MW metallicity")

    # 5. Stellar mass
    ax = axes[1, 1]
    mstar_ok = df.mstar[np.isfinite(df.mstar) & (df.mstar > 0)]
    if len(mstar_ok):
        ax.hist(np.log10(mstar_ok), bins=12, color="tab:cyan", alpha=0.8)
    ax.set_xlabel("log10(M* / Msun)")
    ax.set_ylabel("N galaxies")
    ax.set_title(f"Stellar mass ({len(mstar_ok)}/{len(df)} finite)")

    # 6. Mass-to-light
    ax = axes[1, 2]
    ax.hist(df.ml, bins=12, color="tab:brown", alpha=0.8)
    ax.set_xlabel("M*/L (SDSS/r, solar units)")
    ax.set_ylabel("N galaxies")
    ax.set_title("Mass-to-light ratio")

    # 7. Kinematics: v vs sigma, flagging galaxies at their search bound
    ax = axes[2, 0]
    at_bound = df.kin_at_bound.astype(bool)
    ax.scatter(df.v[~at_bound], df.sigma[~at_bound], c="tab:green", s=35,
              label="kin OK", alpha=0.8)
    ax.scatter(df.v[at_bound], df.sigma[at_bound], c="tab:orange",
              marker="s", s=35, label="kin_at_bound", alpha=0.8)
    ax.set_xlabel("v (km/s)")
    ax.set_ylabel("sigma (km/s)")
    ax.set_title(f"Kinematics ({at_bound.sum()}/{len(df)} at bound)")
    ax.legend(fontsize=8)

    # unused panels
    axes[2, 1].axis("off")
    axes[2, 2].axis("off")

    fig.suptitle(f"{args.csv}  --  {len(df)} galaxies", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(args.output, dpi=140, bbox_inches="tight")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
