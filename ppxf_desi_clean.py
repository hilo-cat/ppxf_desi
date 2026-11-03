"""
ppxf_desi.py
============
Combined stellar-continuum + gas emission-line fitting of DESI galaxy spectra
with pPXF, optionally jointly with DECam g/r/z photometry.

Written for Python 3.13 / ppxf 9.4.8.

Overview
--------
One `ppxf` call fits stars and gas SIMULTANEOUSLY, with separate kinematic
components:

    component 0 = stellar templates       (SSP library, e.g. GALAXEV/BC03)
    component 1 = Balmer emission lines
    component 2 = forbidden emission lines

This follows pPXF's own official combined-fit example
(ppxf_example_population_bootstrap.ipynb).

Pipeline (see fit_galaxy)
-------------------------
1. Milky Way extinction correction       correct_mw_extinction
2. Degrade galaxy to template resolution degrade_to_template_res
3. De-redshift, trim, log-rebin          prepare_galaxy
4. Stellar + gas templates               prepare_templates,
                                         prepare_gas_templates,
                                         combine_templates
5. Scale spectrum to photometry          prepare_photometry   (optional)
6. Regularization                        tune_regularization
7. Final fit and derived quantities      derive_quantities, get_sfr
8. Wild bootstrap                        _bootstrap           (optional)

Key choices (and why)
---------------------
* sps_name="galaxev" (BC03) by default, not E-MILES. E-MILES contains no
  SSPs younger than 63 Myr and is explicitly flagged in pPXF's example
  notebook as unsuitable for highly star-forming galaxies. Testing across 8
  galaxies spanning z=0.05-0.49 showed GALAXEV giving systematically younger
  light-weighted ages, better chi2, and more stable kinematics.

* mdegree=-1 and degree=-1: NO polynomials. A multiplicative polynomial
  competes directly with both the regularization and the dust parameter for
  the same broad continuum-shape information. With mdegree=10 the
  regularization signal was almost entirely absorbed (Delta_chi2 flat out to
  regul=1e7); mdegree=-1 matches the official notebook recipe and let more
  galaxies reach the statistical regularization target. Dust/continuum shape
  is handled instead by `reddening` (stars) and `gas_reddening` (gas).

* Resolution matching by DEGRADING THE GALAXY (see degrade_to_template_res).
  DESI's true instrumental resolution (FWHM ~1.0-1.5 A) is SHARPER than the
  SSP templates' native resolution (~2.5 A floor across the fitted range).
  You cannot sharpen a template by convolution, so the honest fix is to
  broaden the galaxy down to the template resolution. After this, fwhm_gal
  passed to the template builders is the template's own native FWHM, i.e. no
  further template convolution.

* Stellar kinematics are FIXED by default at STELLAR_KIN_FIXED during the
  regularization search and the final fit (pass fix_stellar_kin=None to
  fit_galaxy to free them). With them fixed, `kin_at_bound` is not
  informative.

Known open issues
-----------------
1. **A_V** is probably unreliable as of now, need to figure out more.
2. **Halpha/Hbeta** ratios and dust approximation from that needs to be
   confirmed too.
3. **Stellar kinematics** (v, sigma) often land right on the search
   bounds for these faint galaxies when left free -- flagged per galaxy via
   `kin_at_bound` in the CSV. Doesn't affect the population/mass fit,
   just the kinematics themselves.
"""

from __future__ import annotations

import hashlib
import os
import pickle
import time
import traceback
import urllib.request
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from importlib import resources

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from dustmaps.sfd import SFDQuery
from extinction import fitzpatrick99
from sedpy.smoothing import smoothspec

import ppxf.ppxf_util as util
import ppxf.sps_util as sps_util
from ppxf.ppxf import ppxf


__all__ = [
    "FitResult",
    "load_desi_pickle",
    "correct_mw_extinction",
    "degrade_to_template_res",
    "prepare_galaxy",
    "prepare_templates",
    "prepare_gas_templates",
    "combine_templates",
    "prepare_photometry",
    "tune_regularization",
    "rest_frame_L_over_Lsun",
    "derive_quantities",
    "get_sfr",
    "fit_galaxy",
    "results_to_rows",
]


# ======================================================================
# Constants
# ======================================================================
_COSMO = FlatLambdaCDM(H0=70, Om0=0.3)

SIGMA_TO_FWHM = 2.3548
FWHM_TO_SIGMA = 1.0 / SIGMA_TO_FWHM

# Kinematic search bounds (km/s).
V_BOUNDS = (-1000.0, 1000.0)
SIGMA_BOUNDS_STARS = (10.0, 500.0)
SIGMA_BOUNDS_GAS = (5.0, 500.0)

# (v, sigma) at which the stellar kinematics are held fixed by default.
STELLAR_KIN_FIXED = (0.0, 100.0)

DEFAULT_REGUL_GRID = (0.1, 0.3, 1, 3, 10, 30, 100, 300,
                      1000, 3000, 10000, 30000)

# Resolution matching.
DEFAULT_DATA_FWHM = 1.53   # A, used if no per-pixel LSF is supplied
MIN_KERNEL_FWHM = 0.3      # A, ~1/3 pixel: effectively no smoothing

# Template coverage: galaxy blue edge minus a margin for velocity shifts, up
# to the red edge of the DECam z-band transmission (plus the same margin) so
# the templates can predict the photometry.
TEMPLATE_LAM_MARGIN = 500.0
DECAM_Z_RED_EDGE = 10200.0

PHOT_BANDS = ("DECam/DECam_g", "DECam/DECam_r", "DECam/DECam_z")
SPEC_SCALE_BANDS = PHOT_BANDS[:2]   # bands used to scale spectrum -> phot

# Stellar mass.
MASS_BAND = "SDSS/r"
M_SUN_AB_R = 4.64             # SDSS r solar abs. AB mag, Blanton & Roweis (2007)
CALZETTI_AR_OVER_AV = 0.88    # Calzetti A_r / A_V

SF11_RECALIBRATION = 0.86     # Schlafly & Finkbeiner (2011) SFD rescaling

SPS_DATA_URL = "https://raw.githubusercontent.com/micappe/ppxf_data/main/"
SPS_FILENAMES = {
    "emiles":  "spectra_emiles_9.0.npz",
    "galaxev": "spectra_galaxev_9.0.npz",
    "fsps":    "spectra_fsps_9.0.npz",
    "xsl":     "spectra_xsl_9.0.npz",
}

# Vacuum rest wavelengths (A) of the strong lines; DESI is in vacuum.
_STRONG_LINES = [3727.09, 3729.88, 3869.86, 3968.59, 4102.89, 4341.69,
                 4364.44, 4687.02, 4862.68, 4960.30, 5008.24, 5877.25,
                 6302.05, 6549.86, 6564.61, 6585.27, 6718.29, 6732.67]

# Quantities that get 16/50/84 percentiles from the bootstrap. Each name X
# must exist on FitResult together with X_16 and X_84.
BOOT_QUANTITIES = (
    "logage_lw", "logage_mw", "metal_lw", "metal_mw", "mstar",
    "av_stars", "av_gas", "v", "sigma", "v_gas", "sigma_gas",
    "log_sfr_10myr", "log_sfr_30myr", "log_sfr_100myr",
)


def _as_float(x):
    """float(x), or nan for None (e.g. pp.reddening when dust is not fitted)."""
    return np.nan if x is None else float(x)


# ======================================================================
# Results container
# ======================================================================
@dataclass
class FitResult:
    """Everything we keep from one galaxy's fit."""

    name: str
    z: float
    ok: bool = True
    error_msg: str = ""

    chi2: float = np.nan
    regul_used: float = np.nan
    chi2_cont: float = np.nan
    spec_scale: float = np.nan
    max_age_gyr: float = np.nan

    # stellar population
    logage_lw: float = np.nan     # light-weighted log10(age/yr)
    logage_mw: float = np.nan     # mass-weighted
    metal_lw: float = np.nan      # light-weighted [M/H]
    metal_mw: float = np.nan      # mass-weighted
    ml: float = np.nan            # M/L in SDSS/r, solar units
    mstar: float = np.nan         # surviving stellar mass, Msun
    mstar_formed: float = np.nan  # formed stellar mass, Msun

    # dust
    av_stars: float = np.nan
    av_gas: float = np.nan
    ebv_mw: float = np.nan

    # stellar kinematics
    v: float = np.nan
    sigma: float = np.nan
    kin_at_bound: bool = False

    # gas kinematics (Balmer component)
    v_gas: float = np.nan
    sigma_gas: float = np.nan

    # emission-line fluxes, keyed by line name
    gas_flux: dict = field(default_factory=dict)
    gas_flux_err: dict = field(default_factory=dict)

    # star-formation rate
    log_sfh: np.ndarray | None = None
    log_sfr_10myr: float = np.nan
    log_sfr_30myr: float = np.nan
    log_sfr_100myr: float = np.nan

    # bootstrap percentiles (nan unless nboot > 0); see BOOT_QUANTITIES
    logage_lw_16: float = np.nan
    logage_mw_16: float = np.nan
    metal_lw_16: float = np.nan
    metal_mw_16: float = np.nan
    mstar_16: float = np.nan
    av_stars_16: float = np.nan
    av_gas_16: float = np.nan
    v_16: float = np.nan
    sigma_16: float = np.nan
    v_gas_16: float = np.nan
    sigma_gas_16: float = np.nan
    log_sfr_10myr_16: float = np.nan
    log_sfr_30myr_16: float = np.nan
    log_sfr_100myr_16: float = np.nan

    logage_lw_84: float = np.nan
    logage_mw_84: float = np.nan
    metal_lw_84: float = np.nan
    metal_mw_84: float = np.nan
    mstar_84: float = np.nan
    av_stars_84: float = np.nan
    av_gas_84: float = np.nan
    v_84: float = np.nan
    sigma_84: float = np.nan
    v_gas_84: float = np.nan
    sigma_gas_84: float = np.nan
    log_sfr_10myr_84: float = np.nan
    log_sfr_30myr_84: float = np.nan
    log_sfr_100myr_84: float = np.nan

    # arrays kept in memory for plotting, not written to CSV
    weights: np.ndarray | None = None       # light weights (n_age, n_metal)
    mass_weights: np.ndarray | None = None
    age_grid: np.ndarray | None = None
    metal_grid: np.ndarray | None = None
    lam_gal: np.ndarray | None = None       # rest-frame wavelength, as fitted
    galaxy: np.ndarray | None = None        # the data, as fitted
    bestfit: np.ndarray | None = None       # total model (stars + gas)
    gas_bestfit: np.ndarray | None = None   # gas-only component of the model


# Fields of FitResult that are arrays/dicts and so are not CSV columns.
_ARRAY_FIELDS = ("weights", "mass_weights", "age_grid", "metal_grid",
                 "lam_gal", "galaxy", "bestfit", "gas_bestfit",
                 "gas_flux", "gas_flux_err", "log_sfh")


# ======================================================================
# Data loading and Milky Way extinction
# ======================================================================
def load_desi_pickle(path):
    """
    Read the DESI spectra pickle into a list of plain dicts.

    Bad pixels: DESI flags a pixel bad if ivar <= 0 OR mask != 0. Treating
    any nonzero mask as bad is the conservative choice; bad pixels get a
    noise of 1e10 so they carry no weight.

    `name` is the TARGETID when the pickle has an `input_TARGETID` column,
    otherwise the SPARCL specid.
    """
    with open(path, "rb") as f:
        df = pickle.load(f)

    has_tid = "input_TARGETID" in df.columns
    tid_col = df["input_TARGETID"].astype("int64") if has_tid else None

    galaxies = []
    for idx, row in df.iterrows():
        ivar = np.asarray(row["ivar"], dtype=float)
        mask = np.asarray(row["mask"], dtype=int)

        noise = np.full_like(ivar, 1e10)
        good = (ivar > 0) & (mask == 0)
        noise[good] = 1.0 / np.sqrt(ivar[good])

        galaxies.append(dict(
            name=str(int(tid_col[idx])) if has_tid else str(row["specid"]),
            sparcl_specid=str(row["specid"]),
            z=float(row["redshift"]),
            ra=float(row["ra"]),
            dec=float(row["dec"]),
            lam=np.asarray(row["wavelength"], dtype=float),
            flux=np.asarray(row["flux"], dtype=float),
            noise=noise,
            wave_fwhm=np.asarray(row["wave_sigma"], dtype=float) * SIGMA_TO_FWHM,
        ))
    return galaxies


@lru_cache(maxsize=1)
def _sfd_query():
    """The SFD map is loaded once, on first use, rather than at import."""
    return SFDQuery()


def correct_mw_extinction(lam, flux, noise, ra, dec, rv=3.1):
    """
    Correct observed flux and uncertainty for Milky Way foreground extinction.

    E(B-V) comes from the SFD dust map with the Schlafly & Finkbeiner (2011)
    recalibration; A_lambda from Fitzpatrick (1999), evaluated at
    observed-frame wavelengths.

    Parameters
    ----------
    lam : observed-frame wavelength, Angstrom.
    flux, noise : flux density and its 1-sigma uncertainty.
    ra, dec : coordinates in degrees.
    rv : R_V for the Fitzpatrick (1999) law.

    Returns
    -------
    flux_corr, noise_corr, ebv
    """
    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    ebv = SF11_RECALIBRATION * float(_sfd_query()(coord))
    a_lambda = fitzpatrick99(np.asarray(lam, dtype=float), rv * ebv, rv)
    correction = 10 ** (0.4 * a_lambda)
    return np.asarray(flux) * correction, np.asarray(noise) * correction, ebv


# ======================================================================
# Template files and resolution matching
# ======================================================================
def _ensure_template_file(sps_name, ppxf_dir=None):
    """Local path to an SPS template file, downloading it on first use."""
    if ppxf_dir is None:
        ppxf_dir = os.path.dirname(os.path.realpath(sps_util.__file__))
    models_dir = os.path.join(ppxf_dir, "sps_models")
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, SPS_FILENAMES[sps_name])

    if not os.path.exists(path):
        url = SPS_DATA_URL + SPS_FILENAMES[sps_name]
        print(f"    {sps_name} templates not cached -- downloading from "
              f"{url} (one-time)...")
        urllib.request.urlretrieve(url, path)
    return path


def get_template_native_fwhm(sps_name, ppxf_dir=None):
    """
    Native instrumental FWHM(lambda) stored inside an SPS template file.
    Returns (lam, fwhm), both in Angstroms, template rest frame.
    """
    data = np.load(_ensure_template_file(sps_name, ppxf_dir))
    return np.asarray(data["lam"], dtype=float), np.asarray(data["fwhm"], dtype=float)


def degrade_to_template_res(lam_obs, flux, noise, z, sps_name,
                            data_fwhm=None, verbose=False):
    """
    Broaden an observed-frame galaxy spectrum so its spectral resolution
    matches the SSP templates.

    Why: DESI resolves finer detail (FWHM ~1.0-1.5 A) than the SSP libraries
    do (~2.5 A floor over 3500-8000 A rest frame). Convolution can only
    broaden, never sharpen, so the templates cannot be brought to DESI's
    resolution -- the galaxy must instead be brought down to the templates'.

    The kernel follows from Gaussians adding in quadrature:

        fwhm_kernel = sqrt(fwhm_target^2 - fwhm_data^2)

    `data_fwhm` may be a scalar or per-pixel array (observed frame, A);
    non-finite or non-positive entries are replaced by the median of the
    valid ones.

    Returns (flux_deg, noise_deg, fwhm_target_rest).
    """
    if data_fwhm is None:
        data_fwhm = DEFAULT_DATA_FWHM
    data_fwhm = np.broadcast_to(np.asarray(data_fwhm, dtype=float), lam_obs.shape)
    valid = np.isfinite(data_fwhm) & (data_fwhm > 0)
    data_fwhm = np.where(valid, data_fwhm, np.nanmedian(data_fwhm[valid]))

    lam_temp, fwhm_temp = get_template_native_fwhm(sps_name)
    fwhm_target_rest = np.interp(lam_obs / (1.0 + z), lam_temp, fwhm_temp)
    fwhm_target_obs = fwhm_target_rest * (1.0 + z)

    diff_sq = fwhm_target_obs ** 2 - data_fwhm ** 2
    n_zero = int(np.sum(diff_sq <= 0))
    if n_zero:
        print(f"    WARNING: {n_zero} pixels where data FWHM >= template FWHM "
              f"(max data FWHM {data_fwhm.max():.2f} A, "
              f"min target {fwhm_target_obs.min():.2f} A)")
    fwhm_kernel = np.maximum(np.sqrt(np.clip(diff_sq, 0.0, None)), MIN_KERNEL_FWHM)

    flux_deg = smoothspec(lam_obs, flux, resolution=fwhm_kernel * FWHM_TO_SIGMA,
                          smoothtype="lsf")

    # Smoothing averages over ~2*sqrt(pi)*sigma_pix pixels, which reduces the
    # per-pixel noise (but introduces correlations between pixels).
    sigma_pix = fwhm_kernel * FWHM_TO_SIGMA / np.median(np.diff(lam_obs))
    noise_factor = np.minimum(
        1.0, 1.0 / np.sqrt(2.0 * np.sqrt(np.pi) * np.maximum(sigma_pix, 1e-3)))
    noise_deg = noise * noise_factor

    if verbose:
        print(f"    resolution: degraded galaxy by sigma={np.median(sigma_pix):.2f} "
              f"pix ({np.median(fwhm_kernel):.2f} A) to reach template FWHM "
              f"~{np.median(fwhm_target_rest):.2f} A rest frame")

    return flux_deg, noise_deg, fwhm_target_rest


# ======================================================================
# Galaxy and template preparation
# ======================================================================
def prepare_galaxy(lam_obs, flux, noise, z, edge_trim=50.0, verbose=False):
    """
    De-redshift, trim, and log-rebin.

    `edge_trim` is the ONLY fixed quantity: it trims that many Angstroms off
    each edge of whatever rest-frame range this galaxy's spectrum covers.
    DESI's observed coverage (~3600-9824 A) maps to a different rest-frame
    range for every z, so the fitted window is wider for low-z targets and
    narrower for high-z ones.

    The spectrum is NOT renormalised (norm=1); fluxes stay in the DESI units
    of 1e-17 erg/s/cm^2/A, which rest_frame_L_over_Lsun relies on.

    Returns a dict; `wave_range` is the (min, max) rest-frame range used,
    needed downstream to size the templates.
    """
    lam_rest_full = lam_obs / (1.0 + z)
    wave_range = (float(lam_rest_full.min() + edge_trim),
                  float(lam_rest_full.max() - edge_trim))

    keep = (lam_rest_full > wave_range[0]) & (lam_rest_full < wave_range[1])
    lam_rest, flux, noise = lam_rest_full[keep], flux[keep], noise[keep]
    if verbose:
        print(f"    full de-redshifted range: "
              f"{lam_rest_full.min():.0f}-{lam_rest_full.max():.0f} A")
        print(f"    trimmed {edge_trim:.0f} A off each edge -> fitting "
              f"{wave_range[0]:.0f}-{wave_range[1]:.0f} A: {keep.sum()} pixels")

    lam_range = np.array([lam_rest.min(), lam_rest.max()])
    galaxy, ln_lam, velscale = util.log_rebin(lam_range, flux)
    noise_rebin, _, _ = util.log_rebin(lam_range, noise)

    bad = ~np.isfinite(noise_rebin) | (noise_rebin <= 0)
    if bad.any():
        noise_rebin[bad] = 1e10
        if verbose:
            print(f"    {bad.sum()} pixels had bad noise -> downweighted")

    if verbose:
        print(f"    log-rebinned to {galaxy.size} pixels, "
              f"velscale={float(velscale):.2f} km/s/pix")

    return dict(galaxy=galaxy, noise=noise_rebin, lam_gal=np.exp(ln_lam),
                ln_lam_gal=ln_lam, velscale=float(velscale), norm=1.0,
                wave_range=wave_range)


def prepare_templates(velscale, fwhm_gal, z, sps_name="galaxev",
                      lam_range_temp=None, norm_range=(5000.0, 6000.0),
                      max_age_gyr=None, verbose=False, debug=False):
    """
    Build the SSP template library.

    Two libraries are built from the same file:
      * `sps` with norm_range=None -> the templates actually fitted, so the
        pPXF weights are (up to a scalar) MASS weights;
      * a light-normalised copy over `norm_range`, whose `.flux` is stored as
        `sps.lw_flux_conv` and used to convert to LIGHT weights later.

    Ages are capped at the age of the Universe at z (and at `max_age_gyr`).

    `fwhm_gal` is the galaxy's resolution AFTER degradation, i.e. the
    templates' own native resolution, so in practice no convolution occurs.
    """
    filename = _ensure_template_file(sps_name)

    age_max = _COSMO.age(z).value
    if max_age_gyr is not None:
        age_max = min(age_max, max_age_gyr)

    common = dict(lam_range=lam_range_temp, age_range=[0, age_max])
    sps = sps_util.sps_lib(filename, velscale, fwhm_gal, norm_range=None, **common)
    sps_lw = sps_util.sps_lib(filename, velscale, fwhm_gal,
                              norm_range=norm_range, **common)
    sps.lw_flux_conv = sps_lw.flux

    if debug:
        # Youngest and oldest SSP at the first metallicity.
        for i in (0, -1):
            ratio = np.median(sps.templates[:, i, 0] / sps_lw.templates[:, i, 0])
            print(f"    age index {i}: mass/light template ratio = {ratio:.6e}, "
                  f"sps.flux = {sps.flux[i, 0]:.6e}, "
                  f"sps_lw.flux = {sps_lw.flux[i, 0]:.6e}")

    # NOTE: this extra scalar rescaling is not reflected in sps.flux.
    sps.templates /= np.median(sps.templates)

    if debug:
        print("\n=== SPS NORMALIZATION CHECK ===")
        for label, arr in (("sps.flux", sps.flux),
                           ("sps.mass_no_gas_grid", sps.mass_no_gas_grid),
                           ("sps.lw_flux_conv", sps.lw_flux_conv)):
            print(f"{label}: shape {arr.shape}, "
                  f"min/max {np.nanmin(arr):.4g} / {np.nanmax(arr):.4g}")

    if verbose:
        print(f"    templates: {sps_name}, shape {sps.templates.shape} "
              f"(n_wave, n_age, n_metal)")
    return sps


def prepare_gas_templates(sps, lam_range_gal, fwhm_gal, tie_balmer=True,
                          limit_doublets=True, verbose=False):
    """
    Gaussian emission-line templates on the stellar templates' log-lambda grid.

    tie_balmer=True fixes the Balmer line ratios to Case B recombination.
    That is required for `gas_reddening` to be meaningful, but it means the
    individual Balmer fluxes are not independent measurements. Use
    tie_balmer=False for a free Halpha/Hbeta ratio -- but then do not also
    fit gas_reddening, as the two are degenerate.
    """
    gas_templates, gas_names, gas_wave = util.emission_lines(
        sps.ln_lam_temp, lam_range_gal, fwhm_gal,
        tie_balmer=tie_balmer, limit_doublets=limit_doublets, vacuum=True)

    if verbose:
        print(f"    gas templates: {len(gas_names)} lines "
              f"(tie_balmer={tie_balmer})")
    return gas_templates, list(gas_names), gas_wave


def combine_templates(sps, gas_templates, gas_names):
    """
    Stack stellar and gas templates into one matrix and build the
    component/gas_component bookkeeping pPXF needs.

    Forbidden lines are identified by a '[' in the line name, e.g.
    '[OIII]5007'. They get their own kinematic component because forbidden
    and Balmer lines can have different kinematics. (This relies on
    util.emission_lines listing Balmer lines before forbidden ones.)
    """
    reg_dim = sps.templates.shape[1:]
    stars_2d = sps.templates.reshape(sps.templates.shape[0], -1)
    templates = np.column_stack([stars_2d, gas_templates])

    n_stars = stars_2d.shape[1]
    n_forbidden = sum("[" in name for name in gas_names)
    n_balmer = len(gas_names) - n_forbidden

    component = [0] * n_stars + [1] * n_balmer + [2] * n_forbidden
    gas_component = np.array(component) > 0

    return dict(templates=templates, reg_dim=reg_dim, component=component,
                gas_component=gas_component, gas_names=gas_names,
                n_stars=n_stars, n_balmer=n_balmer, n_forbidden=n_forbidden)


def prepare_photometry(phot, combo, z, prep, sps):
    """
    Build pPXF's `phot` dict and the spectrum -> photometry scale factor.

    `phot` is the dict built by the runner (keys phot_galaxy, noise,
    flux_orig), in 1e-17 erg/s/cm^2/A, ordered as PHOT_BANDS. Bands the
    templates cannot predict at this z are dropped.

    spec_scale is the least-squares factor that brings the spectrum's
    synthetic g and r fluxes onto the photometry.

    Returns (phot_ppxf, spec_scale).
    """
    phot_galaxy = np.asarray(phot["phot_galaxy"])
    phot_noise = np.asarray(phot["noise"])

    p1 = util.synthetic_photometry(sps.lam_temp, combo["templates"],
                                   bands=list(PHOT_BANDS), redshift=z)
    phot_ppxf = {
        "templates": p1.flux[p1.ok],
        "galaxy": phot_galaxy[p1.ok],
        "noise": phot_noise[p1.ok],
        "lam": p1.lam_eff[p1.ok],
        "flux_orig": phot["flux_orig"],
    }

    p2 = util.synthetic_photometry(prep["lam_gal"], prep["galaxy"],
                                   bands=list(SPEC_SCALE_BANDS), redshift=z)
    g = phot_ppxf["galaxy"][:2]
    n = phot_ppxf["noise"][:2]
    spec_scale = np.sum(p2.flux * g / n**2) / np.sum(p2.flux**2 / n**2)

    return phot_ppxf, spec_scale


# ======================================================================
# pPXF call setup
# ======================================================================
def _start_and_bounds(n_moments=2):
    """Per-component starting guesses and bounds: [stars, Balmer, forbidden]."""
    start = [[0.0, 150.0], [0.0, 50.0], [0.0, 50.0]]
    bounds = [
        [list(V_BOUNDS), list(SIGMA_BOUNDS_STARS)],
        [list(V_BOUNDS), list(SIGMA_BOUNDS_GAS)],
        [list(V_BOUNDS), list(SIGMA_BOUNDS_GAS)],
    ]
    moments = [n_moments, n_moments, n_moments]
    return start, bounds, moments


def _base_ppxf_kwargs(prep, sps, combo, goodpixels, phot, fit_dust, tie_balmer):
    """Keyword arguments shared by every ppxf call for one galaxy."""
    _, bounds, moments = _start_and_bounds()
    kw = dict(moments=moments, degree=-1, mdegree=-1, bounds=bounds,
              component=combo["component"],
              gas_component=combo["gas_component"],
              gas_names=combo["gas_names"], reg_dim=combo["reg_dim"],
              lam=prep["lam_gal"], lam_temp=sps.lam_temp,
              goodpixels=goodpixels, quiet=True, phot=phot)
    if fit_dust:
        kw["reddening"] = 0.0
        # gas_reddening is only meaningful with tied Balmer ratios
        if tie_balmer:
            kw["gas_reddening"] = 0.0
    return kw


def _at_bound(v, sigma, tol=1.0):
    """True if either stellar kinematic parameter sits on its search boundary."""
    return bool(
        abs(v - V_BOUNDS[0]) < tol or abs(v - V_BOUNDS[1]) < tol
        or abs(sigma - SIGMA_BOUNDS_STARS[0]) < tol
        or abs(sigma - SIGMA_BOUNDS_STARS[1]) < tol
    )


# ======================================================================
# Regularization
# ======================================================================
def _print_noise_diagnostics(noise):
    noise = np.asarray(noise)
    finite = noise[np.isfinite(noise)]
    bad = np.where(~np.isfinite(noise) | (noise <= 0))[0]
    print("\n=== pPXF NOISE DIAGNOSTICS ===")
    print(f"shape {noise.shape}; NaN {np.isnan(noise).sum()}, "
          f"Inf {np.isinf(noise).sum()}, zero {(noise == 0).sum()}, "
          f"negative {(noise < 0).sum()}")
    print(f"finite min/max: {finite.min():.4g} / {finite.max():.4g}")
    if bad.size:
        print("first bad indices:", bad[:20])


def tune_regularization(prep, combo, base_kwargs,
                        fix_stellar_kin=STELLAR_KIN_FIXED,
                        regul_grid=DEFAULT_REGUL_GRID,
                        verbose=False, debug=False):
    """
    Find the largest regularization the data will tolerate.

    Recipe (pPXF documentation):
      1. Fit unregularised, rescale the noise so chi2/DOF = 1.
      2. With all kinematics frozen, raise regul until
         Delta_chi2 = chi2(regul) - chi2(0) reaches sqrt(2*N_good), the
         1-sigma width of the chi2 distribution.

    Stellar kinematics are frozen at `fix_stellar_kin` (or at the
    unregularised solution if None); gas kinematics at the unregularised
    solution. Stops early if Delta_chi2 plateaus -- a data-quality limit,
    not a grid-size problem.

    Returns (best_regul, prep_with_rescaled_noise).
    """
    templates, velscale = combo["templates"], prep["velscale"]
    n_good = base_kwargs["goodpixels"].size
    target = np.sqrt(2.0 * n_good)
    start, _, _ = _start_and_bounds()

    if debug:
        _print_noise_diagnostics(prep["noise"])

    # 1. Unregularised fit, then rescale the noise to chi2/DOF = 1 and refit.
    pp0 = ppxf(templates, prep["galaxy"], prep["noise"], velscale, start,
               regul=0, **base_kwargs)
    prep = dict(prep, noise=prep["noise"] * np.sqrt(pp0.chi2))
    pp0 = ppxf(templates, prep["galaxy"], prep["noise"], velscale, start,
               regul=0, **base_kwargs)

    # 2. Freeze kinematics, then scan regul.
    frozen = [list(s) for s in pp0.sol]
    if fix_stellar_kin is not None:
        frozen[0] = list(fix_stellar_kin)
    fixed = [[True, True]] * len(frozen)

    def chi2_total(regul):
        pp = ppxf(templates, prep["galaxy"], prep["noise"], velscale, frozen,
                  regul=regul, fixed=fixed, **base_kwargs)
        return pp.chi2 * n_good

    chi2_0 = chi2_total(0)
    if verbose:
        print(f"    tuning regul: {n_good} good pixels, "
              f"target Delta_chi2={target:.1f}")

    best_regul, prev, n_flat = 0.0, None, 0
    for regul in regul_grid:
        dchi2 = chi2_total(regul) - chi2_0
        if verbose:
            print(f"      regul={regul:<8g} Delta_chi2={dchi2:8.1f}")
        if dchi2 > target:
            break
        best_regul = regul

        if prev is not None and abs(dchi2 - prev) < 0.5:
            n_flat += 1
            if n_flat >= 2:
                if verbose:
                    print("      Delta_chi2 plateaued -- data cannot "
                          "constrain the SFH further")
                break
        else:
            n_flat = 0
        prev = dchi2

    if verbose:
        print(f"    -> regul = {best_regul}")
    return best_regul, prep


# ======================================================================
# Stellar mass, M/L and SFR
# ======================================================================
def formed_mass_to_light(sps, weights, band=MASS_BAND, redshift=0, quiet=False):
    """
    Stellar M/L in solar units for the FORMED mass (cf. sps.mass_to_light,
    which gives the surviving mass: living stars + remnants).

    Accepts light or mass weights from pPXF; the weights are divided by
    sps.flux to revert any template normalisation. The overall
    normalisation of the weights does not affect M/L.
    """
    assert sps.templates_full.shape[1:] == weights.shape, \
        "Input weight dimensions do not match"

    p1 = util.synthetic_photometry(sps.lam_temp_full, sps.templates_full, band,
                                   redshift=redshift, quiet=True)
    dist = 3.085677581491367e+19    # 10 pc in cm
    p1.flux /= 4 * np.pi * dist**2  # luminosity -> flux at 10 pc
    p1.flux *= 3.828e+33            # spectra are in Lsun (IAU 2015)

    sun = np.load(resources.files("ppxf") / "sps_models/spectra_sun_vega.npz")
    p2 = util.synthetic_photometry(sun["lam"], sun["flux_sun"], band,
                                   redshift=redshift, quiet=True)

    mass_weights = weights / sps.flux
    lum = p1.flux / p2.flux          # in solar luminosities
    ml_formed = np.sum(mass_weights) / np.sum(weights * lum)

    if not quiet:
        ml_surviving = (np.sum(mass_weights * sps.mass_no_gas_grid)
                        / np.sum(weights * lum))
        print(f"(M*/L)={ml_formed:#.4g} ({band} at z={redshift:#.4f})")
        print(f"(M*/L)_surviving={ml_surviving:#.4g} ({band} at z={redshift:#.4f})")

    return ml_formed


def rest_frame_L_over_Lsun(lam_gal, galaxy, norm, z, *,
                           flux_unit_1e17=True, band=MASS_BAND,
                           M_sun_AB=M_SUN_AB_R, verbose=False, tag=""):
    """
    L/Lsun in `band` from a rest-frame spectrum, for converting M/L into an
    absolute stellar mass: mstar = ml * L_over_Lsun.

    Standard distance-modulus calculation (flat LCDM, H0=70, Om0=0.3) with
    pPXF's util.mag_spectrum:

        M_abs = m_AB - 5*log10(d_L / 10 pc)
        L/Lsun = 10**(-0.4*(M_abs - M_sun_AB))

    Assumptions:
    1. flux_unit_1e17=True: `galaxy * norm` is in the DESI/SDSS convention
       of 1e-17 erg/s/cm^2/A.
    2. The observed f_lambda is multiplied by (1+z) and then evaluated with
       redshift=0, because `lam_gal` is ALREADY rest frame -- this gives the
       rest-frame magnitude directly.

    Returns L/Lsun, or nan if mag_spectrum fails -- e.g. if the fitted
    rest-frame range does not fully cover `band` (happens for the
    highest-redshift galaxies).
    """
    d_L_pc = _COSMO.luminosity_distance(z).to(u.pc).value

    flux = galaxy * norm * (1.0 + z)
    if flux_unit_1e17:
        flux = flux * 1e-17

    try:
        m_AB = float(np.atleast_1d(util.mag_spectrum(
            lam_gal, flux, bands=band, redshift=0, system="AB", quiet=True))[0])
    except Exception as exc:
        if verbose:
            print(f"[{tag}] L/Lsun skipped -- mag_spectrum failed: {exc!r}")
        return np.nan

    M_abs = m_AB - 5 * np.log10(d_L_pc / 10.0)
    return 10 ** (-0.4 * (M_abs - M_sun_AB))


_AGE_BINS = [(0.0, 0.01, "<10 Myr"), (0.01, 0.1, "10-100 Myr"),
             (0.1, 0.5, "100-500 Myr"), (0.5, 1.0, "500 Myr-1 Gyr"),
             (1.0, 5.0, "1-5 Gyr"), (5.0, np.inf, ">5 Gyr")]


def derive_quantities(z, pp, sps, combo, prep, compute_mstar=True,
                      verbose=False, debug=False):
    """
    Everything derived from one pPXF fit: light/mass weights, mean ages and
    metallicities, M/L, stellar masses, SFRs, dust and kinematics.

    Used for the main fit and for every bootstrap realisation. Returns a
    dict containing (at least) every name in BOOT_QUANTITIES.
    """
    weights = pp.weights[~combo["gas_component"]].reshape(combo["reg_dim"])

    light_weights = weights * sps.lw_flux_conv / sps.flux
    light_weights = light_weights / light_weights.sum()
    logage_lw, metal_lw = sps.mean_age_metal(light_weights, quiet=True)

    mass_weights = weights / sps.flux
    mass_weights = mass_weights / mass_weights.sum()
    logage_mw, metal_mw = sps.mean_age_metal(mass_weights, quiet=True)

    ml = sps.mass_to_light(mass_weights, band=MASS_BAND, redshift=0, quiet=True)
    ml_formed = formed_mass_to_light(sps, mass_weights, band=MASS_BAND,
                                     redshift=0, quiet=True)

    if debug:
        ages = np.asarray(sps.age_grid)
        print("\n=== STELLAR POPULATION CONTRIBUTIONS ===")
        for lo, hi, label in _AGE_BINS:
            sel = (ages >= lo) & (ages < hi)
            print(f"{label:15s}: mass={mass_weights[sel].sum():.4f}, "
                  f"light={light_weights[sel].sum():.4f}")
        ml_z = sps.mass_to_light(mass_weights, band=MASS_BAND, redshift=z, quiet=True)
        print(f"M/L (z=0): {ml:.6f}   M/L (z={z:.5f}): {ml_z:.6f}   "
              f"ratio {ml_z / ml:.6f}")

    mstar = mstar_formed = np.nan
    if compute_mstar:
        l_over_lsun = rest_frame_L_over_Lsun(
            prep["lam_gal"], pp.bestfit - pp.gas_bestfit, prep["norm"], z,
            verbose=verbose)
        # Undo the fitted stellar attenuation in the r band.
        av_star = float(getattr(pp, "reddening", 0.0) or 0.0)
        l_over_lsun *= 10 ** (0.4 * CALZETTI_AR_OVER_AV * av_star)
        if np.isfinite(l_over_lsun):
            mstar = float(ml) * l_over_lsun
            mstar_formed = float(ml_formed) * l_over_lsun

    log_sfh, log_sfr = get_sfr(sps.age_grid, mass_weights, mstar_formed)

    return dict(
        light_weights=light_weights, mass_weights=mass_weights,
        logage_lw=float(logage_lw), metal_lw=float(metal_lw),
        logage_mw=float(logage_mw), metal_mw=float(metal_mw),
        ml=float(ml), mstar=mstar, mstar_formed=mstar_formed,
        log_sfh=log_sfh, log_sfr_10myr=log_sfr[0],
        log_sfr_30myr=log_sfr[1], log_sfr_100myr=log_sfr[2],
        av_stars=_as_float(getattr(pp, "reddening", None)),
        av_gas=_as_float(getattr(pp, "gas_reddening", None)),
        v=float(pp.sol[0][0]), sigma=float(pp.sol[0][1]),
        v_gas=float(pp.sol[1][0]), sigma_gas=float(pp.sol[1][1]),
    )


def get_sfr(age_grid, mass_weights, mstar_formed):
    """
    Star formation history and recent average SFRs.

    Parameters
    ----------
    age_grid : SSP ages in Gyr, shape (n_age, n_metal).
    mass_weights : fraction of the total formed mass in each SSP,
        shape (n_age, n_metal); must sum to 1.
    mstar_formed : total stellar mass formed, Msun.

    Returns
    -------
    log_sfh : log10(SFR / Msun/yr) in each age bin (-inf where zero).
    log_sfr : [log10 SFR] averaged over the last 10, 30, 100 Myr.
    """
    ages = np.asarray(age_grid[:, 0], dtype=float)
    mass_fraction = np.sum(mass_weights, axis=1)   # sum over metallicity

    if (not np.isfinite(mstar_formed) or mstar_formed <= 0
            or not np.all(np.isfinite(ages)) or not np.all(ages > 0)
            or not np.all(np.diff(ages) > 0)
            or not np.all(np.isfinite(mass_fraction))
            or np.any(mass_fraction < 0)
            or not np.isclose(mass_fraction.sum(), 1.0)):
        return np.full(len(ages), np.nan), [np.nan] * 3

    # Bin edges halfway between ages in log(age); youngest bin reaches t=0.
    log_age = np.log10(ages)
    log_edges = np.concatenate([
        [log_age[0] - (log_age[1] - log_age[0]) / 2],
        (log_age[:-1] + log_age[1:]) / 2,
        [log_age[-1] + (log_age[-1] - log_age[-2]) / 2],
    ])
    age_edges = 10 ** log_edges
    age_edges[0] = 0.0
    bin_width = np.diff(age_edges)                 # Gyr

    mass_formed = mass_fraction * mstar_formed
    sfh = mass_formed / (bin_width * 1e9)

    def average_sfr(window_gyr):
        """Mean SFR over the last `window_gyr`, splitting bins by overlap."""
        overlap = np.clip(np.minimum(age_edges[1:], window_gyr) - age_edges[:-1],
                          0.0, None)
        recent_mass = np.sum(mass_formed * overlap / bin_width)
        return recent_mass / (window_gyr * 1e9)

    sfrs = [average_sfr(w) for w in (0.01, 0.03, 0.1)]

    # -inf for zero SFR rather than inventing a floor
    with np.errstate(divide="ignore"):
        return np.log10(sfh), np.log10(sfrs).tolist()


def continuum_chi2(lam_rest, galaxy, bestfit, noise, half_width=15.0):
    """Reduced chi2 on line-free, good pixels only. Returns (chi2, n_pix)."""
    use = noise < 1e9
    for l0 in _STRONG_LINES:
        use &= np.abs(lam_rest - l0) > half_width
    n = int(use.sum())
    if n == 0:
        return np.nan, 0
    return float(np.sum(((galaxy - bestfit)[use] / noise[use])**2) / n), n


# ======================================================================
# Debug printouts
# ======================================================================
def _print_input_check(name, z, galaxy_dict):
    flux = np.asarray(galaxy_dict["flux"], dtype=np.float64)
    wave = np.asarray(galaxy_dict["lam"], dtype=np.float64)
    print(f"INPUT CHECK: TARGETID={name}, z={z:.6f}, Npix={len(flux)}, "
          f"flux_hash={hashlib.sha256(flux.tobytes()).hexdigest()[:16]}, "
          f"wave_hash={hashlib.sha256(wave.tobytes()).hexdigest()[:16]}")


def _print_fit_diagnostics(prep, pp, z):
    """Luminosity of data vs model, and the fitted kinematics."""
    print("\n=== STELLAR MASS LUMINOSITY DIAGNOSTICS ===")
    L = {tag: rest_frame_L_over_Lsun(prep["lam_gal"], spec, prep["norm"], z,
                                     verbose=True, tag=tag)
         for tag, spec in (("observed", prep["galaxy"]),
                           ("total model", pp.bestfit),
                           ("stellar model", pp.bestfit - pp.gas_bestfit))}
    for tag, val in L.items():
        print(f"{tag + ' luminosity:':26s}{val:.4e} Lsun")
    if np.isfinite(L["observed"]) and L["observed"] > 0 and np.isfinite(L["stellar model"]):
        print(f"stellar / observed:       {L['stellar model'] / L['observed']:.4f}")
        if np.isfinite(L["total model"]) and L["total model"] > 0:
            print(f"emission-line fraction:   "
                  f"{1 - L['stellar model'] / L['total model']:.4f}")

    print("\n=== KINEMATICS DIAGNOSTICS ===")
    for i, label in enumerate(("stellar", "Balmer gas", "forbidden gas")):
        print(f"{label:14s} v = {pp.sol[i][0]:8.2f} km/s   "
              f"sigma = {pp.sol[i][1]:7.2f} km/s")


def _print_sfh_check(name, sps, q):
    ages = sps.age_grid[:, 0]
    light_by_age = q["light_weights"].sum(axis=1)
    mass_by_age = q["mass_weights"].sum(axis=1)

    print("\nRECENT SSP MASS DISTRIBUTION")
    for age, frac in zip(ages, mass_by_age):
        if age <= 1.0:
            print(f"  age = {age * 1000:8.2f} Myr   mass fraction = {frac:.5f}")

    print(f"\nSFH CHECK: {name}")
    print(f"  total formed mass:    {q['mstar_formed']:.3e} Msun")
    print(f"  current stellar mass: {q['mstar']:.3e} Msun")
    for age_limit in (0.01, 0.03, 0.1, 1.0, 5.0):
        young = ages < age_limit
        print(f"  age < {age_limit:.2f} Gyr: "
              f"light fraction = {light_by_age[young].sum():.4f}, "
              f"mass fraction = {mass_by_age[young].sum():.4f}")


# ======================================================================
# The fit
# ======================================================================
def fit_galaxy(galaxy_dict, sps_name="galaxev", edge_trim=50.0, phot=None,
               tie_balmer=True, fit_dust=True, tune_regul=True,
               fixed_regul=None, max_age_gyr=None,
               fix_stellar_kin=STELLAR_KIN_FIXED,
               compute_mstar=True, nboot=0, seed=0,
               verbose=False, debug=False):
    """
    Fit one galaxy: stars + gas (+ photometry) simultaneously.

    Parameters
    ----------
    galaxy_dict : one entry from load_desi_pickle().
    sps_name : SSP library ("galaxev", "emiles", "fsps").
    edge_trim : Angstroms trimmed off each edge of this galaxy's own
        de-redshifted range (see prepare_galaxy).
    phot : photometry dict from the runner (phot_galaxy, noise, flux_orig),
        or None to fit the spectrum alone.
    tie_balmer : tie Balmer ratios to Case B (see prepare_gas_templates).
    fit_dust : fit reddening for stars, and for gas if tie_balmer.
    tune_regul : run the regularization search. If False, regul=0.
    fixed_regul : use this regul instead (overrides tune_regul).
    max_age_gyr : cap on SSP ages (default: age of the Universe at z).
    fix_stellar_kin : (v, sigma) at which stellar kinematics are held fixed,
        or None to fit them.
    compute_mstar : compute absolute stellar masses (see
        rest_frame_L_over_Lsun for the assumptions).
    nboot : wild-bootstrap iterations for uncertainties. 0 disables it.
    verbose : progress output.
    debug : extra diagnostic printouts.

    Returns
    -------
    (FitResult, pp) -- pp is the final ppxf object, or None if the fit failed.
    """
    name, z = galaxy_dict["name"], galaxy_dict["z"]
    t0 = time.time()
    if verbose:
        print(f"\n=== {name}  z={z:.4f} ===")
    if debug:
        _print_input_check(name, z, galaxy_dict)

    try:
        # 1. Milky Way extinction.
        flux_mw, noise_mw, ebv_mw = correct_mw_extinction(
            galaxy_dict["lam"], galaxy_dict["flux"], galaxy_dict["noise"],
            galaxy_dict["ra"], galaxy_dict["dec"])
        if verbose:
            print(f"    MW correction: E(B-V) = {ebv_mw:.4f}")

        # 2. Degrade the galaxy to the template resolution.
        flux_deg, noise_deg, fwhm_target_rest = degrade_to_template_res(
            galaxy_dict["lam"], flux_mw, noise_mw, z, sps_name,
            data_fwhm=galaxy_dict["wave_fwhm"], verbose=verbose)

        # 3. De-redshift, trim, log-rebin.
        prep = prepare_galaxy(galaxy_dict["lam"], flux_deg, noise_deg, z,
                              edge_trim=edge_trim, verbose=verbose)

        # 4. Templates. The galaxy is now at the templates' native
        #    resolution, so that is what we pass as fwhm_gal.
        fwhm_gal = {"lam": galaxy_dict["lam"] / (1.0 + z), "fwhm": fwhm_target_rest}
        lam_range_temp = [max(100.0, prep["wave_range"][0] - TEMPLATE_LAM_MARGIN),
                          DECAM_Z_RED_EDGE + TEMPLATE_LAM_MARGIN]

        sps = prepare_templates(prep["velscale"], fwhm_gal, z, sps_name=sps_name,
                                lam_range_temp=lam_range_temp,
                                max_age_gyr=max_age_gyr,
                                verbose=verbose, debug=debug)
        gas_templates, gas_names, _ = prepare_gas_templates(
            sps, np.array(lam_range_temp), fwhm_gal,
            tie_balmer=tie_balmer, verbose=verbose)
        combo = combine_templates(sps, gas_templates, gas_names)

        # 5. Photometry, and scale the spectrum onto it.
        spec_scale = np.nan
        if phot is not None:
            phot, spec_scale = prepare_photometry(phot, combo, z, prep, sps)
            prep["galaxy"] = prep["galaxy"] * spec_scale
            prep["noise"] = prep["noise"] * spec_scale
            if verbose:
                print(f"    spectrum scaled to photometry: spec_scale = {spec_scale:.6g}")

        # prepare_galaxy already trimmed to wave_range, so every pixel is good.
        goodpixels = np.arange(prep["lam_gal"].size)
        base_kwargs = _base_ppxf_kwargs(prep, sps, combo, goodpixels, phot,
                                        fit_dust, tie_balmer)

        # 6. Regularization.
        if fixed_regul is not None:
            regul = float(fixed_regul)
            if regul < 0:
                raise ValueError("fixed_regul must be non-negative")
            if verbose:
                print(f"    using manually specified regul = {regul}")
        elif tune_regul:
            regul, prep = tune_regularization(
                prep, combo, base_kwargs, fix_stellar_kin=fix_stellar_kin,
                verbose=verbose, debug=debug)
        else:
            regul = 0.0

        # 7. Final fit.
        start, _, _ = _start_and_bounds()
        fit_kwargs = dict(base_kwargs, regul=regul)
        if fix_stellar_kin is not None:
            start[0] = list(fix_stellar_kin)
            fit_kwargs["fixed"] = [[True, True], [False, False], [False, False]]

        pp = ppxf(combo["templates"], prep["galaxy"], prep["noise"],
                  prep["velscale"], start, **fit_kwargs)

        chi2_cont, _ = continuum_chi2(prep["lam_gal"], prep["galaxy"],
                                      pp.bestfit, prep["noise"])
        if debug:
            _print_fit_diagnostics(prep, pp, z)

        q = derive_quantities(z, pp, sps, combo, prep, compute_mstar=compute_mstar,
                              verbose=verbose, debug=debug)
        if debug:
            _print_sfh_check(name, sps, q)

        gas_flux, gas_flux_err = {}, {}
        if getattr(pp, "gas_flux", None) is not None:
            for nm, fl, er in zip(combo["gas_names"], pp.gas_flux, pp.gas_flux_error):
                gas_flux[nm] = float(fl)
                gas_flux_err[nm] = float(er)

        res = FitResult(
            name=name, z=z, ok=True,
            chi2=float(pp.chi2), chi2_cont=chi2_cont, regul_used=float(regul),
            spec_scale=float(spec_scale),
            max_age_gyr=max_age_gyr if max_age_gyr is not None else np.nan,
            logage_lw=q["logage_lw"], logage_mw=q["logage_mw"],
            metal_lw=q["metal_lw"], metal_mw=q["metal_mw"],
            ml=q["ml"], mstar=q["mstar"], mstar_formed=q["mstar_formed"],
            av_stars=q["av_stars"], av_gas=q["av_gas"], ebv_mw=float(ebv_mw),
            v=q["v"], sigma=q["sigma"], kin_at_bound=_at_bound(q["v"], q["sigma"]),
            v_gas=q["v_gas"], sigma_gas=q["sigma_gas"],
            gas_flux=gas_flux, gas_flux_err=gas_flux_err,
            log_sfh=q["log_sfh"], log_sfr_10myr=q["log_sfr_10myr"],
            log_sfr_30myr=q["log_sfr_30myr"], log_sfr_100myr=q["log_sfr_100myr"],
            weights=q["light_weights"], mass_weights=q["mass_weights"],
            age_grid=sps.age_grid, metal_grid=sps.metal_grid,
            lam_gal=prep["lam_gal"], galaxy=prep["galaxy"],
            bestfit=pp.bestfit, gas_bestfit=getattr(pp, "gas_bestfit", None),
        )

        # 8. Optional bootstrap: replaces each quantity by its median and
        #    fills in the 16th/84th percentiles.
        if nboot > 0:
            boot = _bootstrap(z, pp, prep, sps, combo, fit_kwargs, nboot, seed,
                              compute_mstar, verbose)
            for qname, (lo, med, hi) in boot.items():
                setattr(res, f"{qname}_16", float(lo))
                setattr(res, qname, float(med))
                setattr(res, f"{qname}_84", float(hi))

        if verbose:
            print(f"    chi2={res.chi2:.3f}  logage_lw={res.logage_lw:.2f}  "
                  f"metal_lw={res.metal_lw:.2f}  A_V={res.av_stars:.2f}  "
                  f"mstar={res.mstar:.2e}")
            print(f"    v={res.v:.0f}  sigma={res.sigma:.0f}"
                  f"{'  [AT BOUND]' if res.kin_at_bound else ''}")
            print(f"    done in {time.time() - t0:.1f}s")
        return res, pp

    except Exception as exc:
        if verbose:
            print(f"    FAILED: {exc!r}")
            filename = getattr(exc, "filename", None)
            if filename:
                print(f"    (missing file: {filename})")
            traceback.print_exc()
        return FitResult(name=name, z=z, ok=False, error_msg=repr(exc)), None


def _bootstrap(z, pp, prep, sps, combo, fit_kwargs, nboot, seed,
               compute_mstar, verbose):
    """
    Wild bootstrap: refit bestfit + residuals * random(+-1) `nboot` times,
    with the same settings as the final fit (regul, dust, fixed kinematics).

    Returns {quantity: [p16, p50, p84]} for every name in BOOT_QUANTITIES.
    """
    rng = np.random.default_rng(seed)
    resid = prep["galaxy"] - pp.bestfit
    warm = [list(s) for s in pp.sol]
    samples = {qname: [] for qname in BOOT_QUANTITIES}

    for i in range(nboot):
        pert = pp.bestfit + resid * rng.choice([-1.0, 1.0], size=resid.size)
        ppb = ppxf(combo["templates"], pert, prep["noise"], prep["velscale"],
                   warm, **fit_kwargs)
        q = derive_quantities(z, ppb, sps, combo, prep, compute_mstar=compute_mstar)
        for qname in BOOT_QUANTITIES:
            samples[qname].append(q[qname])

        if verbose and (i + 1) % max(1, nboot // 4) == 0:
            print(f"      bootstrap {i + 1}/{nboot}")

    return {qname: np.percentile(vals, [16, 50, 84])
            for qname, vals in samples.items()}


# ======================================================================
# Output
# ======================================================================
def results_to_rows(results, line_subset=None):
    """
    Flatten FitResults into dicts suitable for csv.DictWriter.

    Emission-line fluxes become columns named flux_<line> / fluxerr_<line>.
    All results share the same columns, with blanks where a line is absent
    (the available lines depend on redshift, since coverage varies).
    """
    if line_subset is None:
        line_subset = sorted({ln for r in results if r.ok for ln in r.gas_flux})

    rows = []
    for r in results:
        d = asdict(r)
        for key in _ARRAY_FIELDS:
            d.pop(key, None)
        for ln in line_subset:
            d[f"flux_{ln}"] = r.gas_flux.get(ln, "")
            d[f"fluxerr_{ln}"] = r.gas_flux_err.get(ln, "")
        rows.append(d)
    return rows, line_subset
