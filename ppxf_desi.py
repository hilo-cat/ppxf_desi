"""
ppxf_desi.py
============
Combined stellar-continuum + gas emission-line fitting of DESI galaxy spectra
with pPXF.

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

Known open issues
------------------
1. **A_V** is probably unreliable as of now, need to figure out more.
2. **Halpha/Hbeta** ratios and dust approximation from that needs to be 
   confirmed too.
3. **Stellar kinematics** (v, sigma) often land right on the search
   bounds for these faint galaxies -- flagged per galaxy via
   `kin_at_bound` in the CSV. Doesn't affect the population/mass fit,
   just the kinematics themselves.
"""

from __future__ import annotations

import os
import pickle
import time
import traceback
import glob
from dataclasses import dataclass, field, asdict
from importlib import resources

import numpy as np
from scipy.ndimage import gaussian_filter1d

from ppxf.ppxf import ppxf
import ppxf.ppxf_util as util
import ppxf.sps_util as sps_util

from sedpy.smoothing import smoothspec
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
_COSMO = FlatLambdaCDM(H0=70, Om0=0.3)

__all__ = [
    "FitResult",
    "load_desi_pickle",
    "degrade_to_template_res",
    "prepare_galaxy",
    "prepare_templates",
    "prepare_gas_templates",
    "combine_templates",
    "rest_frame_L_over_Lsun",
    "fit_galaxy",
    "results_to_rows",
]

C_KMS = 299792.458
FWHM_TO_SIGMA = 1.0 / 2.3548

# Kinematic search bounds.
V_BOUNDS = (-1000.0, 1000.0)
SIGMA_BOUNDS_STARS = (10.0, 500.0)
SIGMA_BOUNDS_GAS = (5.0, 500.0)

SPS_DATA_URL = "https://raw.githubusercontent.com/micappe/ppxf_data/main/"
SPS_FILENAMES = {
    "emiles":  "spectra_emiles_9.0.npz",
    "galaxev": "spectra_galaxev_9.0.npz",
    "fsps":    "spectra_fsps_9.0.npz",
    "xsl":     "spectra_xsl_9.0.npz",
}


def _ensure_template_file(sps_name, ppxf_dir=None):
    """Return the local path to an SPS template file, downloading it first
    if it is not already cached."""
    if ppxf_dir is None:
        ppxf_dir = os.path.dirname(os.path.realpath(sps_util.__file__))
    models_dir = os.path.join(ppxf_dir, "sps_models")
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, SPS_FILENAMES[sps_name])

    if not os.path.exists(path):
        import urllib.request
        url = SPS_DATA_URL + SPS_FILENAMES[sps_name]
        print(f"    {sps_name} templates not cached -- downloading from "
              f"{url} (one-time)...")
        urllib.request.urlretrieve(url, path)

    return path


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

    # stellar population
    logage_lw: float = np.nan     # light-weighted log10(age/yr)
    logage_mw: float = np.nan     # mass-weighted
    metal_lw: float = np.nan      # light-weighted [M/H]
    metal_mw: float = np.nan      # mass-weighted
    ml: float = np.nan            # M/L in SDSS/r, solar units
    mstar: float = np.nan         # absolute stellar mass, Msun (see
                                   # rest_frame_L_over_Lsun for assumptions)

    # dust
    av_stars: float = np.nan
    av_gas: float = np.nan

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

    # star=formation rate
    log_sfh: np.ndarray = None
    log_sfr_10myr: float = np.nan
    log_sfr_30myr: float = np.nan
    log_sfr_100myr: float = np.nan

    
    # bootstrap uncertainties (nan unless nboot > 0)
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
    weights: np.ndarray = None
    age_grid: np.ndarray = None
    metal_grid: np.ndarray = None
    lam_gal: np.ndarray = None    # rest-frame wavelength of the fitted spectrum
    galaxy: np.ndarray = None     # the data, normalised, as fitted
    bestfit: np.ndarray = None    # total model (stars + gas)
    gas_bestfit: np.ndarray = None  # gas-only component of the model

def formed_mass_to_light(sps, weights, band='SDSS/r', redshift=0, quiet=False):
    """
    This function calculates the stellar mass-to-light ratio (M*/L), in
    solar units, in a given band at a specified redshift, using the pPXF
    output weights. The M*/L accounts for both living and stellar remnants,
    but not the gas ejected during stellar evolution.

    The function accepts either light or mass weights as returned from
    pPXF. If the weights are light weights, they are automatically
    converted to mass weights using the .flux attribute of the class. The
    weights overall normalization does not affect the M*/L calculation.
    """
    assert sps.templates_full.shape[1:] == weights.shape, "Input weight dimensions do not match"

    p1 = util.synthetic_photometry(sps.lam_temp_full, sps.templates_full, band, 
                                   redshift=redshift, quiet=True)
    dist = 3.085677581491367e+19    # 10pc in cm by definition
    p1.flux /= 4*np.pi*dist**2      # convert luminosity to observed flux/cm^2 at 10pc
    p1.flux *= 3.828e+33            # spectra are in units of Lsun (erg/s IAU 2015)

    ppxf_dir = resources.files('ppxf')  # path of current file
    filename = ppxf_dir / 'sps_models/spectra_sun_vega.npz'
    a = np.load(filename)           # Spectrum in cgs/A at 10pc
    p2 = util.synthetic_photometry(a["lam"], a["flux_sun"], band, 
                                       redshift=redshift, quiet=True) 

    mass_weights = weights/sps.flux    # Revert possible templates normalization
    lum = p1.flux/p2.flux               # Lum in solar luminosities
    mlpop = np.sum(mass_weights*sps.mass_no_gas_grid)/np.sum(weights*lum)
    mlpop = np.sum(mass_weights)/np.sum(weights*lum)
    
    if not quiet:
        print(f'(M*/L)={mlpop:#.4g} ({band} at z={redshift:#.4f})')

    return mlpop

    
# ======================================================================
# Data loading
# ======================================================================
def load_desi_pickle(path):
    """
    Read the DESI spectra pickle into a list of plain dicts.

    Bad pixels: DESI flags a pixel bad if ivar <= 0 OR mask != 0. Treating
    any nonzero mask as bad is the conservative choice.
    """
    with open(path, "rb") as f:
        df = pickle.load(f)

    galaxies = []
    for _, row in df.iterrows():
        lam = np.asarray(row["wavelength"], dtype=float)
        flux = np.asarray(row["flux"], dtype=float)
        ivar = np.asarray(row["ivar"], dtype=float)
        mask = np.asarray(row["mask"], dtype=int)

        noise = np.full_like(ivar, 1e10)
        good = (ivar > 0) & (mask == 0)
        noise[good] = 1.0 / np.sqrt(ivar[good])

        galaxies.append(dict(
            name=str(row["specid"]),
            z=float(row["redshift"]),
            lam=lam,
            flux=flux,
            noise=noise,
        ))
    return galaxies


# ======================================================================
# Resolution matching
# ======================================================================
def get_template_native_fwhm(sps_name, ppxf_dir=None):
    """
    Read the native instrumental FWHM(lambda) stored inside an SPS template
    file. Returns (lam, fwhm), both in Angstroms, rest frame of the templates.
    """
    path = _ensure_template_file(sps_name, ppxf_dir)
    data = np.load(path)
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

    The required kernel follows from Gaussians adding in quadrature:

        fwhm_kernel = sqrt(fwhm_target^2 - fwhm_data^2)

    Returns (flux_deg, noise_deg, fwhm_target_rest).
    """
    if data_fwhm is None:
        data_fwhm = 1.53
    data_fwhm = np.broadcast_to(np.asarray(data_fwhm, dtype=float), lam_obs.shape)

    lam_temp, fwhm_temp = get_template_native_fwhm(sps_name)
    lam_rest = lam_obs / (1.0 + z)
    fwhm_target_rest = np.interp(lam_rest, lam_temp, fwhm_temp)
    fwhm_target_obs = fwhm_target_rest * (1.0 + z)

    diff_sq = fwhm_target_obs ** 2 - data_fwhm ** 2
    fwhm_kernel = np.sqrt(np.clip(diff_sq, 0.0, None))
    flux_smoothed = smoothspec(lam_obs,flux,resolution=fwhm_kernel*FWHM_TO_SIGMA,smoothtype='lsf')

    dlam = np.median(np.diff(lam_obs))
    sigma_pix = fwhm_kernel * FWHM_TO_SIGMA / dlam
    sigma_med = float(np.median(sigma_pix))

    if sigma_med <= 0:
        if verbose:
            print("    resolution: data already at or below template "
                  "resolution -- no degradation applied")
        return flux.copy(), noise.copy(), fwhm_target_rest

    #flux_deg = gaussian_filter1d(flux, sigma_med)

    noise_factor = 1.0 / np.sqrt(2.0 * np.sqrt(np.pi) * sigma_pix) #max(sigma_med, 1e-3))
    noise_deg = noise * noise_factor

    if verbose:
        print(f"    resolution: degraded galaxy by sigma={sigma_med:.2f} pix "
              f"({np.median(fwhm_kernel):.2f} A) to reach template FWHM "
              f"~{np.median(fwhm_target_rest):.2f} A rest frame")

    return flux_smoothed, noise_deg, fwhm_target_rest


# ======================================================================
# Preparation
# ======================================================================
def prepare_galaxy(lam_obs, flux, noise, z, edge_trim=50.0, verbose=False):
    """
    De-redshift, trim, log-rebin, and normalise.

    Unlike a fixed rest-frame window, `edge_trim` is the ONLY fixed quantity
    here: it trims that many Angstroms off each edge of whatever rest-frame
    range this particular galaxy's spectrum actually covers, and fits
    everything else. DESI's observed-frame coverage is fixed (roughly
    3600-9824 A) but maps to a DIFFERENT rest-frame range for every galaxy
    depending on z, so the resulting fitted window is different for every
    galaxy -- wider for low-z targets (more rest-frame range fits inside a
    fixed observed window), narrower for high-z ones. The edge trim itself
    just avoids the first/last few pixels, where interpolation at the
    detector edge is least reliable.

    Returns a dict; `wave_range` in it is the actual (min, max) used for
    this galaxy, needed downstream to size the templates correctly.
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
    velscale = float(velscale)

    bad = ~np.isfinite(noise_rebin) | (noise_rebin <= 0)
    if bad.any():
        noise_rebin[bad] = 1e10
        if verbose:
            print(f"    {bad.sum()} pixels had bad noise -> downweighted")

    norm = np.median(galaxy[galaxy > 0]) if np.any(galaxy > 0) else 1.0
    galaxy = galaxy / norm
    noise_rebin = noise_rebin / norm

    lam_gal = np.exp(ln_lam)
    if verbose:
        print(f"    log-rebinned to {galaxy.size} pixels, "
              f"velscale={velscale:.2f} km/s/pix")

    return dict(galaxy=galaxy, noise=noise_rebin, lam_gal=lam_gal,
                ln_lam_gal=ln_lam, velscale=velscale, norm=norm,
                wave_range=wave_range)


def prepare_templates(velscale, fwhm_gal, z, sps_name="galaxev",
                      lam_range_temp=None, norm_range=(5000.0, 6000.0),
                      verbose=False):
    """
    Build the SSP template library.

    `norm_range` matters: with it set, template weights are LIGHT weights and
    mass-weighted quantities are recovered by dividing by sps.flux.

    `fwhm_gal` here is the galaxy's resolution AFTER degradation, i.e. the
    templates' own native resolution, so in practice no convolution occurs.
    """
    filename = _ensure_template_file(sps_name)
    sps = sps_util.sps_lib(filename, velscale, fwhm_gal,
                           lam_range=lam_range_temp,
                           norm_range=list(norm_range),
                           age_range=[0,_COSMO.age(z).value])
    sps.templates /= np.median(sps.templates)

    if verbose:
        print(f"    templates: {sps_name}, shape {sps.templates.shape} "
              f"(n_wave, n_age, n_metal)")
    return sps


def prepare_gas_templates(sps, lam_range_gal, fwhm_gal, tie_balmer=True,
                          limit_doublets=True, verbose=False):
    """
    Build Gaussian emission-line templates on the same log-lambda grid as the
    stellar templates.

    tie_balmer=True fixes the Balmer line ratios to Case B recombination.
    That is required for `gas_reddening` to be meaningful, but it means the
    individual Balmer fluxes are not independent measurements. Use
    tie_balmer=False for a free Halpha/Hbeta ratio -- but then do not also
    fit gas_reddening, as the two are degenerate.
    """
    gas_templates, gas_names, gas_wave = util.emission_lines(
        sps.ln_lam_temp, lam_range_gal, fwhm_gal,
        tie_balmer=tie_balmer, limit_doublets=limit_doublets)

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
    and Balmer lines can have different kinematics.
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

def prepare_photometry(
        phot, combo, z,
        spec_lam, spec_flux,
        temp_wave, prep, sps):
    phot_ppxf = {
        'noise':np.array(phot['noise']),
        'galaxy':np.array(phot['phot_galaxy']),
    }


    p1 = util.synthetic_photometry(sps.lam_temp, combo['templates'], bands=['DECam/DECam_g','DECam/DECam_r','DECam/DECam_z'], redshift=z)
    phot_lam, phot_lam_piv, phot_templates, phot_galaxy, bands = \
        p1.lam_eff[p1.ok], p1.lam_piv[p1.ok], p1.flux[p1.ok], phot_ppxf['galaxy'][p1.ok], np.array(['DECam/DECam_g','DECam/DECam_r','DECam/DECam_z'])[p1.ok]
    phot_ppxf = {"templates": phot_templates, "galaxy": phot_galaxy, "noise": phot_ppxf['noise'][p1.ok], "lam": phot_lam}



    p2 = util.synthetic_photometry(np.exp(prep['ln_lam_gal']),prep['galaxy'],bands=['DECam/DECam_g','DECam/DECam_r'],redshift=z)
    spec_scale = np.sum(p2.flux*phot_ppxf['galaxy'][0:2]/phot_ppxf['noise'][0:2]**2.)/np.sum(p2.flux**2./phot_ppxf['noise'][0:2]**2.)

    return phot_ppxf, spec_scale
    
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


def _at_bound(v, sigma, tol=1.0):
    """True if either kinematic parameter sits on its search boundary."""
    return bool(
        abs(v - V_BOUNDS[0]) < tol or abs(v - V_BOUNDS[1]) < tol
        or abs(sigma - SIGMA_BOUNDS_STARS[0]) < tol
        or abs(sigma - SIGMA_BOUNDS_STARS[1]) < tol
    )


# ======================================================================
# Regularization
# ======================================================================
def tune_regularization(prep, phot, sps, combo, goodpixels, regul_grid=None,
                        verbose=False):
    """
    Find the largest regularization the data will tolerate.

    Recipe (pPXF documentation):
      1. Fit unregularised, rescale the noise so chi2/DOF = 1.
      2. Raise regul until Delta_chi2 = chi2(regul) - chi2(0) reaches
         sqrt(2*N_good), the 1-sigma width of the chi2 distribution.

    Stops early if Delta_chi2 plateaus -- a data-quality limit, not a
    grid-size problem.

    Returns (best_regul, prep_with_rescaled_noise).
    """
    if regul_grid is None:
        regul_grid = (1280, 2560, 5120, 10240, 20480, 40960)

    start, bounds, moments = _start_and_bounds()
    n_good = goodpixels.size
    target = np.sqrt(2.0 * n_good)

    base = dict(moments=moments, degree=-1, mdegree=-1, bounds=bounds,
                component=combo["component"],
                gas_component=combo["gas_component"],
                gas_names=combo["gas_names"], reg_dim=combo["reg_dim"],
                lam=prep["lam_gal"], lam_temp=sps.lam_temp,
                goodpixels=goodpixels, quiet=True, phot=phot)

    pp0 = ppxf(combo["templates"], prep["galaxy"], prep["noise"],
               prep["velscale"], start, regul=0, **base)

    prep = dict(prep)
    prep["noise"] = prep["noise"] * np.sqrt(pp0.chi2)

    pp0 = ppxf(combo["templates"], prep["galaxy"], prep["noise"],
               prep["velscale"], start, regul=0, **base)
    chi2_0 = pp0.chi2 * n_good

    frozen = [list(s) for s in pp0.sol]
    fixed = [[True, True]] * 3

    if verbose:
        print(f"    tuning regul: {n_good} good pixels, target "
              f"Delta_chi2={target:.1f}")

    best_regul = 0.0
    prev = None
    n_flat = 0
    for regul in regul_grid:
        pp = ppxf(combo["templates"], prep["galaxy"], prep["noise"],
                  prep["velscale"], frozen, regul=regul, fixed=fixed, **base)
        dchi2 = pp.chi2 * n_good - chi2_0
        if verbose:
            print(f"      regul={regul:<7.0f} Delta_chi2={dchi2:8.1f}")

        best_regul = regul
        if dchi2 >= target:
            break
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
# Stellar mass
# ======================================================================
def rest_frame_L_over_Lsun(lam_gal, galaxy_normalized, norm, z, *,
                           flux_unit_1e17=True, band="SDSS/r",
                           M_sun_AB=4.64, verbose=False, tag=""):
    """
    Compute L/Lsun in `band` from the rest-frame spectrum, for converting
    M/L into an absolute stellar mass: mstar = ml * L_over_Lsun.

    Standard distance-modulus calculation using a flat LCDM cosmology
    (H0=70, Om0=0.3) and pPXF's own util.mag_spectrum:

        M_abs = m_AB - 5*log10(d_L / 10 pc)
        L/Lsun = 10**(-0.4*(M_abs - M_sun_AB))

    Two assumptions worth being aware of:

    1. flux_unit_1e17=True assumes the ORIGINAL DESI spectrum (before this
       module's internal /median renormalisation, undone here via `norm`) is
       in the standard DESI/SDSS convention of 1e-17 erg/s/cm^2/A.

    2. redshift=0 is passed to mag_spectrum because `lam_gal` is ALREADY
       rest frame -- this gives the rest-frame magnitude directly.

    M_sun_AB=4.64 is the SDSS r-band solar absolute AB magnitude
    stolen from Blanton & Roweis (2007).

    Returns L/Lsun, or nan (with a printed warning) if anything fails --
    e.g. if the fitted rest-frame range does not fully cover the requested
    band (this happens for the highest-redshift galaxies in this sample,
    where SDSS/r falls partly outside DESI's fixed observed coverage,
    which maps to a narrower rest-frame range at higher z).
    """
    if _COSMO is None:
        if verbose:
            print(f"[{tag}] mstar skipped -- astropy not installed")
        return np.nan

    d_L_pc = _COSMO.luminosity_distance(z).to(u.pc).value

    flux_orig = galaxy_normalized * norm
    if flux_unit_1e17:
        flux_orig = flux_orig * 1e-17

    try:
        m_AB = float(np.atleast_1d(util.mag_spectrum(
            lam_gal, flux_orig, bands=band, redshift=0,
            system="AB", quiet=True))[0])
    except Exception as exc:
        if verbose:
            print(f"[{tag}] mstar skipped -- mag_spectrum failed: {exc!r}")
        return np.nan

    M_abs = m_AB - 5 * np.log10(d_L_pc / 10.0)
    return 10 ** (-0.4 * (M_abs - M_sun_AB))

def get_weights_and_masses(z,pp,sps,combo,prep,compute_mstar=True, verbose=True):
    light_weights = pp.weights[~combo["gas_component"]].reshape(combo["reg_dim"])
    light_weights = light_weights / light_weights.sum()
    logage_lw, metal_lw = sps.mean_age_metal(light_weights, quiet=True)

    mass_weights = light_weights / sps.flux
    mass_weights = mass_weights / mass_weights.sum()
    logage_mw, metal_mw = sps.mean_age_metal(mass_weights, quiet=True)

    ml = sps.mass_to_light(light_weights, band="SDSS/r", quiet=True)
    ml_formed = formed_mass_to_light(sps,light_weights, band="SDSS/r", quiet=True)
        
    mstar = np.nan; mstar_formed = np.nan
    if compute_mstar:
        l_over_lsun = rest_frame_L_over_Lsun(
            prep["lam_gal"], prep["galaxy"], prep["norm"], z,
            verbose=verbose)
        if np.isfinite(l_over_lsun):
            mstar = float(ml) * l_over_lsun
            mstar_formed = float(ml_formed) * l_over_lsun

    return light_weights, mass_weights, logage_lw, metal_lw, logage_mw, metal_mw, mstar, mstar_formed, ml
            
def get_sfr(age_grid,mass_weights,mstar_formed):

    # Extrapolate the outer edges symmetrically - weird claude crap to give age deltas same length as weights
    log_age = np.log10(age_grid[:, 0])
    log_edges_inner = (log_age[:-1] + log_age[1:]) / 2
    first_edge = log_age[0] - (log_age[1] - log_age[0]) / 2
    last_edge  = log_age[-1] + (log_age[-1] - log_age[-2]) / 2
    log_edges = np.concatenate([[first_edge], log_edges_inner, [last_edge]])
    age_edges = 10**log_edges   # yr, length N+1 -- one more than the number of ages
    delta_t = np.diff(age_edges)

    sfh = np.sum(mass_weights,axis=1)*mstar_formed / (delta_t*1e9)
    log_sfr = [
        np.log10(np.sum(sfh[delta_t < 0.01])/len(sfh[delta_t < 0.01])),
        np.log10(np.sum(sfh[delta_t < 0.03])/len(sfh[delta_t < 0.03])),
        np.log10(np.sum(sfh[delta_t < 0.1])/len(sfh[delta_t < 0.1]))
    ] # 10, 30, 100 Myr averages

    
    return np.log10(sfh),log_sfr

# ======================================================================
# The fit
# ======================================================================
def fit_galaxy(galaxy_dict, sps_name="galaxev", edge_trim=50.0,
               phot=None,
               tie_balmer=True, fit_dust=True, tune_regul=True,
               compute_mstar=True, nboot=0, seed=0, verbose=False):
    """
    Fit one galaxy: stars + gas simultaneously.

    Parameters
    ----------
    galaxy_dict : one entry from load_desi_pickle().
    sps_name : SSP library ("galaxev", "emiles", "fsps").
    edge_trim : Angstroms trimmed off each edge of this galaxy's own
        de-redshifted range before fitting (see prepare_galaxy). The FITTED
        range itself is NOT fixed -- it is whatever this galaxy's own
        redshift maps DESI's fixed observed coverage into, so it differs
        galaxy to galaxy (wider for low-z, narrower for high-z).
    tie_balmer : tie Balmer ratios to Case B (see prepare_gas_templates).
    fit_dust : fit reddening for stars and gas. Gas reddening is only
        meaningful with tie_balmer=True, so it is disabled otherwise.
    tune_regul : run the regularization search. If False, regul=0.
    compute_mstar : compute an absolute stellar mass (see
        rest_frame_L_over_Lsun for the assumptions this relies on).
    nboot : bootstrap iterations for uncertainties. 0 disables it.

    Returns a FitResult.
    """
    name, z = galaxy_dict["name"], galaxy_dict["z"]
    t0 = time.time()
    if verbose:
        print(f"\n=== {name}  z={z:.4f} ===")

    try:
        # 1. Degrade the galaxy to the template resolution.
        flux_deg, noise_deg, fwhm_target_rest = degrade_to_template_res(
            galaxy_dict["lam"], galaxy_dict["flux"], galaxy_dict["noise"],
            z, sps_name, verbose=verbose, data_fwhm = galaxy_dict['wave_fwhm'])

        # 2. De-redshift, trim (this galaxy's own full range minus edge_trim
        #    on each side -- NOT a fixed window shared across galaxies), log-rebin.
        prep = prepare_galaxy(galaxy_dict["lam"], flux_deg, noise_deg, z,
                              edge_trim=edge_trim, verbose=verbose)
        wave_range = prep["wave_range"]   # this galaxy's actual fitted range

        # 3. Templates. The galaxy now sits at the templates' own resolution,
        #    so pass that as fwhm_gal (no further convolution results).
        lam_gal = prep["lam_gal"]
        fwhm_gal = {"lam": galaxy_dict["lam"] / (1.0 + z),
                    "fwhm": fwhm_target_rest}

        # Margin beyond wave_range for the velocity shift pPXF is allowed to
        # apply. The floor/ceiling here are only a safety net against
        # pathological values (e.g. non-positive wavelengths) -- they must
        # NEVER be tighter than wave_range itself, since templates have to
        # cover at least the galaxy's own data or pPXF raises
        # "TEMPLATES length cannot be smaller than GALAXY". GALAXEV's own
        # file spans 99-50000 A, so 100/49000 is generous but genuinely
        # non-binding for anything this dataset can produce -- unlike the
        # old fixed 3000/10500 clip, which WAS binding (and wrong) once
        # wave_range started varying per galaxy instead of being fixed at
        # (3600, 7400).
        lam_range_temp = [
            max(100.0, wave_range[0] - 500.0),
            min(49000.0, 10200 + 500) # max wavelength of the DECam z-band transmission function
            #wave_range[1] + 500.0)
        ]

        sps = prepare_templates(prep["velscale"], fwhm_gal, z, sps_name=sps_name,
                                lam_range_temp=lam_range_temp, verbose=verbose)

        gas_templates, gas_names, _ = prepare_gas_templates(
            sps, np.array([lam_range_temp[0], lam_range_temp[1]]), fwhm_gal,
            tie_balmer=tie_balmer, verbose=verbose)

        combo = combine_templates(sps, gas_templates, gas_names)

        # 4. Prepare the photometry dictionary
        # and figure out a scaling between spectra and photometry
        if phot is not None:
            phot,spec_scale = prepare_photometry(
                phot, combo, z,
                prep["lam_gal"]*(1+z), prep['galaxy'],
                sps.lam_temp, prep, sps
            )
            prep['galaxy'] *= spec_scale

        # 5. Good pixels: prep["galaxy"] is ALREADY trimmed to exactly
        #    wave_range in prepare_galaxy, so every pixel here is fair game --
        #    no further wavelength filtering needed.
        goodpixels = np.arange(lam_gal.size)

        # 6. Regularization.
        if tune_regul:
            regul, prep = tune_regularization(prep, phot, sps, combo, goodpixels,
                                              verbose=verbose)
        else:
            regul = 0.0

        # 6. Final fit.
        start, bounds, moments = _start_and_bounds()
        kwargs = dict(moments=moments, degree=-1, mdegree=-1, bounds=bounds,
                      component=combo["component"],
                      gas_component=combo["gas_component"],
                      gas_names=combo["gas_names"], reg_dim=combo["reg_dim"],
                      lam=prep["lam_gal"], lam_temp=sps.lam_temp,
                      goodpixels=goodpixels, regul=regul, quiet=True, phot=phot)
        if fit_dust:
            kwargs["reddening"] = 0.0
            if tie_balmer:
                kwargs["gas_reddening"] = 0.0

        pp = ppxf(combo["templates"], prep["galaxy"], prep["noise"],
                  prep["velscale"], start, **kwargs)
        
        # 7. Unpack.
        light_weights, mass_weights, logage_lw, metal_lw, logage_mw, metal_mw, mstar, mstar_formed, ml = \
            get_weights_and_masses(z,pp,sps,combo,prep,compute_mstar=compute_mstar,verbose=verbose)
        log_sfh,log_sfr = get_sfr(sps.age_grid,mass_weights,mstar_formed)

        gas_flux, gas_flux_err = {}, {}
        if getattr(pp, "gas_flux", None) is not None:
            for nm, fl, er in zip(combo["gas_names"], pp.gas_flux,
                                  pp.gas_flux_error):
                gas_flux[nm] = float(fl)
                gas_flux_err[nm] = float(er)

        v, sigma = float(pp.sol[0][0]), float(pp.sol[0][1])

        res = FitResult(
            name=name, z=z, ok=True,
            chi2=float(pp.chi2), regul_used=float(regul),
            logage_lw=float(logage_lw), logage_mw=float(logage_mw),
            metal_lw=float(metal_lw), metal_mw=float(metal_mw),
            ml=float(ml), mstar=mstar,
            av_stars=float(getattr(pp, "reddening", np.nan) or np.nan),
            av_gas=float(getattr(pp, "gas_reddening", np.nan) or np.nan),
            v=v, sigma=sigma, kin_at_bound=_at_bound(v, sigma),
            v_gas=float(pp.sol[1][0]), sigma_gas=float(pp.sol[1][1]),
            gas_flux=gas_flux, gas_flux_err=gas_flux_err,
            weights=light_weights, age_grid=sps.age_grid,
            metal_grid=sps.metal_grid,
            lam_gal=prep["lam_gal"], galaxy=prep["galaxy"],
            bestfit=pp.bestfit, gas_bestfit=getattr(pp, "gas_bestfit", None),
            log_sfh = log_sfh,
            log_sfr_10myr = log_sfr[0],
            log_sfr_30myr = log_sfr[1],
            log_sfr_100myr = log_sfr[2]
        )

        # 8. Optional bootstrap.
        if nboot > 0:
            logage_lw_cis,metal_lw_cis,logage_mw_cis,metal_mw_cis,mstar_cis,log_sfr_10myr_cis,log_sfr_30myr_cis,log_sfr_100myr_cis,\
                v_cis,sigma_cis,av_stars_cis,av_gas_cis,v_gas_cis,sigma_gas_cis = \
                _bootstrap(
                    z, pp, prep, phot, sps, combo, goodpixels, nboot, seed, verbose
                )

            # update all values
            res.logage_lw_16,res.logage_lw,res.logage_lw_84 = logage_lw_cis[:]
            res.logage_mw_16,res.logage_mw,res.logage_mw_84 = logage_mw_cis[:]
            res.metal_lw_16,res.metal_lw,res.metal_lw_84 = metal_lw_cis[:]
            res.metal_mw_16,res.metal_mw,res.metal_mw_84 = metal_mw_cis[:]
            res.mstar_16,res.mstar,res.mstar_84 = mstar_cis[:]
            res.av_stars_16,res.av_stars,res.av_stars_84 = av_stars_cis[:]
            res.av_gas_16,res.av_gas,res.av_gas_84 = av_gas_cis[:]
            res.v_16,res.v,res.v_84 = v_cis[:]
            res.sigma_16,res.sigma,res.sigma_84 = sigma_cis[:]
            res.v_gas_16,res.v_gas,res.v_gas_84 = v_gas_cis[:]
            res.sigma_gas_16,res.sigma_gas,res.sigma_gas_84 = sigma_gas_cis[:]
            res.log_sfr_10myr_16,res.log_sfr_10myr,res.log_sfr_10myr_84 = log_sfr_10myr_cis[:]
            res.log_sfr_30myr_16,res.log_sfr_30myr,res.log_sfr_30myr_84 = log_sfr_30myr_cis[:]
            res.log_sfr_100myr_16,res.log_sfr_100myr,res.log_sfr_100myr_84 = log_sfr_100myr_cis[:]

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
            print("    --- full traceback ---")
            traceback.print_exc()
            print("    -----------------------")
        return FitResult(name=name, z=z, ok=False, error_msg=repr(exc)), None


def _bootstrap(z, pp, prep, phot, sps, combo, goodpixels, nboot, seed, verbose):
    """Wild bootstrap for age/metallicity uncertainties."""
    rng = np.random.default_rng(seed)
    resid = prep["galaxy"] - pp.bestfit
    warm = [list(s) for s in pp.sol]
    _, bounds, moments = _start_and_bounds()

    #ages, metals = [], []
    logage_lws, metal_lws, logage_mws, metal_mws, mstars, sfrs_10myr, sfrs_30myr, sfrs_100myr,\
        vs, sigmas, avs_stars, avs_gas, vs_gas, sigmas_gas  = \
        [], [], [], [], [], [], [], [], [], [], [], [], [], []
    for i in range(nboot):
        pert = pp.bestfit + resid * rng.choice([-1.0, 1.0], size=resid.size)
        ppb = ppxf(combo["templates"], pert, prep["noise"], prep["velscale"],
                   warm, moments=moments, degree=-1, mdegree=-1, bounds=bounds,
                   component=combo["component"],
                   gas_component=combo["gas_component"],
                   gas_names=combo["gas_names"], reg_dim=combo["reg_dim"],
                   lam=prep["lam_gal"], lam_temp=sps.lam_temp,
                   goodpixels=goodpixels, quiet=True, phot=phot)
        w = ppb.weights[~combo["gas_component"]].reshape(combo["reg_dim"])
        w = w / w.sum()
        a, m = sps.mean_age_metal(w, quiet=True)

        light_weights, mass_weights, logage_lw, metal_lw, logage_mw, metal_mw, mstar, mstar_formed, ml = \
            get_weights_and_masses(z,pp,sps,combo,prep,verbose=False)
        log_sfh,log_sfr = get_sfr(sps.age_grid,mass_weights,mstar_formed)

        logage_lws.append(logage_lw)
        metal_lws.append(metal_lw)
        logage_mws.append(logage_mw)
        metal_mws.append(metal_mw)
        mstars.append(mstar)
        sfrs_10myr.append(log_sfr[0])
        sfrs_30myr.append(log_sfr[1])
        sfrs_100myr.append(log_sfr[2])
        vs.append(pp.sol[0][0])
        sigmas.append(pp.sol[0][1])
        avs_stars.append(float(getattr(pp, "reddening", np.nan) or np.nan))
        avs_gas.append(float(getattr(pp, "gas_reddening", np.nan) or np.nan))
        vs_gas.append(float(pp.sol[1][0]))
        sigmas_gas.append(float(pp.sol[1][1]))

        
        if verbose and (i + 1) % max(1, nboot // 4) == 0:
            print(f"      bootstrap {i + 1}/{nboot}")

    logage_lw_cis = np.percentile(logage_lws,[16,50,84])
    metal_lw_cis = np.percentile(metal_lws,[16,50,84])
    logage_mw_cis = np.percentile(logage_mws,[16,50,84])
    metal_mw_cis = np.percentile(metal_mws,[16,50,84])
    mstar_cis = np.percentile(mstars,[16,50,84])
    sfr_10myr_cis = np.percentile(sfrs_10myr,[16,50,84])
    sfr_30myr_cis = np.percentile(sfrs_30myr,[16,50,84])
    sfr_100myr_cis = np.percentile(sfrs_100myr,[16,50,84])
    v_cis = np.percentile(vs,[16,50,84])
    sigma_cis = np.percentile(sigmas,[16,50,84])
    av_stars_cis = np.percentile(avs_stars,[16,50,84])
    av_gas_cis = np.percentile(avs_gas,[16,50,84])
    v_gas_cis = np.percentile(vs_gas,[16,50,84])
    sigma_gas_cis = np.percentile(sigmas_gas,[16,50,84])

    
    return logage_lw_cis,metal_lw_cis,logage_mw_cis,metal_mw_cis,mstar_cis,\
        sfr_10myr_cis,sfr_30myr_cis,sfr_100myr_cis,v_cis,sigma_cis,av_stars_cis,\
        av_gas_cis,v_gas_cis,sigma_gas_cis



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

    array_fields = ("weights", "age_grid", "metal_grid", "lam_gal", "galaxy",
                    "bestfit", "gas_bestfit", "gas_flux", "gas_flux_err", "log_sfh", "log_sfr")

    rows = []
    for r in results:
        d = asdict(r)
        for key in array_fields:
            d.pop(key, None)
        for ln in line_subset:
            d[f"flux_{ln}"] = r.gas_flux.get(ln, "")
            d[f"fluxerr_{ln}"] = r.gas_flux_err.get(ln, "")
        rows.append(d)
    return rows, line_subset
