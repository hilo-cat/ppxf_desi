import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================

INPUT_CSV = "MPA_matches_phot.csv"

BANDS = ["G", "R", "Z"]

# Number of nearest neighbors in magnitude used to estimate noise
N_NEIGHBORS = 10

# Output file with estimated uncertainties added
OUTPUT_CSV = "MPA_matches_phot_noise.csv"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def flux_to_mag(flux):
    """
    Convert nanomaggy flux to AB magnitude.

    m_AB = 22.5 - 2.5 log10(flux_nanomaggy)

    Returns NaN for non-positive or non-finite flux.
    """
    flux = np.asarray(flux, dtype=float)

    mag = np.full_like(flux, np.nan, dtype=float)

    good = np.isfinite(flux) & (flux > 0)

    mag[good] = 22.5 - 2.5 * np.log10(flux[good])

    return mag


def ivar_to_noise(ivar):
    """
    Convert inverse variance to 1-sigma flux uncertainty.

    sigma = 1 / sqrt(ivar)

    Returns NaN for ivar <= 0 or non-finite ivar.
    """
    ivar = np.asarray(ivar, dtype=float)

    noise = np.full_like(ivar, np.nan, dtype=float)

    good = np.isfinite(ivar) & (ivar > 0)

    noise[good] = 1.0 / np.sqrt(ivar[good])

    return noise


def estimate_noise_from_mag(
    target_mag,
    training_mag,
    training_noise,
    n_neighbors=10
):
    """
    Estimate flux uncertainty for one target magnitude.

    Finds the N closest training galaxies in magnitude
    and returns the median of their flux uncertainties.
    """

    target_mag = float(target_mag)

    training_mag = np.asarray(training_mag, dtype=float)
    training_noise = np.asarray(training_noise, dtype=float)

    good = (
        np.isfinite(training_mag)
        & np.isfinite(training_noise)
        & (training_noise > 0)
    )

    training_mag = training_mag[good]
    training_noise = training_noise[good]

    if len(training_mag) == 0:
        return np.nan

    # Do not ask for more neighbors than actually exist
    n_use = min(n_neighbors, len(training_mag))

    distance = np.abs(training_mag - target_mag)

    idx = np.argsort(distance)[:n_use]

    return np.median(training_noise[idx])


def leave_one_out_validation(
    valid_mag,
    valid_noise,
    n_neighbors=10
):
    """
    Leave-one-out validation.

    For each valid galaxy:
      1. Remove it from the training sample
      2. Predict its noise from the other galaxies
      3. Compare predicted noise to its real noise
    """

    valid_mag = np.asarray(valid_mag, dtype=float)
    valid_noise = np.asarray(valid_noise, dtype=float)

    predicted = np.full(len(valid_mag), np.nan)

    for i in range(len(valid_mag)):

        use = np.ones(len(valid_mag), dtype=bool)
        use[i] = False

        predicted[i] = estimate_noise_from_mag(
            target_mag=valid_mag[i],
            training_mag=valid_mag[use],
            training_noise=valid_noise[use],
            n_neighbors=n_neighbors
        )

    return predicted


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_CSV)

print(f"Loaded {len(df)} objects from:")
print(INPUT_CSV)


# ============================================================
# PROCESS EACH BAND
# ============================================================

for band in BANDS:

    print("\n" + "=" * 70)
    print(f"{band}-BAND")
    print("=" * 70)

    flux_col = f"FLUX_{band}"
    ivar_col = f"FLUX_IVAR_{band}"

    # --------------------------------------------------------
    # Basic input arrays
    # --------------------------------------------------------

    flux = pd.to_numeric(
        df[flux_col],
        errors="coerce"
    ).to_numpy(float)

    ivar = pd.to_numeric(
        df[ivar_col],
        errors="coerce"
    ).to_numpy(float)

    mag = flux_to_mag(flux)
    noise = ivar_to_noise(ivar)

    # Add magnitude to dataframe for inspection
    df[f"MAG_{band}"] = mag

    # --------------------------------------------------------
    # Define valid training sample
    # --------------------------------------------------------

    valid = (
        np.isfinite(flux)
        & (flux > 0)
        & np.isfinite(ivar)
        & (ivar > 0)
        & np.isfinite(mag)
        & np.isfinite(noise)
    )

    missing_ivar = (
        np.isfinite(flux)
        & (flux > 0)
        & np.isfinite(mag)
        & (
            ~np.isfinite(ivar)
            | (ivar <= 0)
        )
    )

    valid_mag = mag[valid]
    valid_noise = noise[valid]

    print(f"Valid IVAR objects:       {valid.sum()}")
    print(f"Need imputed noise:       {missing_ivar.sum()}")

    # --------------------------------------------------------
    # STEP 1:
    # VALIDATE MAGNITUDE -> NOISE RELATION
    # --------------------------------------------------------

    predicted_noise = leave_one_out_validation(
        valid_mag=valid_mag,
        valid_noise=valid_noise,
        n_neighbors=N_NEIGHBORS
    )

    good_validation = (
        np.isfinite(predicted_noise)
        & np.isfinite(valid_noise)
        & (valid_noise > 0)
    )

    actual = valid_noise[good_validation]
    predicted = predicted_noise[good_validation]

    ratio = predicted / actual

    print("\nLeave-one-out validation")
    print("------------------------")
    print(f"N validated: {len(ratio)}")

    print(
        "Median predicted / actual noise:",
        np.median(ratio)
    )

    print(
        "16th-84th percentile ratio:",
        np.percentile(ratio, [16, 84])
    )

    print(
        "Median |log10(predicted/actual)|:",
        np.median(
            np.abs(np.log10(ratio))
        )
    )

    # --------------------------------------------------------
    # VALIDATION PLOT 1:
    # actual vs predicted
    # --------------------------------------------------------

    plt.figure(figsize=(6, 6))

    plt.scatter(
        actual,
        predicted,
        s=25,
        alpha=0.7
    )

    minimum = min(
        np.min(actual),
        np.min(predicted)
    )

    maximum = max(
        np.max(actual),
        np.max(predicted)
    )

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--"
    )

    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel(
        r"Actual $\sigma_F$ [nanomaggies]"
    )

    plt.ylabel(
        r"Predicted $\sigma_F$ [nanomaggies]"
    )

    plt.title(
        f"{band}-band leave-one-out validation"
    )

    plt.tight_layout()

    plt.savefig(
        f"noise_validation_{band}.png",
        dpi=200
    )

    plt.close()

    # --------------------------------------------------------
    # VALIDATION PLOT 2:
    # magnitude vs true noise
    # --------------------------------------------------------

    plt.figure(figsize=(7, 5))

    plt.scatter(
        valid_mag,
        valid_noise,
        s=25,
        alpha=0.7,
        label="Measured IVAR"
    )

    plt.yscale("log")

    plt.xlabel(
        f"{band.lower()} magnitude"
    )

    plt.ylabel(
        r"$\sigma_F$ [nanomaggies]"
    )

    plt.title(
        f"{band}-band magnitude vs flux uncertainty"
    )

    plt.tight_layout()

    plt.savefig(
        f"noise_vs_mag_{band}.png",
        dpi=200
    )

    plt.close()

    # --------------------------------------------------------
    # STEP 2:
    # ESTIMATE NOISE FOR ZERO-IVAR OBJECTS
    # --------------------------------------------------------

    estimated_noise = np.full(
        len(df),
        np.nan
    )

    for i in np.where(missing_ivar)[0]:

        estimated_noise[i] = estimate_noise_from_mag(
            target_mag=mag[i],
            training_mag=valid_mag,
            training_noise=valid_noise,
            n_neighbors=N_NEIGHBORS
        )

    # --------------------------------------------------------
    # STEP 3:
    # CREATE FINAL NOISE COLUMN
    #
    # Use real catalog noise where available.
    # Use empirical noise only for IVAR <= 0.
    # --------------------------------------------------------

    final_noise = noise.copy()

    final_noise[missing_ivar] = (
        estimated_noise[missing_ivar]
    )

    df[f"FLUX_NOISE_{band}"] = final_noise

    df[f"FLUX_NOISE_IMPUTED_{band}"] = (
        missing_ivar
    )

    # --------------------------------------------------------
    # Print imputed objects
    # --------------------------------------------------------

    print("\nObjects with imputed noise")
    print("--------------------------")

    for i in np.where(missing_ivar)[0]:

        targetid = (
            df.iloc[i]["TARGETID"]
            if "TARGETID" in df.columns
            else i
        )

        print(
            f"TARGETID={targetid}  "
            f"mag={mag[i]:.3f}  "
            f"flux={flux[i]:.4f}  "
            f"estimated sigma={estimated_noise[i]:.5f}"
        )


# ============================================================
# ADD A SINGLE GALAXY-LEVEL FLAG
# ============================================================

df["PHOT_NOISE_IMPUTED"] = (
    df[
        [
            "FLUX_NOISE_IMPUTED_G",
            "FLUX_NOISE_IMPUTED_R",
            "FLUX_NOISE_IMPUTED_Z"
        ]
    ]
    .any(axis=1)
)


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_CSV,
    index=False
)

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

print(
    f"Saved photometry with noise estimates to:\n{OUTPUT_CSV}"
)

print(
    "\nNumber of galaxies with any imputed photometric noise:",
    df["PHOT_NOISE_IMPUTED"].sum()
)