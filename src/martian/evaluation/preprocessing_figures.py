"""Figures 4.5-4.8 -- preprocessing and spatiotemporal-structure figures for
Chapter 4 (CHAPTER_4_PARAMETER_VALUES.md open item 3, implemented
2026-08-12). None of these existed anywhere in this repository before this
module: `evaluation/regression_figures.py` and
`evaluation/classification_figures.py` cover only per-model diagnostic
plots (obs-vs-predicted, residuals, confusion matrices, PR/ROC curves), not
the raw-data preprocessing figures the chapter draft's [FIGURE TO
REGENERATE FROM LEVEL 3 DATA] markers describe.

Each `figure_4_*` function returns (fig, source_df), mirroring
evaluation/regression_figures.py's convention -- the caller saves both via
evaluation.figure_provenance.save_figure_with_provenance. Data-computation
helpers (matrix construction, ACF/PACF, semivariogram binning) are exposed
separately from plotting so they can be unit-tested without matplotlib.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from preprocessing.grid import GridDefinition, bands_from_cell_id

MARS_RADIUS_KM = 3389.5  # mean Mars radius (IAU), used for great-circle cell-centroid distances


# ---------------------------------------------------------------------------
# Representative-cell selection
# ---------------------------------------------------------------------------

def select_representative_cell(
    aggregated_df: pd.DataFrame,
    split_series: pd.Series,
    *,
    excluded_cell_ids: tuple[int, ...] = (),
) -> int:
    """The most-observed cell_id in the TRAIN split, excluding any cell_id
    in `excluded_cell_ids` -- see configs/preprocessing_figures.yaml for why
    cell 1296 is excluded by default (a real, previously undocumented
    data-quality finding, not an arbitrary exclusion)."""
    train_mask = split_series.to_numpy() == "train"
    train_df = aggregated_df.loc[train_mask]
    train_df = train_df.loc[~train_df["cell_id"].isin(excluded_cell_ids)]
    if train_df.empty:
        raise ValueError("No train-split rows remain after excluding excluded_cell_ids")
    counts = train_df.groupby("cell_id").size().sort_values(ascending=False)
    return int(counts.index[0])


def cell_centroid_lat_lon(cell_id_value, grid: GridDefinition) -> tuple[np.ndarray, np.ndarray]:
    """True grid-geometry cell-centre latitude/longitude from cell_id (not
    the mean observed lat/lon within the cell, which drifts slightly with
    sub-cell footprint position) -- consistent with how
    labeling/catalogue_validation.py treats cell geometry."""
    lat_band, lon_band = bands_from_cell_id(cell_id_value, grid)
    lat = -90.0 + (np.asarray(lat_band, dtype=float) + 0.5) * grid.resolution_deg
    lon = (np.asarray(lon_band, dtype=float) + 0.5) * grid.resolution_deg
    return lat, lon


def haversine_distance_km(lat1, lon1, lat2, lon2, radius_km: float = MARS_RADIUS_KM):
    lat1r, lon1r, lat2r, lon2r = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    return 2.0 * radius_km * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


# ---------------------------------------------------------------------------
# Figure 4.5 -- raw vs QC'd vs retained-observation-count, representative cell
# ---------------------------------------------------------------------------

def figure_4_5_raw_qc_comparison(
    raw_qc_df: pd.DataFrame,
    aggregated_cell_df: pd.DataFrame,
    filled_cell_df: pd.DataFrame,
    *,
    cell_id: int,
    mars_year: int,
):
    """`raw_qc_df`: every raw retrieval for ONE cell, with `qc_status`
    ('accepted'/'rejected') -- see generate_figure_4_5_raw_data.py.
    `aggregated_cell_df`/`filled_cell_df`: that same cell's rows from
    data/aggregated_cell_sol.csv / data/trailing_filled_cell_sol.csv.

    `raw_qc_df` is explicitly restricted to `mars_year` HERE, not left to
    the caller: sol numbering resets every Mars Year, so plotting two years'
    raw retrievals on one shared sol axis without filtering would silently
    overlay unrelated observations at the same x-position (a real bug
    caught during this figure's own 2026-08-12 generation -- the initial
    run mixed MY36 and MY37 retrievals for this reason)."""
    raw_sorted = raw_qc_df.loc[raw_qc_df["mars_year"] == mars_year].sort_values("sol")
    accepted = raw_sorted.loc[raw_sorted["qc_status"] == "accepted"]
    rejected = raw_sorted.loc[raw_sorted["qc_status"] == "rejected"]

    agg_sorted = aggregated_cell_df.sort_values("sol")
    filled_sorted = filled_cell_df.sort_values("sol")
    filled_only = filled_sorted.loc[filled_sorted["interpolation_flag"] == True]  # noqa: E712

    counts = raw_sorted.groupby(["sol", "qc_status"]).size().unstack(fill_value=0)
    for col in ("accepted", "rejected"):
        if col not in counts.columns:
            counts[col] = 0

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    ax = axes[0]
    ax.scatter(accepted["sol"], accepted["dust_optical_depth"], s=14, color="steelblue", label="raw retrieval, QC-accepted")
    ax.scatter(rejected["sol"], rejected["dust_optical_depth"], s=14, color="lightcoral", marker="x", label="raw retrieval, QC-rejected")
    ax.set_ylabel("Raw CDOD (per retrieval)")
    ax.set_title(f"Cell {cell_id}: raw per-retrieval CDOD, coloured by QC outcome")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(agg_sorted["sol"], agg_sorted["dust_optical_depth"], "o-", color="steelblue", markersize=4, label="QC-accepted, cell-sol aggregated")
    ax.plot(filled_only["sol"], filled_only["dust_optical_depth"], "s", color="darkorange", markersize=5, label="trailing-filled (gap-treated, LOCF)")
    ax.set_ylabel("CDOD (cell-sol)")
    ax.set_title("QC'd and gap-treated cell-sol series")
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.bar(counts.index, counts["accepted"], color="steelblue", label="accepted")
    ax.bar(counts.index, counts["rejected"], bottom=counts["accepted"], color="lightcoral", label="rejected")
    ax.set_xlabel("Sol")
    ax.set_ylabel("Retrieval count")
    ax.set_title("Retained (and rejected) observation count per sol")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2)  # reserve room for save_figure_with_provenance's stamped caption block

    source_df = raw_sorted[["sol", "mars_year", "dust_optical_depth", "qc_status"]].copy()
    return fig, source_df


# ---------------------------------------------------------------------------
# Figure 4.6 -- missingness map, gap-fill map, pre/post distribution vs climatology
# ---------------------------------------------------------------------------

def compute_missingness_matrix(aggregated_df: pd.DataFrame, cell_ids: list[int], sol_min: int, sol_max: int) -> pd.DataFrame:
    """cell_id (rows) x sol (columns) -> 1 if a genuine observation exists,
    else 0. `aggregated_df` must already be restricted to one mars_year."""
    sols = np.arange(sol_min, sol_max + 1)
    observed = aggregated_df.loc[aggregated_df["cell_id"].isin(cell_ids), ["cell_id", "sol"]].drop_duplicates()
    matrix = pd.DataFrame(0, index=cell_ids, columns=sols)
    for cid, sol in observed.itertuples(index=False):
        if sol in matrix.columns:
            matrix.loc[cid, sol] = 1
    return matrix


def compute_gapfill_state_matrix(filled_df: pd.DataFrame, cell_ids: list[int], sol_min: int, sol_max: int) -> pd.DataFrame:
    """cell_id x sol -> 2 (genuine observation), 1 (trailing-filled), 0
    (absent -- gap wider than the configured max, or never observed)."""
    sols = np.arange(sol_min, sol_max + 1)
    matrix = pd.DataFrame(0, index=cell_ids, columns=sols)
    subset = filled_df.loc[filled_df["cell_id"].isin(cell_ids)]
    for cid, sol, is_observed in subset[["cell_id", "sol", "is_observed"]].itertuples(index=False):
        if sol in matrix.columns:
            matrix.loc[cid, sol] = 2 if is_observed else 1
    return matrix


def figure_4_6_missingness_gapfill_distribution(
    aggregated_df: pd.DataFrame,
    filled_df: pd.DataFrame,
    mcd_provider,
    *,
    representative_cell_id: int,
    grid: GridDefinition,
    mars_year: int,
    sol_min: int,
    sol_max: int,
):
    lat_band, _ = bands_from_cell_id(representative_cell_id, grid)
    n_lon = grid.n_lon_bands
    band_cell_ids = [int(lat_band) * n_lon + lon_b for lon_b in range(n_lon)]

    my_agg = aggregated_df.loc[aggregated_df["mars_year"] == mars_year]
    my_filled = filled_df.loc[filled_df["mars_year"] == mars_year]

    missingness = compute_missingness_matrix(my_agg, band_cell_ids, sol_min, sol_max)
    gapfill = compute_gapfill_state_matrix(my_filled, band_cell_ids, sol_min, sol_max)

    fig = plt.figure(figsize=(13, 10))
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 1, 2)

    ax1.imshow(missingness.to_numpy(), aspect="auto", cmap="Greys", interpolation="none")
    ax1.set_title(f"(a) Cell-sol missingness map -- latitude band containing cell {representative_cell_id}")
    ax1.set_xlabel("Sol")
    ax1.set_ylabel("Longitude cell (within band)")

    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(["white", "darkorange", "steelblue"])
    ax2.imshow(gapfill.to_numpy(), aspect="auto", cmap=cmap, vmin=0, vmax=2, interpolation="none")
    ax2.set_title("(b) Trailing-fill map (white=absent, orange=filled, blue=genuine obs.)")
    ax2.set_xlabel("Sol")
    ax2.set_ylabel("Longitude cell (within band)")

    pre_values = my_agg["dust_optical_depth"].to_numpy(dtype=float)
    post_values = my_filled["dust_optical_depth"].to_numpy(dtype=float)
    clim = mcd_provider.query(
        latitude=my_agg["latitude"].to_numpy(),
        longitude=my_agg["longitude"].to_numpy(),
        solar_longitude_ls=my_agg["solar_longitude_ls"].to_numpy(),
        local_true_solar_time=my_agg["local_true_solar_time"].to_numpy(),
        variable="dust_optical_depth",
    )
    clim_mean = float(np.nanmean(clim["mu"].to_numpy()))

    bins = np.linspace(0, max(np.nanpercentile(pre_values, 99), np.nanpercentile(post_values, 99)), 40)
    ax3.hist(pre_values, bins=bins, alpha=0.55, color="steelblue", label=f"pre-fill (genuine obs. only, n={len(pre_values)})", density=True)
    ax3.hist(post_values, bins=bins, alpha=0.45, color="darkorange", label=f"post-fill (incl. LOCF, n={len(post_values)})", density=True)
    ax3.axvline(clim_mean, color="black", linestyle="--", linewidth=1.5, label=f"Montabone climatology mean mu ({clim_mean:.3g}, sample-averaged)")
    ax3.set_xlabel("Column dust optical depth")
    ax3.set_ylabel("Density")
    ax3.set_title(f"(c) MY{mars_year} pre/post-fill CDOD distribution vs. gridded climatology")
    ax3.legend(fontsize=8)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2)  # reserve room for save_figure_with_provenance's stamped caption block

    source_df = pd.DataFrame(
        {
            "pre_fill_n": [len(pre_values)],
            "post_fill_n": [len(post_values)],
            "pre_fill_mean": [float(np.nanmean(pre_values))],
            "post_fill_mean": [float(np.nanmean(post_values))],
            "climatology_mean_sample_averaged": [clim_mean],
            "representative_lat_band": [int(lat_band)],
        }
    )
    return fig, source_df


# ---------------------------------------------------------------------------
# Figure 4.7 -- episode-centred composites, pooled ACF/PACF, semivariogram
# ---------------------------------------------------------------------------

def compute_episode_centred_composite(
    labeled_train_df: pd.DataFrame,
    episode_catalogue: pd.DataFrame,
    *,
    window_before: int,
    window_after: int,
    value_col: str = "dust_optical_depth_z",
) -> pd.DataFrame:
    """For each QUALIFYING storm_episode with an onset sol inside the
    training split, and each of its member cells, looks up `value_col` at
    every relative-sol offset in [-window_before, +window_after] from onset
    -- only where a genuine (non-interpolated) observation exists at that
    absolute (cell, mars_year, sol). Returns one row per offset: mean, std,
    median, q25, q75, n across all (episode, cell) contributions.

    Both mean/std AND median/q25/q75 are reported because the real
    dust_optical_depth_z distribution is heavy-tailed -- some (cell, Ls-bin)
    climatology cells have a genuinely small but non-degenerate sigma_clim
    (not caught by Eq 4.2's epsilon=1e-6 floor, which only guards against
    near-zero sigma), producing z-scores as extreme as ~2000 for a small
    number of real cell-sols (found 2026-08-12: 1,323 of 142,464 labeled
    rows, 0.93%, have |z|>50; see CHAPTER_4_PARAMETER_VALUES.md). This does
    not change any atmospheric_anomaly/storm_episode LABEL (the exceedance
    test z>tau=2.0 is a threshold crossing, unaffected by how far past 2.0
    z lands), but it does mean the mean/std of z is dominated by a handful
    of extreme rows and is not representative of a typical episode -- the
    figure itself plots median/IQR, not mean/std, for this reason."""
    qualifying = episode_catalogue.loc[episode_catalogue["qualifies"]]
    lookup = labeled_train_df.set_index(["cell_id", "mars_year", "sol"])[value_col]

    rows: list[dict] = []
    for _, ep in qualifying.iterrows():
        member_cells = labeled_train_df.loc[
            (labeled_train_df["episode_id"] == ep["episode_id"]) & (labeled_train_df["mars_year"] == ep["mars_year"]),
            "cell_id",
        ].unique()
        for cid in member_cells:
            for offset in range(-window_before, window_after + 1):
                key = (int(cid), int(ep["mars_year"]), int(ep["onset_sol"]) + offset)
                if key in lookup.index:
                    val = lookup.loc[key]
                    if isinstance(val, pd.Series):
                        val = val.iloc[0]
                    rows.append({"offset_sols": offset, "value": float(val)})

    if not rows:
        return pd.DataFrame(columns=["offset_sols", "mean", "std", "median", "q25", "q75", "n"])
    df = pd.DataFrame(rows)
    summary = (
        df.groupby("offset_sols")["value"]
        .agg(mean="mean", std="std", median="median", q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75), n="count")
        .reset_index()
    )
    return summary.sort_values("offset_sols").reset_index(drop=True)


def compute_pooled_lag_autocorrelation(
    df: pd.DataFrame,
    *,
    value_col: str,
    cell_col: str = "cell_id",
    mars_year_col: str = "mars_year",
    sol_col: str = "sol",
    max_lag: int,
) -> np.ndarray:
    """Pooled panel autocorrelation: for lag k=0..max_lag, averages
    (v_i - mean)(v_{i+k} - mean) over every (cell, mars_year) pair of
    genuine observations whose sols are EXACTLY k apart -- gaps are never
    bridged; a pair only contributes at lag k if sol_i+k truly exists for
    that cell in `df`. Returns an array of length max_lag+1, acf[0] == 1.0."""
    values = df[value_col].to_numpy(dtype=float)
    mean = np.nanmean(values)
    var = np.nanvar(values)

    lookup: dict[tuple[int, int, int], float] = {}
    for cid, my, sol, val in df[[cell_col, mars_year_col, sol_col, value_col]].itertuples(index=False, name=None):
        lookup[(int(cid), int(my), int(sol))] = float(val)

    acf = np.zeros(max_lag + 1)
    acf[0] = 1.0
    for lag in range(1, max_lag + 1):
        products = []
        for (cid, my, sol), val in lookup.items():
            if np.isnan(val):
                continue  # a NaN value_col entry (e.g. a genuinely undefined z-score)
                # must not poison this lag's aggregate -- np.mean over a list
                # containing even one NaN returns NaN regardless of how many
                # valid pairs it also contains; skip NaN endpoints explicitly
                # rather than relying on a nan-aware reducer applied too late.
            key2 = (cid, my, sol + lag)
            neighbour_val = lookup.get(key2)
            if neighbour_val is not None and not np.isnan(neighbour_val):
                products.append((val - mean) * (neighbour_val - mean))
        acf[lag] = (float(np.mean(products)) / var) if products and var > 0 else np.nan
    return acf


def durbin_levinson_pacf(acf: np.ndarray) -> np.ndarray:
    """Standard Durbin-Levinson recursion converting an ACF sequence
    (acf[0]==1) into the corresponding PACF sequence, treating the pooled
    panel ACF as if from a single stationary process -- a documented
    simplification (configs/preprocessing_figures.yaml)."""
    max_lag = len(acf) - 1
    pacf = np.zeros(max_lag + 1)
    pacf[0] = 1.0
    if max_lag == 0:
        return pacf
    phi = np.zeros((max_lag + 1, max_lag + 1))
    phi[1, 1] = acf[1]
    pacf[1] = acf[1]
    for k in range(2, max_lag + 1):
        num = acf[k] - sum(phi[k - 1, j] * acf[k - j] for j in range(1, k))
        den = 1.0 - sum(phi[k - 1, j] * acf[j] for j in range(1, k))
        phi[k, k] = num / den if den != 0 else np.nan
        for j in range(1, k):
            phi[k, j] = phi[k - 1, j] - phi[k, k] * phi[k - 1, k - j]
        pacf[k] = phi[k, k]
    return pacf


def compute_semivariogram(
    labeled_train_df: pd.DataFrame,
    grid: GridDefinition,
    *,
    distance_bins_km: list[float],
    latitude_bands: dict[str, tuple[float, float]],
    value_col: str = "dust_optical_depth_z",
) -> pd.DataFrame:
    """Same-sol spatial semivariogram gamma(h) = 0.5 * mean((z_i - z_j)^2),
    binned by great-circle distance h between cell centroids, computed
    separately per (latitude_band, Ls quartile). Only same-(mars_year, sol)
    pairs are compared -- a genuine spatial-structure test, not conflated
    with temporal change."""
    df = labeled_train_df.copy()
    lat, lon = cell_centroid_lat_lon(df["cell_id"].to_numpy(), grid)
    df["_centroid_lat"] = lat
    df["_centroid_lon"] = lon
    ls_quartile_edges = np.unique(np.nanpercentile(df["solar_longitude_ls"], [0, 25, 50, 75, 100]))
    if len(ls_quartile_edges) < 2:
        # Degenerate case (e.g. every row shares the same Ls, as in a small
        # test fixture): everything falls in a single "quartile" rather than
        # raising on non-unique bin edges.
        df["_ls_quartile"] = 1
    else:
        n_labels = len(ls_quartile_edges) - 1
        df["_ls_quartile"] = pd.cut(
            df["solar_longitude_ls"], bins=ls_quartile_edges, labels=list(range(1, n_labels + 1)), include_lowest=True
        )

    distance_bins_arr = np.asarray(distance_bins_km, dtype=float)
    n_bins = len(distance_bins_arr) - 1

    rows: list[dict] = []
    for band_name, (lat_lo, lat_hi) in latitude_bands.items():
        band_df = df.loc[(df["_centroid_lat"] >= lat_lo) & (df["_centroid_lat"] < lat_hi)]
        for quartile, quartile_df in band_df.groupby("_ls_quartile", observed=True):
            bin_sq_diffs: list[list[float]] = [[] for _ in range(n_bins)]
            # Vectorised per-sol pairwise distances/differences (numpy
            # broadcasting), NOT a Python-level combinations loop -- at real
            # scale (~600 train sols x ~100+ cells/sol) an itertools.combinations
            # loop with a scalar haversine call per pair would be minutes-to-
            # hours slower than this.
            for (_my, _sol), group in quartile_df.groupby(["mars_year", "sol"]):
                n = len(group)
                if n < 2:
                    continue
                lat_arr = group["_centroid_lat"].to_numpy()
                lon_arr = group["_centroid_lon"].to_numpy()
                z = group[value_col].to_numpy(dtype=float)

                dist_matrix = haversine_distance_km(lat_arr[:, None], lon_arr[:, None], lat_arr[None, :], lon_arr[None, :])
                sq_diff_matrix = (z[:, None] - z[None, :]) ** 2

                iu, ju = np.triu_indices(n, k=1)
                valid = ~(np.isnan(z[iu]) | np.isnan(z[ju]))
                d_pairs = dist_matrix[iu, ju][valid]
                sq_pairs = sq_diff_matrix[iu, ju][valid]
                if len(d_pairs) == 0:
                    continue

                bin_idx = np.searchsorted(distance_bins_arr, d_pairs, side="right") - 1
                in_range = (bin_idx >= 0) & (bin_idx < n_bins)
                for b, v in zip(bin_idx[in_range], sq_pairs[in_range]):
                    bin_sq_diffs[b].append(v)

            for bin_idx in range(n_bins):
                sq_diffs = bin_sq_diffs[bin_idx]
                if sq_diffs:
                    # MEDIAN of squared differences, not mean: the real
                    # dust_optical_depth_z distribution is heavy-tailed (see
                    # compute_episode_centred_composite's docstring -- up to
                    # |z|~2000 for a small number of cell-sols with a
                    # genuinely small, non-degenerate climatological sigma).
                    # A mean-based semivariogram is dominated by whichever
                    # distance bin happens to contain one of those rows,
                    # producing a spurious ~10x-scale spike unrelated to
                    # genuine spatial structure (found and fixed 2026-08-12,
                    # see CHAPTER_4_PARAMETER_VALUES.md). The median-based
                    # estimator is a standard, simpler alternative to the
                    # Cressie-Hawkins robust semivariogram estimator for
                    # exactly this situation.
                    rows.append(
                        {
                            "latitude_band": band_name,
                            "ls_quartile": int(quartile),
                            "distance_bin_km_lo": distance_bins_arr[bin_idx],
                            "distance_bin_km_hi": distance_bins_arr[bin_idx + 1],
                            "semivariance": 0.5 * float(np.median(sq_diffs)),
                            "n_pairs": len(sq_diffs),
                        }
                    )
    return pd.DataFrame(rows)


def figure_4_7_spatiotemporal_structure(
    labeled_train_df: pd.DataFrame,
    episode_catalogue: pd.DataFrame,
    grid: GridDefinition,
    *,
    config: dict,
):
    composite = compute_episode_centred_composite(
        labeled_train_df, episode_catalogue,
        window_before=config["episode_composite"]["window_sols_before"],
        window_after=config["episode_composite"]["window_sols_after"],
    )
    acf = compute_pooled_lag_autocorrelation(
        labeled_train_df, value_col="dust_optical_depth_z", max_lag=config["acf"]["max_lag_sols"]
    )
    pacf = durbin_levinson_pacf(acf)
    semivariogram = compute_semivariogram(
        labeled_train_df, grid,
        distance_bins_km=config["semivariogram"]["distance_bins_km"],
        latitude_bands=config["semivariogram"]["latitude_bands"],
    )

    fig = plt.figure(figsize=(14, 10))
    ax_a = fig.add_subplot(2, 2, 1)
    ax_b = fig.add_subplot(2, 2, 2)
    ax_c = fig.add_subplot(2, 1, 2)

    if len(composite):
        # Median/IQR, not mean/std: the real dust_optical_depth_z
        # distribution is heavy-tailed (see compute_episode_centred_composite's
        # docstring) -- a mean/std band would be dominated by a handful of
        # extreme z values and would not represent a typical episode.
        ax_a.plot(composite["offset_sols"], composite["median"], "o-", color="firebrick")
        ax_a.fill_between(
            composite["offset_sols"], composite["q25"], composite["q75"],
            alpha=0.2, color="firebrick",
        )
    ax_a.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax_a.set_xlabel("Sols from qualifying-episode onset")
    ax_a.set_ylabel("Median CDOD climatological z-score (band = IQR)")
    ax_a.set_title(f"(a) Episode-centred composite (n_episodes={episode_catalogue['qualifies'].sum()})")

    lags = np.arange(len(acf))
    ax_b.stem(lags, acf, linefmt="steelblue", markerfmt="o", basefmt=" ", label="ACF")
    ax_b.stem(lags, pacf, linefmt="darkorange", markerfmt="x", basefmt=" ", label="PACF")
    ax_b.set_xlabel("Lag (sols)")
    ax_b.set_ylabel("Correlation")
    ax_b.set_title("(b) Pooled panel ACF / PACF of cell-sol CDOD z-score")
    ax_b.legend(fontsize=8)

    for (band, quartile), group in semivariogram.groupby(["latitude_band", "ls_quartile"]):
        mid = (group["distance_bin_km_lo"] + group["distance_bin_km_hi"]) / 2.0
        ax_c.plot(mid, group["semivariance"], marker="o", markersize=3, alpha=0.7, label=f"{band}, Ls Q{quartile}")
    ax_c.set_xlabel("Distance (km)")
    ax_c.set_ylabel("Semivariance")
    ax_c.set_title("(c) CDOD z-score semivariogram by latitude band and Ls quartile")
    ax_c.legend(fontsize=6, ncol=3)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2)  # reserve room for save_figure_with_provenance's stamped caption block

    return fig, {"episode_composite": composite, "acf_pacf": pd.DataFrame({"lag": lags, "acf": acf, "pacf": pacf}), "semivariogram": semivariogram}


# ---------------------------------------------------------------------------
# Figure 4.8 -- sol-aggregated CDOD series with Ls markers, representative cell
# ---------------------------------------------------------------------------

def build_sol_to_ls_lookup(aggregated_df: pd.DataFrame, mars_year: int) -> pd.Series:
    """Median real solar_longitude_ls across ALL cells observed on each sol
    of `mars_year` -- Ls is a global (time-only) quantity, so pooling across
    cells gives a smooth, real sol->Ls curve even though any single cell has
    gaps."""
    my_df = aggregated_df.loc[aggregated_df["mars_year"] == mars_year]
    return my_df.groupby("sol")["solar_longitude_ls"].median().sort_index()


def _ls_crossing_sols(sol_to_ls: pd.Series, ls_targets: list[float]) -> dict[float, float]:
    """Linear-interpolates the sol at which the real, monotonically
    increasing sol->Ls curve crosses each target Ls value."""
    sols = sol_to_ls.index.to_numpy(dtype=float)
    ls = sol_to_ls.to_numpy(dtype=float)
    out: dict[float, float] = {}
    for target in ls_targets:
        idx = np.searchsorted(ls, target)
        if 0 < idx < len(ls):
            s0, s1 = sols[idx - 1], sols[idx]
            l0, l1 = ls[idx - 1], ls[idx]
            frac = (target - l0) / (l1 - l0) if l1 != l0 else 0.0
            out[target] = s0 + frac * (s1 - s0)
    return out


def figure_4_8_seasonal_series(aggregated_df: pd.DataFrame, *, cell_id: int, mars_year: int):
    cell_df = aggregated_df.loc[(aggregated_df["cell_id"] == cell_id) & (aggregated_df["mars_year"] == mars_year)].sort_values("sol")
    sol_to_ls = build_sol_to_ls_lookup(aggregated_df, mars_year)
    crossings = _ls_crossing_sols(sol_to_ls, [0.0, 90.0, 180.0, 270.0, 360.0])

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(cell_df["sol"], cell_df["dust_optical_depth"], "o-", color="steelblue", markersize=4)
    for ls_val, sol_val in crossings.items():
        ax.axvline(sol_val, color="gray", linestyle="--", linewidth=1)
        ax.text(sol_val, ax.get_ylim()[1], f"Ls={ls_val:.0f}°", rotation=90, fontsize=7, va="top", ha="right", color="dimgray")
    ax.set_xlabel(f"Sol (MY{mars_year})")
    ax.set_ylabel("Column dust optical depth")
    ax.set_title(f"Cell {cell_id}: sol-aggregated CDOD, MY{mars_year}, aligned to solar-longitude markers")

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2)  # reserve room for save_figure_with_provenance's stamped caption block

    source_df = cell_df[["sol", "mars_year", "dust_optical_depth", "solar_longitude_ls"]].copy()
    return fig, source_df
