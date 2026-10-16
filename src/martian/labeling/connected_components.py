"""Spatiotemporal episode construction over exceedance cell-sols (Eq. 4.3's
connected-component step, before the persistence/area gate is applied).

Two exceedance cell-sols (c1, s1) and (c2, s2) within the SAME mars_year are
connected iff:
  - s1 == s2 and c1, c2 are eight-neighbour spatial neighbours (longitude
    wraparound applied, via preprocessing.grid.neighbours), OR
  - |s1 - s2| == 1 and c1 == c2 OR c1, c2 are eight-neighbour spatial
    neighbours.

The second rule is what makes this a genuinely SPATIOTEMPORAL construction
rather than independent per-sol spatial blobs stitched together only when
they land on the exact same cell: a storm that drifts by one cell between
consecutive sols is still one episode. This is the standard 26-connectivity
analogue used for 3D (lat, lon, time) volumes in image processing, restricted
here to a single mars_year (sol indices restart at 0 each Mars year, so
mars_year equality is enforced on every edge to avoid a false connection
between, e.g., MY1 sol 667 and MY2 sol 0).

A connected component is a candidate episode; whether it becomes a
`storm_episode` label depends on the persistence/area gate applied by
labeling/anomaly_labels.py, not by this module.
"""
from __future__ import annotations

import pandas as pd

from preprocessing.grid import GridDefinition, bands_from_cell_id, cell_id as grid_cell_id, neighbours


class _UnionFind:
    def __init__(self, keys) -> None:
        self._parent = {k: k for k in keys}

    def find(self, x):
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a, b) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[ra] = rb


def assign_episode_ids(
    exceedance_df: pd.DataFrame,
    grid: GridDefinition,
    *,
    cell_id_col: str = "cell_id",
    mars_year_col: str = "mars_year",
    sol_col: str = "sol",
) -> pd.Series:
    """`exceedance_df` must contain ONLY rows that are exceedance-positive
    (e.g. `atmospheric_anomaly == True`). Returns a Series of integer episode
    ids aligned to `exceedance_df.index` (each connected component gets its
    own id; ids are arbitrary and not meaningful across calls)."""
    if exceedance_df.empty:
        return pd.Series([], dtype=int, index=exceedance_df.index)

    uf = _UnionFind(exceedance_df.index)

    lookup: dict[tuple[int, int, int], object] = {}
    for idx, my, cid, sol in exceedance_df[[mars_year_col, cell_id_col, sol_col]].itertuples(name=None):
        lookup[(int(my), int(cid), int(sol))] = idx

    lat_band_arr, lon_band_arr = bands_from_cell_id(
        exceedance_df[cell_id_col].to_numpy(), grid
    )

    for row_pos, (idx, my, cid, sol) in enumerate(
        exceedance_df[[mars_year_col, cell_id_col, sol_col]].itertuples(name=None)
    ):
        my, sol = int(my), int(sol)
        lat_band, lon_band = int(lat_band_arr[row_pos]), int(lon_band_arr[row_pos])
        spatial_neighbours = neighbours(lat_band, lon_band, grid)

        # Same-sol spatial edges.
        for nla, nlo in spatial_neighbours:
            ncid = int(grid_cell_id(nla, nlo, grid))
            other = lookup.get((my, ncid, sol))
            if other is not None:
                uf.union(idx, other)

        # Adjacent-sol edges: same cell OR spatial neighbour.
        for d_sol in (-1, 1):
            other_sol = sol + d_sol
            for nla, nlo in spatial_neighbours + [(lat_band, lon_band)]:
                ncid = int(grid_cell_id(nla, nlo, grid))
                other = lookup.get((my, ncid, other_sol))
                if other is not None:
                    uf.union(idx, other)

    roots = pd.Series([uf.find(idx) for idx in exceedance_df.index], index=exceedance_df.index)
    episode_ids, _ = pd.factorize(roots)
    return pd.Series(episode_ids, index=exceedance_df.index)


def summarize_episodes(
    exceedance_df_with_episode_id: pd.DataFrame,
    *,
    episode_id_col: str = "episode_id",
    cell_id_col: str = "cell_id",
    mars_year_col: str = "mars_year",
    sol_col: str = "sol",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
) -> pd.DataFrame:
    """One row per episode: onset/end sol, duration, distinct-cell count,
    peak simultaneous-cell count (the spatial-extent metric used by the
    persistence/area gate), and a coarse centroid for mapping."""
    rows = []
    for episode_id, group in exceedance_df_with_episode_id.groupby(episode_id_col, sort=True):
        sols = group[sol_col].astype(int)
        per_sol_cell_counts = group.groupby(sol_col)[cell_id_col].nunique()
        # Longitude sectors spanned (out of 3 x 120-degree sectors), used by
        # labeling/storm_taxonomy.py's planet_encircling test -- computed
        # directly from raw longitude, no grid dependency needed here.
        sectors_touched = set((group[lon_col].to_numpy(dtype=float) // 120.0).astype(int) % 3)
        rows.append(
            {
                "episode_id": int(episode_id),
                "mars_year": int(group[mars_year_col].iloc[0]),
                "onset_sol": int(sols.min()),
                "end_sol": int(sols.max()),
                "duration_sols": int(sols.max() - sols.min() + 1),
                "n_member_cell_sols": int(len(group)),
                "n_distinct_cells": int(group[cell_id_col].nunique()),
                "peak_area_cells": int(per_sol_cell_counts.max()),
                "centroid_lat": float(group[lat_col].mean()),
                "centroid_lon": float(group[lon_col].mean()),
                "longitude_sectors_spanned": len(sectors_touched),
            }
        )
    return pd.DataFrame(rows)
