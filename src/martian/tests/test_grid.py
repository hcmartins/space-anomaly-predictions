"""Tests for grid geometry: cell boundaries, longitude wraparound, poles,
resolution validation, and neighbour adjacency."""
import numpy as np
import pandas as pd
import pytest

from martian.preprocessing.grid import (
    GridDefinition,
    assign_cells,
    bands_from_cell_id,
    cell_id,
    lat_to_band_index,
    load_primary_grid,
    load_sensitivity_grids,
    lon_to_band_index,
    neighbours,
)


def test_primary_grid_matches_configured_values():
    grid = load_primary_grid()
    assert grid.resolution_deg == 5.0
    assert grid.n_lat_bands == 36
    assert grid.n_lon_bands == 72
    assert grid.n_cells == 2592


def test_sensitivity_grids_are_2p5_and_10_and_5():
    grids = load_sensitivity_grids()
    resolutions = sorted(g.resolution_deg for g in grids)
    assert resolutions == [2.5, 5.0, 10.0]
    by_res = {g.resolution_deg: g for g in grids}
    assert by_res[2.5].n_cells == 72 * 144
    assert by_res[10.0].n_cells == 18 * 36


def test_resolution_not_dividing_180_or_360_raises():
    with pytest.raises(ValueError):
        GridDefinition.from_resolution(7.0)


def test_lat_band_boundaries_exact_5deg_grid():
    grid = GridDefinition.from_resolution(5.0)
    assert lat_to_band_index(np.array([-90.0]), grid)[0] == 0
    assert lat_to_band_index(np.array([-85.0]), grid)[0] == 1
    assert lat_to_band_index(np.array([0.0]), grid)[0] == 18
    # exact north pole clamped into the last band, not overflowing by one
    assert lat_to_band_index(np.array([90.0]), grid)[0] == grid.n_lat_bands - 1
    assert lat_to_band_index(np.array([89.999]), grid)[0] == grid.n_lat_bands - 1


def test_lat_out_of_range_raises():
    grid = GridDefinition.from_resolution(5.0)
    with pytest.raises(ValueError):
        lat_to_band_index(np.array([-91.0]), grid)
    with pytest.raises(ValueError):
        lat_to_band_index(np.array([91.0]), grid)


def test_lat_nan_raises():
    grid = GridDefinition.from_resolution(5.0)
    with pytest.raises(ValueError):
        lat_to_band_index(np.array([np.nan]), grid)


def test_longitude_wraparound_adjacent_bands():
    grid = GridDefinition.from_resolution(5.0)
    # 359.9 and 0.05 straddle the wrap point but land in the last / first band
    band_last = lon_to_band_index(np.array([359.9]), grid)[0]
    band_first = lon_to_band_index(np.array([0.05]), grid)[0]
    assert band_last == grid.n_lon_bands - 1
    assert band_first == 0


def test_longitude_360_equivalent_to_0():
    grid = GridDefinition.from_resolution(5.0)
    assert lon_to_band_index(np.array([360.0]), grid)[0] == lon_to_band_index(np.array([0.0]), grid)[0]


def test_longitude_negative_wraps():
    grid = GridDefinition.from_resolution(5.0)
    # -0.05 degrees == 359.95 degrees, same band as 359.9
    band_neg = lon_to_band_index(np.array([-0.05]), grid)[0]
    assert band_neg == grid.n_lon_bands - 1


def test_lon_nan_raises():
    grid = GridDefinition.from_resolution(5.0)
    with pytest.raises(ValueError):
        lon_to_band_index(np.array([np.nan]), grid)


def test_cell_id_is_deterministic_and_unique():
    grid = GridDefinition.from_resolution(5.0)
    assert cell_id(0, 0, grid) == 0
    assert cell_id(0, 1, grid) == 1
    assert cell_id(1, 0, grid) == grid.n_lon_bands
    ids = {cell_id(la, lo, grid) for la in range(grid.n_lat_bands) for lo in range(grid.n_lon_bands)}
    assert len(ids) == grid.n_cells


def test_bands_from_cell_id_is_exact_inverse_of_cell_id():
    grid = GridDefinition.from_resolution(5.0)
    for la in range(0, grid.n_lat_bands, 7):
        for lo in range(0, grid.n_lon_bands, 11):
            cid = cell_id(la, lo, grid)
            recovered_la, recovered_lo = bands_from_cell_id(cid, grid)
            assert int(recovered_la) == la
            assert int(recovered_lo) == lo


def test_assign_cells_adds_expected_columns():
    grid = GridDefinition.from_resolution(5.0)
    df = pd.DataFrame({"latitude": [-90.0, 0.0, 89.9], "longitude": [0.0, 180.0, 359.9]})
    out = assign_cells(df, grid)
    assert list(out.columns[-3:]) == ["lat_band", "lon_band", "cell_id"]
    assert out.loc[0, "lat_band"] == 0
    assert out.loc[0, "lon_band"] == 0


def test_neighbours_interior_cell_has_eight():
    grid = GridDefinition.from_resolution(5.0)
    result = neighbours(5, 10, grid)
    assert len(result) == 8
    assert (5, 10) not in result  # excludes itself


def test_neighbours_pole_cell_has_five():
    grid = GridDefinition.from_resolution(5.0)
    south_pole_neighbours = neighbours(0, 10, grid)
    north_pole_neighbours = neighbours(grid.n_lat_bands - 1, 10, grid)
    assert len(south_pole_neighbours) == 5
    assert len(north_pole_neighbours) == 5


def test_neighbours_wrap_at_longitude_edges():
    grid = GridDefinition.from_resolution(5.0)
    result = neighbours(5, 0, grid)
    lon_bands = {lo for _, lo in result}
    assert grid.n_lon_bands - 1 in lon_bands  # wrapped left neighbour
    assert 1 in lon_bands  # right neighbour


def test_neighbours_out_of_range_raises():
    grid = GridDefinition.from_resolution(5.0)
    with pytest.raises(ValueError):
        neighbours(-1, 0, grid)
    with pytest.raises(ValueError):
        neighbours(0, grid.n_lon_bands, grid)
