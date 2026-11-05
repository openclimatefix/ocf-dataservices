import tempfile
import unittest
from typing import ClassVar

import dagster as dg
import icechunk
import numpy as np
import xarray as xr
from pandera.typing.xarray import Coordinate
from schemas import (
    NwpDatasetSchema,
    ensemble_member,
    init_time,
    latitude,
    longitude,
    pressure_reduced_to_mean_sea_level,
    step,
    temperature_2m,
    x_laea,
    y_laea,
)

from ocf_dataservices.resources.iomanager_xarray_icechunk import XarrayIcechunkIOManager


class _EnsembleProjectedSchema(NwpDatasetSchema):
    _dims = ("init_time", "step", "ensemble_member", "y_laea", "x_laea")
    _chunks: ClassVar[dict[str, int]] = {
        "init_time": 1,
        "step": 1,
        "ensemble_member": 1,
        "y_laea": -1,
        "x_laea": -1,
    }
    _shards: ClassVar[dict[str, int]] = {
        "init_time": 1,
        "step": -1,
        "ensemble_member": -1,
        "y_laea": -1,
        "x_laea": -1,
    }

    init_time: Coordinate[np.datetime64] = init_time()
    step: Coordinate[np.timedelta64] = step(ge_hours=0, le_hours=6)
    ensemble_member: Coordinate[np.int16] = ensemble_member(ge=1, le=2)
    y_laea: Coordinate[np.float64] = y_laea(ge=-576000, le=700000)
    x_laea: Coordinate[np.float64] = x_laea(ge=-576000, le=332000)
    pressure_reduced_to_mean_sea_level: np.float32 = pressure_reduced_to_mean_sea_level(
        dims=_dims, nullable=False
    )


class _DummySchema(NwpDatasetSchema):
    _dims = ("init_time", "step", "latitude", "longitude")
    _chunks: ClassVar[dict[str, int]] = {
        "init_time": 1,
        "step": 1,
        "latitude": -1,
        "longitude": -1,
    }
    _shards: ClassVar[dict[str, int]] = {
        "init_time": 1,
        "step": -1,
        "latitude": -1,
        "longitude": -1,
    }

    init_time: Coordinate[np.datetime64] = init_time()
    step: Coordinate[np.timedelta64] = step(ge_hours=0, le_hours=6)
    latitude: Coordinate[np.float64] = latitude()
    longitude: Coordinate[np.float64] = longitude()
    temperature_2m: np.float32 = temperature_2m(dims=_dims, nullable=False)


def _make_partition_ds(init_time_str: str, base_val: float) -> xr.Dataset:
    dims = _DummySchema.dims()
    # Shape: (1, 2, 2, 3) -> 2 steps, 2x3 spatial grid
    data = np.arange(12, dtype=np.float32).reshape(1, 2, 2, 3) + np.float32(base_val)
    return xr.Dataset(
        data_vars={"temperature_2m": (dims, data)},
        coords={
            "init_time": np.array([init_time_str], dtype="datetime64[ns]"),
            "step": np.array([np.timedelta64(0, "h"), np.timedelta64(1, "h")]),
            "latitude": np.array([50.0, 51.0], dtype=np.float64),
            "longitude": np.array([-1.0, 0.0, 1.0], dtype=np.float64),
        },
    )


class TestXarrayIcechunkIOManagerStats(unittest.TestCase):
    def test_writes_and_collates_monthly_stats_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            io_manager = XarrayIcechunkIOManager(path=tmp, keep_mantissa_bits=None)
            asset_key = dg.AssetKey(["nwp", "l1_dummy_v1"])

            ds1 = _make_partition_ds("2024-04-01T00:00", 10.0)
            ctx1 = dg.build_output_context(
                asset_key=asset_key,
                partition_key="2024-04-01T00:00",
                definition_metadata={"schema": _DummySchema},
            )
            io_manager.handle_output(ctx1, ds1)
            logged_meta = ctx1.get_logged_metadata()
            self.assertIn("stats_summary", logged_meta)
            self.assertIn("stats_overview", logged_meta)
            self.assertIn("data:image/svg+xml;base64,", logged_meta["stats_overview"].md_str)

            # Second partition in the SAME month (April 2024) -> merges into month 2024-04-01
            ds2 = _make_partition_ds("2024-04-01T06:00", 20.0)
            ctx2 = dg.build_output_context(
                asset_key=asset_key,
                partition_key="2024-04-01T06:00",
                definition_metadata={"schema": _DummySchema},
            )
            io_manager.handle_output(ctx2, ds2)

            # Duplicate write should be skipped without double-counting into monthly stats
            ctx2_dup = dg.build_output_context(
                asset_key=asset_key,
                partition_key="2024-04-01T06:00",
                definition_metadata={"schema": _DummySchema},
            )
            io_manager.handle_output(ctx2_dup, ds2)

            # Third partition in a NEW month (May 2024) -> appends month 2024-05-01
            ds3 = _make_partition_ds("2024-05-01T00:00", 30.0)
            ctx3 = dg.build_output_context(
                asset_key=asset_key,
                partition_key="2024-05-01T00:00",
                definition_metadata={"schema": _DummySchema},
            )
            io_manager.handle_output(ctx3, ds3)

            store_path = f"{tmp}/nwp/l1_dummy_v1"
            repo = icechunk.Repository.open(icechunk.local_filesystem_storage(store_path))
            session = repo.readonly_session(branch="main")

            root_ds = xr.open_zarr(session.store, consolidated=False)
            stats_ds = xr.open_zarr(session.store, group="stats", consolidated=False)

            with self.assertRaises((KeyError, ValueError, FileNotFoundError)):
                xr.open_zarr(session.store, group="stats_spatial", consolidated=False)

            self.assertEqual(list(root_ds.data_vars), ["temperature_2m"])
            self.assertEqual(
                dict(root_ds.sizes),
                {"init_time": 3, "step": 2, "latitude": 2, "longitude": 3},
            )

            self.assertEqual(
                dict(stats_ds.sizes),
                {"month": 2, "stat": 5, "latitude": 2, "longitude": 3},
            )
            self.assertEqual(
                list(stats_ds.coords["stat"].values),
                ["count", "mean", "std", "min", "max"],
            )
            np.testing.assert_array_equal(
                stats_ds.coords["month"].values,
                np.array(["2024-04-01", "2024-05-01"], dtype="datetime64[ns]"),
            )

            april_full = xr.concat([ds1, ds2], dim="init_time")
            april_stats = stats_ds["temperature_2m"].sel(month="2024-04-01")
            np.testing.assert_allclose(
                april_stats.sel(stat="count").values,
                np.full((2, 3), 4.0, dtype=np.float32),
            )
            np.testing.assert_allclose(
                april_stats.sel(stat="mean").values,
                april_full["temperature_2m"].mean(dim=["init_time", "step"]).values,
                rtol=1e-5,
            )
            np.testing.assert_allclose(
                april_stats.sel(stat="std").values,
                april_full["temperature_2m"].std(dim=["init_time", "step"]).values,
                rtol=1e-5,
            )
            np.testing.assert_allclose(
                april_stats.sel(stat="min").values,
                april_full["temperature_2m"].min(dim=["init_time", "step"]).values,
            )
            np.testing.assert_allclose(
                april_stats.sel(stat="max").values,
                april_full["temperature_2m"].max(dim=["init_time", "step"]).values,
            )

            may_stats = stats_ds["temperature_2m"].sel(month="2024-05-01")
            np.testing.assert_allclose(
                may_stats.sel(stat="count").values,
                np.full((2, 3), 2.0, dtype=np.float32),
            )
            np.testing.assert_allclose(
                may_stats.sel(stat="mean").values,
                ds3["temperature_2m"].mean(dim=["init_time", "step"]).values,
                rtol=1e-5,
            )

    def test_backfills_stats_on_existing_store_with_ensemble_and_projected_coords(self):
        with tempfile.TemporaryDirectory() as tmp:
            asset_key = dg.AssetKey(["nwp", "l1_ens_ukv_v1"])
            store_path = f"{tmp}/nwp/l1_ens_ukv_v1"

            dims = _EnsembleProjectedSchema.dims()
            # Use pressure values that differ by 1.0 Pa (smaller than 14-bit Veltkamp 4.0 Pa step)
            data = np.array([101325.0, 101326.0, 101327.0, 101328.0], dtype=np.float32)
            grid = np.broadcast_to(data.reshape(1, 2, 2, 1, 1), (1, 2, 2, 2, 3)).copy()
            ds = xr.Dataset(
                data_vars={"pressure_reduced_to_mean_sea_level": (dims, grid)},
                coords={
                    "init_time": np.array(["2024-04-01T00:00"], dtype="datetime64[ns]"),
                    "step": np.array([np.timedelta64(0, "h"), np.timedelta64(1, "h")]),
                    "ensemble_member": np.array([1, 2], dtype=np.int16),
                    "y_laea": np.array([0.0, 2000.0], dtype=np.float64),
                    "x_laea": np.array([-2000.0, 0.0, 2000.0], dtype=np.float64),
                },
            )

            # Simulate pre-existing store with root data only (no stats group yet)
            repo = icechunk.Repository.create(icechunk.local_filesystem_storage(store_path))
            s0 = repo.writable_session(branch="main")
            ds.to_zarr(s0.store, zarr_format=3, mode="w-")
            s0.commit("legacy root write")

            # Re-running the same partition should backfill stats
            # and compute stats before 14-bit Veltkamp rounding
            io_manager = XarrayIcechunkIOManager(path=tmp, keep_mantissa_bits=14)
            ctx = dg.build_output_context(
                asset_key=asset_key,
                partition_key="2024-04-01T00:00",
                definition_metadata={"schema": _EnsembleProjectedSchema},
            )
            io_manager.handle_output(ctx, ds)

            ro = repo.readonly_session(branch="main")
            stats_ds = xr.open_zarr(ro.store, group="stats", consolidated=False)

            self.assertEqual(
                dict(stats_ds.sizes),
                {"month": 1, "stat": 5, "y_laea": 2, "x_laea": 3},
            )
            np.testing.assert_allclose(
                stats_ds["pressure_reduced_to_mean_sea_level"].sel(stat="std").values,
                ds["pressure_reduced_to_mean_sea_level"]
                .std(dim=["init_time", "step", "ensemble_member"])
                .broadcast_like(stats_ds["pressure_reduced_to_mean_sea_level"].sel(stat="std"))
                .values,
            )


if __name__ == "__main__":
    unittest.main()
