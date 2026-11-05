import datetime as dt
import os
import tempfile
import unittest
from typing import ClassVar
from unittest.mock import patch

import numpy as np
import xarray as xr
from pandera.typing.xarray import Coordinate
from schemas import (
    NwpDatasetSchema,
    init_time,
    latitude,
    longitude,
    step,
    temperature_2m,
)
from storage.cli import ProductSpec, build_parser, run_cli
from storage.store import existing_partition, has_partition, open_icechunk_repo, write_to_icechunk


class _TestSchema(NwpDatasetSchema):
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


def _make_ds(init_dt: dt.datetime, base_val: float = 15.0) -> xr.Dataset:
    dims = _TestSchema.dims()
    naive_utc = init_dt.astimezone(dt.UTC).replace(tzinfo=None)
    data = np.full((1, 2, 2, 2), base_val, dtype=np.float32)
    return xr.Dataset(
        data_vars={"temperature_2m": (dims, data)},
        coords={
            "init_time": np.array([np.datetime64(naive_utc, "ns")]),
            "step": np.array([np.timedelta64(0, "h"), np.timedelta64(1, "h")]),
            "latitude": np.array([50.0, 51.0], dtype=np.float64),
            "longitude": np.array([-1.0, 0.0], dtype=np.float64),
        },
    )


class TestStorageAndCli(unittest.TestCase):
    def test_write_and_has_partition(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store_path = f"{tmp}/store"
            t0 = dt.datetime(2026, 4, 8, 0, 0, tzinfo=dt.UTC)
            self.assertFalse(has_partition(store_path, "init_time", t0))

            res = write_to_icechunk(_make_ds(t0), _TestSchema, store_path)
            self.assertFalse(res.skip_write)
            self.assertTrue(has_partition(store_path, "init_time", t0))
            self.assertIsNotNone(existing_partition(store_path, "init_time", "2026-04-08T00:00"))

    def test_run_cli_single_and_range_skips_existing_partitions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store_path = f"{tmp}/cli_store"
            calls: list[dt.datetime] = []

            def fake_fetch(t: dt.datetime) -> xr.Dataset:
                calls.append(t)
                return _TestSchema.validate(_make_ds(t))

            products = {
                "demo": ProductSpec(schema=_TestSchema, fetch=fake_fetch),
            }

            # Single init-time run
            run_cli(
                "demo-data",
                products,
                argv=["--store-path", store_path, "--init-time", "2026-04-08T00:00"],
            )
            self.assertEqual(calls, [dt.datetime(2026, 4, 8, 0, 0, tzinfo=dt.UTC)])

            # Range run spanning 00:00 and 06:00 -> 00:00 should be skipped before calling fetch
            run_cli(
                "demo-data",
                products,
                argv=[
                    "--store-path",
                    store_path,
                    "--start",
                    "2026-04-08T00:00",
                    "--end",
                    "2026-04-08T06:00",
                    "--step-hours",
                    "6",
                ],
            )
            self.assertEqual(
                calls,
                [
                    dt.datetime(2026, 4, 8, 0, 0, tzinfo=dt.UTC),
                    dt.datetime(2026, 4, 8, 6, 0, tzinfo=dt.UTC),
                ],
            )
            # Standalone CLI does not write stats by default
            repo, _ = open_icechunk_repo(store_path)
            session = repo.readonly_session(branch="main")
            with self.assertRaises((KeyError, ValueError, FileNotFoundError)):
                xr.open_zarr(session.store, group="stats", consolidated=False)

    def test_invalid_uri_and_bad_partition_key_raise(self) -> None:
        with self.assertRaises(ValueError):
            open_icechunk_repo("s3://bucket-without-prefix")

        with self.assertRaises(ValueError):
            existing_partition("/tmp/nonexistent", "init_time", "not-a-timestamp")

    def test_empty_step_hours_env_defaults_to_6(self) -> None:
        products = {"demo": ProductSpec(schema=_TestSchema, fetch=_make_ds)}
        with patch.dict(os.environ, {"STEP_HOURS": ""}, clear=False):
            parser = build_parser("demo-data", products)
            args = parser.parse_args(["--store-path", "/tmp/x", "--init-time", "2026-04-08T00:00"])
            self.assertEqual(args.step_hours, 6)

    def test_monthly_stats_exact_known_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store_path = f"{tmp}/exact_stats_store"
            dims = _TestSchema.dims()
            coords_common = {
                "step": np.array([np.timedelta64(0, "h"), np.timedelta64(1, "h")]),
                "latitude": np.array([50.0], dtype=np.float64),
                "longitude": np.array([-1.0, 0.0], dtype=np.float64),
            }

            # Write 1 (2026-04-01T00:00):
            #   lon=-1 has [10, 20] -> count=2, mean=15, std=5, min=10, max=20
            #   lon=0  has [2, 4]   -> count=2, mean=3,  std=1, min=2,  max=4
            ds1 = xr.Dataset(
                data_vars={
                    "temperature_2m": (
                        dims,
                        np.array([[[[10.0, 2.0]], [[20.0, 4.0]]]], dtype=np.float32),
                    )
                },
                coords={"init_time": np.array(["2026-04-01T00:00"], dtype="datetime64[ns]")}
                | coords_common,
            )
            write_to_icechunk(
                ds1, _TestSchema, store_path, keep_mantissa_bits=None, write_stats=True
            )

            # Write 2 (2026-04-15T12:00, same month):
            #   lon=-1 adds [30, 40]  -> combined [10, 20, 30, 40]:
            #       count=4, mean=25, std=sqrt(125)=11.18034, min=10, max=40
            #   lon=0  adds [10, NaN] -> combined [2, 4, 10]:
            #       count=3, mean=16/3=5.3333335, std=sqrt(104)/3=3.3993463, min=2, max=10
            ds2 = xr.Dataset(
                data_vars={
                    "temperature_2m": (
                        dims,
                        np.array([[[[30.0, 10.0]], [[40.0, np.nan]]]], dtype=np.float32),
                    )
                },
                coords={"init_time": np.array(["2026-04-15T12:00"], dtype="datetime64[ns]")}
                | coords_common,
            )
            write_to_icechunk(
                ds2, _TestSchema, store_path, keep_mantissa_bits=None, write_stats=True
            )

            repo, _ = open_icechunk_repo(store_path)
            session = repo.readonly_session(branch="main")
            stats_da = (
                xr.open_zarr(session.store, group="stats", consolidated=False)["temperature_2m"]
                .sel(month="2026-04-01", latitude=50.0)
            )

            np.testing.assert_allclose(stats_da.sel(stat="count").values, [4.0, 3.0], rtol=1e-6)
            np.testing.assert_allclose(
                stats_da.sel(stat="mean").values, [25.0, 16.0 / 3.0], rtol=1e-6
            )
            np.testing.assert_allclose(
                stats_da.sel(stat="std").values,
                [np.sqrt(125.0), np.sqrt(104.0) / 3.0],
                rtol=1e-6,
            )
            np.testing.assert_allclose(stats_da.sel(stat="min").values, [10.0, 2.0], rtol=1e-6)
            np.testing.assert_allclose(stats_da.sel(stat="max").values, [40.0, 10.0], rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
