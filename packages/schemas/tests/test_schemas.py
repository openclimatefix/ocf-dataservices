import unittest
from typing import ClassVar

import numpy as np
import xarray as xr
from pandera.errors import SchemaError
from pandera.typing.xarray import Coordinate
from schemas import (
    NwpDatasetSchema,
    downward_short_wave_radiation_flux_surface,
    ensemble_member,
    high_cloud_cover,
    init_time_coord,
    latitude,
    longitude,
    pressure_reduced_to_mean_sea_level,
    relative_humidity_2m,
    scale_to_percent,
    step,
    temperature_2m,
    total_precipitation_rate,
    validates,
    wind_u_10m,
    wind_v_10m,
)


class _SampleSchema(NwpDatasetSchema):
    _dims = ("init_time", "step", "ensemble_member", "latitude", "longitude")
    _chunks: ClassVar[dict[str, int]] = {
        "init_time": 1,
        "step": 1,
        "ensemble_member": 1,
        "latitude": -1,
        "longitude": -1,
    }
    _shards: ClassVar[dict[str, int]] = {
        "init_time": 1,
        "step": -1,
        "ensemble_member": -1,
        "latitude": -1,
        "longitude": -1,
    }

    init_time: Coordinate[np.datetime64] = init_time_coord()
    step: Coordinate[np.timedelta64] = step(ge_hours=0, le_hours=85)
    ensemble_member: Coordinate[np.int16] = ensemble_member(ge=1, le=50)
    latitude: Coordinate[np.float64] = latitude()
    longitude: Coordinate[np.float64] = longitude()

    temperature_2m: np.float32 = temperature_2m(dims=_dims, nullable=False)
    pressure_reduced_to_mean_sea_level: np.float32 = pressure_reduced_to_mean_sea_level(
        dims=_dims, nullable=False
    )
    high_cloud_cover: np.float32 = high_cloud_cover(dims=_dims, nullable=False)
    relative_humidity_2m: np.float32 = relative_humidity_2m(dims=_dims, nullable=False)
    downward_short_wave_radiation_flux_surface: np.float32 = (
        downward_short_wave_radiation_flux_surface(dims=_dims, nullable=True)
    )
    total_precipitation_rate: np.float32 = total_precipitation_rate(dims=_dims, nullable=True)
    wind_u_10m: np.float32 = wind_u_10m(dims=_dims, nullable=True)
    wind_v_10m: np.float32 = wind_v_10m(dims=_dims, nullable=True)


def _valid_dataset() -> xr.Dataset:
    dims = _SampleSchema.dims()
    shape = (1, 3, 2, 2, 2)
    return xr.Dataset(
        data_vars={
            "temperature_2m": (dims, np.full(shape, 15.0, dtype=np.float32)),
            "pressure_reduced_to_mean_sea_level": (
                dims,
                np.full(shape, 101325.0, dtype=np.float32),
            ),
            "high_cloud_cover": (dims, np.full(shape, 50.0, dtype=np.float32)),
            "relative_humidity_2m": (dims, np.full(shape, 75.0, dtype=np.float32)),
            "downward_short_wave_radiation_flux_surface": (
                dims,
                np.full(shape, 200.0, dtype=np.float32),
            ),
            "total_precipitation_rate": (dims, np.full(shape, 0.001, dtype=np.float32)),
            "wind_u_10m": (dims, np.full(shape, 3.0, dtype=np.float32)),
            "wind_v_10m": (dims, np.full(shape, -2.0, dtype=np.float32)),
        },
        coords={
            "init_time": np.array(["2024-01-01T00:00"], dtype="datetime64[ns]"),
            "step": np.array(
                [np.timedelta64(0, "h"), np.timedelta64(1, "h"), np.timedelta64(2, "h")]
            ),
            "ensemble_member": np.array([1, 2], dtype=np.int16),
            "latitude": np.array([50.0, 51.0], dtype=np.float64),
            "longitude": np.array([-1.0, 0.0], dtype=np.float64),
        },
    )


class TestNwpDatasetSchema(unittest.TestCase):
    def test_valid_dataset_passes(self):
        ds = _valid_dataset()
        _SampleSchema.validate(ds)

    def test_subclass_must_declare_dims_chunks_shards(self):
        with self.assertRaises(ValueError):

            class MissingDimsSchema(NwpDatasetSchema):
                pass

        with self.assertRaises(ValueError):

            class MismatchedChunksSchema(NwpDatasetSchema):
                _dims = ("init_time", "step")
                _chunks: ClassVar[dict[str, int]] = {"step": 1, "init_time": 1}
                _shards: ClassVar[dict[str, int]] = {"init_time": 1, "step": -1}

    def test_temperature_in_kelvin_rejected(self):
        ds = _valid_dataset()
        ds["temperature_2m"] = ds["temperature_2m"] + np.float32(273.15)
        with self.assertRaises(SchemaError):
            _SampleSchema.validate(ds)

    def test_pressure_in_hpa_rejected(self):
        ds = _valid_dataset()
        ds["pressure_reduced_to_mean_sea_level"].values[:] = 1013.25
        with self.assertRaises(SchemaError):
            _SampleSchema.validate(ds)

    def test_clear_sky_cloud_wisp_and_scale_to_percent(self):
        # Clear sky with a 0.5% cloud wisp (0 < max <= 1.1) is valid
        ds_wisp = _valid_dataset()
        ds_wisp["high_cloud_cover"].values[:] = 0.0
        ds_wisp["high_cloud_cover"].values[0, 0, 0, 0, 0] = 0.5
        _SampleSchema.validate(ds_wisp)

        # Out-of-range percentage (> 100) is rejected
        ds_over = _valid_dataset()
        ds_over["high_cloud_cover"].values[:] = 105.0
        with self.assertRaises(SchemaError):
            _SampleSchema.validate(ds_over)

        # GRIB unit metadata prevents scaling an already-percent field with max <= 1.1
        da_pct = xr.DataArray([0.0, 0.8], attrs={"GRIB_units": "%"})
        np.testing.assert_allclose(scale_to_percent(da_pct).values, [0.0, 0.8])
        da_frac = xr.DataArray([0.0, 0.008], attrs={"GRIB_units": "(0 - 1)"})
        np.testing.assert_allclose(scale_to_percent(da_frac).values, [0.0, 0.8])

    def test_late_arriving_rain_and_radiation_bounds(self):
        # Dry forecast where rain begins on the final step passes validation
        ds_rain = _valid_dataset()
        ds_rain["total_precipitation_rate"].values[:] = 0.0
        ds_rain["total_precipitation_rate"].values[0, -1, 0, 0, 0] = 0.0005
        _SampleSchema.validate(ds_rain)

        # Unconverted cumulative radiation in J m-2 (> 2000 W m-2) is rejected
        ds_rad = _valid_dataset()
        ds_rad["downward_short_wave_radiation_flux_surface"].values[:] = 360000.0
        with self.assertRaises(SchemaError):
            _SampleSchema.validate(ds_rad)

    def test_wrong_encoding_dtypes_rejected(self):
        ds_f64 = _valid_dataset()
        ds_f64["temperature_2m"] = ds_f64["temperature_2m"].astype(np.float64)
        with self.assertRaises(SchemaError):
            _SampleSchema.validate(ds_f64)

        ds_i64 = _valid_dataset()
        ds_i64["ensemble_member"] = ds_i64["ensemble_member"].astype(np.int64)
        with self.assertRaises(SchemaError):
            _SampleSchema.validate(ds_i64)

    def test_wrong_dataset_dim_order_rejected(self):
        base = _valid_dataset()
        ds_wrong_order = xr.Dataset(
            coords={
                "step": base.coords["step"],
                "init_time": base.coords["init_time"],
                "ensemble_member": base.coords["ensemble_member"],
                "latitude": base.coords["latitude"],
                "longitude": base.coords["longitude"],
            }
        )
        for name, da in base.data_vars.items():
            ds_wrong_order[name] = da
        self.assertNotEqual(list(ds_wrong_order.dims), list(_SampleSchema.dims()))
        with self.assertRaises(SchemaError):
            _SampleSchema.validate(ds_wrong_order)

    def test_empty_dimension_rejected(self):
        ds_empty = _valid_dataset().isel(latitude=slice(0, 0))
        with self.assertRaises(SchemaError):
            _SampleSchema.validate(ds_empty)

    def test_unsorted_or_duplicate_coordinates_rejected(self):
        ds_unsorted = _valid_dataset()
        ds_unsorted = ds_unsorted.assign_coords(
            step=[np.timedelta64(2, "h"), np.timedelta64(1, "h"), np.timedelta64(0, "h")]
        )
        with self.assertRaises(SchemaError):
            _SampleSchema.validate(ds_unsorted)

    def test_validates_decorator_null_reporting(self):
        # Allowed NaN in nullable=True variable must NOT mask a unit error in temperature_2m
        @validates(_SampleSchema)
        def produce_bad_temp() -> xr.Dataset:
            ds = _valid_dataset()
            ds["wind_u_10m"].values[0, 0, 0, 0, 0] = np.nan
            ds["temperature_2m"] = ds["temperature_2m"] + np.float32(273.15)
            return ds

        with self.assertRaises(SchemaError):
            produce_bad_temp()

        # Disallowed NaN in nullable=False variable SHOULD raise descriptive ValueError
        @validates(_SampleSchema)
        def produce_unexpected_null() -> xr.Dataset:
            ds = _valid_dataset()
            ds["temperature_2m"].values[0, 0, 0, 0, 0] = np.nan
            return ds

        with self.assertRaisesRegex(ValueError, "temperature_2m"):
            produce_unexpected_null()


if __name__ == "__main__":
    unittest.main()
