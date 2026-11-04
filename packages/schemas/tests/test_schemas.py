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
    init_time,
    latitude,
    longitude,
    pressure_reduced_to_mean_sea_level,
    relative_humidity_2m,
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

    init_time: Coordinate[np.datetime64] = init_time()
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

    def test_cloud_cover_and_humidity_fraction_rejected(self):
        # Pure 0-1 fraction
        ds = _valid_dataset()
        ds["high_cloud_cover"].values[:] = 0.75
        with self.assertRaises(SchemaError):
            _SampleSchema.validate(ds)

        # GRIB simple-packing overshoot slightly above 1.0 on a [0, 1] field
        ds_overshoot = _valid_dataset()
        ds_overshoot["high_cloud_cover"].values[:] = 1.001
        with self.assertRaises(SchemaError):
            _SampleSchema.validate(ds_overshoot)

        # Relative humidity as fraction
        ds_rh = _valid_dataset()
        ds_rh["relative_humidity_2m"].values[:] = 0.85
        with self.assertRaises(SchemaError):
            _SampleSchema.validate(ds_rh)

        # All-zero cloud cover is valid
        ds_zero = _valid_dataset()
        ds_zero["high_cloud_cover"].values[:] = 0.0
        _SampleSchema.validate(ds_zero)

    def test_cumulative_radiation_and_precipitation_rejected(self):
        # Monotonically accumulating radiation across steps
        ds_rad = _valid_dataset()
        ds_rad["downward_short_wave_radiation_flux_surface"].values[:, 0, ...] = 0.0
        ds_rad["downward_short_wave_radiation_flux_surface"].values[:, 1, ...] = 100.0
        ds_rad["downward_short_wave_radiation_flux_surface"].values[:, 2, ...] = 350.0
        with self.assertRaises(SchemaError):
            _SampleSchema.validate(ds_rad)

        # Monotonically accumulating precipitation across steps
        ds_precip = _valid_dataset()
        ds_precip["total_precipitation_rate"].values[:, 0, ...] = 0.0
        ds_precip["total_precipitation_rate"].values[:, 1, ...] = 0.01
        ds_precip["total_precipitation_rate"].values[:, 2, ...] = 0.03
        with self.assertRaises(SchemaError):
            _SampleSchema.validate(ds_precip)

        # Fluctuating (instantaneous) radiation passes
        ds_inst = _valid_dataset()
        ds_inst["downward_short_wave_radiation_flux_surface"].values[:, 0, ...] = 50.0
        ds_inst["downward_short_wave_radiation_flux_surface"].values[:, 1, ...] = 250.0
        ds_inst["downward_short_wave_radiation_flux_surface"].values[:, 2, ...] = 100.0
        _SampleSchema.validate(ds_inst)

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
