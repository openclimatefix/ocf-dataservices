import unittest

import numpy as np
import xarray as xr
from ecmwflive_data.schema import EcmwfLiveSchema
from pandera.errors import SchemaError


class TestSchemas(unittest.TestCase):
    def _make_dataset(self, max_step_hours: int = 84) -> xr.Dataset:
        dims = ("init_time", "step", "latitude", "longitude")
        shape = (1, 2, 2, 2)
        return xr.Dataset(
            data_vars={
                "wind_u_10m": (dims, np.ones(shape, dtype=np.float32)),
                "wind_v_10m": (dims, np.ones(shape, dtype=np.float32)),
                "wind_u_100m": (dims, np.ones(shape, dtype=np.float32)),
                "wind_v_100m": (dims, np.ones(shape, dtype=np.float32)),
                "temperature_2m": (dims, np.ones(shape, dtype=np.float32)),
                "wind_u_200m": (dims, np.ones(shape, dtype=np.float32)),
                "wind_v_200m": (dims, np.ones(shape, dtype=np.float32)),
                "high_cloud_cover": (dims, np.full(shape, 50.0, dtype=np.float32)),
                "low_cloud_cover": (dims, np.full(shape, 50.0, dtype=np.float32)),
                "medium_cloud_cover": (dims, np.full(shape, 50.0, dtype=np.float32)),
                "snow_depth": (dims, np.ones(shape, dtype=np.float32)),
                "direct_solar_radiation": (dims, np.ones(shape, dtype=np.float32)),
                "uv_b_radiation": (dims, np.ones(shape, dtype=np.float32)),
                "downward_long_wave_radiation_flux_surface": (
                    dims,
                    np.ones(shape, dtype=np.float32),
                ),
                "downward_short_wave_radiation_flux_surface": (
                    dims,
                    np.ones(shape, dtype=np.float32),
                ),
                "total_precipitation_rate": (dims, np.full(shape, 0.01, dtype=np.float32)),
                "total_cloud_cover_atmosphere": (dims, np.full(shape, 50.0, dtype=np.float32)),
                "visibility": (dims, np.ones(shape, dtype=np.float32)),
            },
            coords={
                "init_time": [np.datetime64("2024-01-01T00:00")],
                "step": [np.timedelta64(0, "h"), np.timedelta64(max_step_hours, "h")],
                "latitude": [52.0, 51.0],
                "longitude": [4.0, 5.0],
                "generating_process_id": ("init_time", np.array([154], dtype=np.int16)),
                "expver": ("init_time", np.array([1], dtype=np.int16)),
                "grib_tables_version": ("init_time", np.array([31], dtype=np.int16)),
                "ingested_at": ("init_time", [np.datetime64("2024-01-01T01:00")]),
            },
        )

    def test_valid_schema_passes(self):
        EcmwfLiveSchema.validate(self._make_dataset(max_step_hours=84))

    def test_schema_rejects_step_above_84(self):
        """Test that the schema correctly rejects steps > 84 hours."""
        with self.assertRaises(SchemaError):
            EcmwfLiveSchema.validate(self._make_dataset(max_step_hours=85))


if __name__ == "__main__":
    unittest.main()
