import unittest

import numpy as np
import xarray as xr
from ecmwflive_data.schema import EcmwfLiveSchema
from pandera.errors import SchemaError


class TestSchemas(unittest.TestCase):
    def test_schema_rejects_step_above_84(self):
        """Test that the schema correctly rejects steps > 84 hours."""
        ds = xr.Dataset(
            data_vars={
                "wind_u_10m": (
                    ("init_time", "step", "latitude", "longitude"),
                    np.ones((1, 2, 2, 2), dtype=np.float32),
                ),
                "wind_v_10m": (
                    ("init_time", "step", "latitude", "longitude"),
                    np.ones((1, 2, 2, 2), dtype=np.float32),
                ),
                "wind_u_100m": (
                    ("init_time", "step", "latitude", "longitude"),
                    np.ones((1, 2, 2, 2), dtype=np.float32),
                ),
                "wind_v_100m": (
                    ("init_time", "step", "latitude", "longitude"),
                    np.ones((1, 2, 2, 2), dtype=np.float32),
                ),
                "temperature_2m": (
                    ("init_time", "step", "latitude", "longitude"),
                    np.ones((1, 2, 2, 2), dtype=np.float32),
                ),
                "wind_u_200m": (
                    ("init_time", "step", "latitude", "longitude"),
                    np.ones((1, 2, 2, 2), dtype=np.float32),
                ),
                "wind_v_200m": (
                    ("init_time", "step", "latitude", "longitude"),
                    np.ones((1, 2, 2, 2), dtype=np.float32),
                ),
                "high_cloud_cover": (
                    ("init_time", "step", "latitude", "longitude"),
                    np.ones((1, 2, 2, 2), dtype=np.float32),
                ),
                "low_cloud_cover": (
                    ("init_time", "step", "latitude", "longitude"),
                    np.ones((1, 2, 2, 2), dtype=np.float32),
                ),
                "medium_cloud_cover": (
                    ("init_time", "step", "latitude", "longitude"),
                    np.ones((1, 2, 2, 2), dtype=np.float32),
                ),
                "snow_depth": (
                    ("init_time", "step", "latitude", "longitude"),
                    np.ones((1, 2, 2, 2), dtype=np.float32),
                ),
                "direct_solar_radiation": (
                    ("init_time", "step", "latitude", "longitude"),
                    np.ones((1, 2, 2, 2), dtype=np.float32),
                ),
                "uv_b_radiation": (
                    ("init_time", "step", "latitude", "longitude"),
                    np.ones((1, 2, 2, 2), dtype=np.float32),
                ),
                "downward_long_wave_radiation_flux_surface": (
                    ("init_time", "step", "latitude", "longitude"),
                    np.ones((1, 2, 2, 2), dtype=np.float32),
                ),
                "downward_short_wave_radiation_flux_surface": (
                    ("init_time", "step", "latitude", "longitude"),
                    np.ones((1, 2, 2, 2), dtype=np.float32),
                ),
                "total_precipitation_rate": (
                    ("init_time", "step", "latitude", "longitude"),
                    np.ones((1, 2, 2, 2), dtype=np.float32),
                ),
                "total_cloud_cover_atmosphere": (
                    ("init_time", "step", "latitude", "longitude"),
                    np.ones((1, 2, 2, 2), dtype=np.float32),
                ),
                "visibility": (
                    ("init_time", "step", "latitude", "longitude"),
                    np.ones((1, 2, 2, 2), dtype=np.float32),
                ),
            },
            coords={
                "init_time": [np.datetime64("2024-01-01T00:00")],
                "step": [np.timedelta64(0, "h"), np.timedelta64(85, "h")],  # 85 hours should fail
                "latitude": [52.0, 51.0],
                "longitude": [4.0, 5.0],
            },
        )
        with self.assertRaises(SchemaError):
            EcmwfLiveSchema.validate(ds)


if __name__ == "__main__":
    unittest.main()
