import unittest

import numpy as np
import xarray as xr
from metoffice_data.schema import MetOfficeUkvSchema
from pandera.errors import SchemaError


class TestSchemas(unittest.TestCase):
    def test_ukv_schema_rejects_step_above_42(self):
        """The UKV schema should reject steps beyond 42 hours."""
        dims = ("init_time", "step", "y_laea", "x_laea")
        var_names = [
            "high_cloud_cover",
            "medium_cloud_cover",
            "low_cloud_cover",
            "visibility",
            "relative_humidity_2m",
            "total_precipitation_rate",
            "snow_depth",
            "downward_long_wave_radiation_flux_surface",
            "downward_short_wave_radiation_flux_surface",
            "temperature_2m",
            "wind_speed_10m",
            "wind_direction_10m",
        ]
        ds = xr.Dataset(
            data_vars={
                name: (dims, np.zeros((1, 2, 2, 2), dtype=np.float32)) for name in var_names
            },
            coords={
                "init_time": [np.datetime64("2024-01-01T00:00")],
                "step": [np.timedelta64(0, "h"), np.timedelta64(43, "h")],  # 43h should fail
                "y_laea": [700000.0, 698000.0],
                "x_laea": [-576000.0, -574000.0],
            },
        )
        with self.assertRaises(SchemaError):
            MetOfficeUkvSchema.validate(ds)


if __name__ == "__main__":
    unittest.main()
