import unittest

import numpy as np
import xarray as xr
from metoffice_data.schema import MetOfficeUkvSchema
from pandera.errors import SchemaError


class TestSchemas(unittest.TestCase):
    def _make_ukv_dataset(self, max_step_hours: int = 42) -> xr.Dataset:
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
            "wind_u_10m",
            "wind_v_10m",
        ]
        return xr.Dataset(
            data_vars={
                name: (
                    dims,
                    np.full(
                        (1, 2, 2, 2),
                        50.0 if "humidity" in name or "cloud" in name else 0.0,
                        dtype=np.float32,
                    ),
                )
                for name in var_names
            },
            coords={
                "init_time": [np.datetime64("2024-01-01T00:00")],
                "step": [np.timedelta64(0, "h"), np.timedelta64(max_step_hours, "h")],
                "y_laea": [700000.0, 698000.0],
                "x_laea": [-576000.0, -574000.0],
                "generating_process_id": ("init_time", np.array([128], dtype=np.int16)),
                "local_tables_version": ("init_time", np.array([1], dtype=np.int16)),
                "grid_first_lat": ("init_time", np.array([48.5], dtype=np.float64)),
                "grid_first_lon": ("init_time", np.array([-11.5], dtype=np.float64)),
                "ingested_at": ("init_time", [np.datetime64("2024-01-01T01:00")]),
            },
        )

    def test_ukv_schema_passes_valid_dataset(self):
        MetOfficeUkvSchema.validate(self._make_ukv_dataset(max_step_hours=42))

    def test_ukv_schema_rejects_step_above_42(self):
        """The UKV schema should reject steps beyond 42 hours."""
        with self.assertRaises(SchemaError):
            MetOfficeUkvSchema.validate(self._make_ukv_dataset(max_step_hours=43))


if __name__ == "__main__":
    unittest.main()
