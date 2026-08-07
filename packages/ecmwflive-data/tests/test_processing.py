from unittest.mock import patch

import numpy as np
import xarray as xr
from ecmwflive_data.processing import process_ecmwf_live


def test_process_ecmwf_live_truncates_step():
    """Test that process_ecmwf_live correctly slices step coordinates."""
    # Create mock dataset
    ds = xr.Dataset(
        data_vars={
            "t2m": (("step", "latitude", "longitude"), np.ones((3, 2, 2), dtype=np.float32)),
        },
        coords={
            "time": np.datetime64("2024-01-01T00:00"),
            "step": [np.timedelta64(0, "h"), np.timedelta64(84, "h"), np.timedelta64(96, "h")],
            "latitude": [52.0, 51.0],
            "longitude": [4.0, 5.0],
        },
    )

    with patch('cfgrib.open_datasets') as mock_open:
        # Mock open return
        mock_open.return_value = [ds]
        
        # Max step is 84
        out_ds = process_ecmwf_live(
            grib_path="dummy.grib",
            bbox_nwse=[53, 3, 50, 6],
            max_step_hours=84
        )
        
        assert "step" in out_ds.coords
        # 96h should be dropped
        assert np.timedelta64(96, "h") not in out_ds.step.values
        assert np.timedelta64(84, "h") in out_ds.step.values
