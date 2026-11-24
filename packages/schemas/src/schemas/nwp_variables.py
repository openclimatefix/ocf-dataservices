from typing import Any

import pandera.xarray as pa
import xarray as xr


def scale_to_percent(da: xr.DataArray, max_val: float = 100.0) -> xr.DataArray:
    """Scale `[0, 1]` fraction fields to `[0, 100]` percent using GRIB unit metadata."""
    units = str(da.attrs.get("GRIB_units") or da.attrs.get("units") or "").strip()
    if units in ("(0 - 1)", "1"):
        scaled = da * 100.0
    elif units == "%":
        scaled = da
    else:
        # Fallback when GRIB unit metadata is absent or 'unknown' (e.g. local GRIB2 codes)
        scaled = da * 100.0 if float(da.max()) <= 1.1 else da
    return scaled.clip(min=0.0, max=max_val)


def temperature_2m(dims: tuple[str, ...], nullable: bool = False) -> Any:
    return pa.Field(dims=dims, ge=-100, le=100, nullable=nullable)


def wind_u_100m(dims: tuple[str, ...], nullable: bool = False) -> Any:
    return pa.Field(dims=dims, ge=-115, le=115, nullable=nullable)


def wind_v_100m(dims: tuple[str, ...], nullable: bool = False) -> Any:
    return pa.Field(dims=dims, ge=-115, le=115, nullable=nullable)


def pressure_reduced_to_mean_sea_level(dims: tuple[str, ...], nullable: bool = False) -> Any:
    return pa.Field(dims=dims, ge=80000, le=115000, nullable=nullable)


def total_cloud_cover_atmosphere(dims: tuple[str, ...], nullable: bool = False) -> Any:
    return pa.Field(dims=dims, ge=0, le=100, nullable=nullable)


def downward_long_wave_radiation_flux_surface(dims: tuple[str, ...], nullable: bool = True) -> Any:
    return pa.Field(dims=dims, ge=0, le=2000, nullable=nullable)


def downward_short_wave_radiation_flux_surface(dims: tuple[str, ...], nullable: bool = True) -> Any:
    return pa.Field(dims=dims, ge=0, le=2000, nullable=nullable)


def categorical_precipitation_type_surface(dims: tuple[str, ...], nullable: bool = True) -> Any:
    return pa.Field(dims=dims, ge=0, le=255, nullable=nullable)


def wind_v_10m(dims: tuple[str, ...], nullable: bool = True) -> Any:
    return pa.Field(dims=dims, ge=-115, le=115, nullable=nullable)


def wind_u_10m(dims: tuple[str, ...], nullable: bool = True) -> Any:
    return pa.Field(dims=dims, ge=-115, le=115, nullable=nullable)


def high_cloud_cover(dims: tuple[str, ...], nullable: bool = False) -> Any:
    return pa.Field(dims=dims, ge=0, le=100, nullable=nullable)


def medium_cloud_cover(dims: tuple[str, ...], nullable: bool = False) -> Any:
    return pa.Field(dims=dims, ge=0, le=100, nullable=nullable)


def low_cloud_cover(dims: tuple[str, ...], nullable: bool = False) -> Any:
    return pa.Field(dims=dims, ge=0, le=100, nullable=nullable)


def wind_u_200m(dims: tuple[str, ...], nullable: bool = False) -> Any:
    return pa.Field(dims=dims, ge=-150, le=150, nullable=nullable)


def wind_v_200m(dims: tuple[str, ...], nullable: bool = False) -> Any:
    return pa.Field(dims=dims, ge=-150, le=150, nullable=nullable)


def direct_solar_radiation(dims: tuple[str, ...], nullable: bool = True) -> Any:
    return pa.Field(dims=dims, ge=0, le=2000, nullable=nullable)


def uv_b_radiation(dims: tuple[str, ...], nullable: bool = True) -> Any:
    return pa.Field(dims=dims, ge=0, le=2000, nullable=nullable)


def total_precipitation_rate(dims: tuple[str, ...], nullable: bool = True) -> Any:
    return pa.Field(dims=dims, ge=0, le=1, nullable=nullable)


def snow_depth(dims: tuple[str, ...], nullable: bool = True) -> Any:
    return pa.Field(dims=dims, ge=0, le=100, nullable=nullable)


def visibility(dims: tuple[str, ...], nullable: bool = True) -> Any:
    return pa.Field(dims=dims, ge=0, le=100000, nullable=nullable)


def relative_humidity_2m(dims: tuple[str, ...], nullable: bool = False) -> Any:
    return pa.Field(
        dims=dims, ge=0, le=120, nullable=nullable
    )  # Supersaturation means > 100% is posible

