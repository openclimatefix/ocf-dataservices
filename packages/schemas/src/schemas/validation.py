import functools
import os
from collections.abc import Callable
from pathlib import Path
from typing import ParamSpec

import pandera.xarray as pa
import xarray as xr

P = ParamSpec("P")

MIN_GRIB_SIZE_BYTES = 100_000


def is_valid_grib_file(
    path: Path | str,
    min_size_bytes: int = MIN_GRIB_SIZE_BYTES,
) -> bool:
    """Return True if `path` exists, is >= `min_size_bytes`, starts with `'GRIB'`, and ends with `'7777'`."""
    file_path = Path(path)
    try:
        if not file_path.is_file() or file_path.stat().st_size < min_size_bytes:
            return False
        with file_path.open("rb") as f:
            if f.read(4) != b"GRIB":
                return False
            f.seek(-4, os.SEEK_END)
            return f.read(4) == b"7777"
    except OSError:
        return False


def _describe_null_variables(
    ds: xr.Dataset,
    schema: type[pa.DatasetModel],
    exc: Exception,
) -> str | None:
    """Return a summary of non-nullable variables that contain null values, or `None`.

    Inspects each variable in `ds` when schema validation fails to report the exact variable names
    and null counts.
    """
    schema_vars = schema.to_schema().data_vars or {}
    is_null_error = "null" in str(exc).lower()
    null_vars = []
    for var_name, data_array in ds.data_vars.items():
        if data_array.size == 0 or not bool(data_array.isnull().any()):
            continue
        var_spec = schema_vars.get(str(var_name))
        disallowed_by_field = var_spec is not None and not getattr(var_spec, "nullable", True)
        if disallowed_by_field or is_null_error:
            null_count = int(data_array.isnull().sum().item())
            total_count = int(data_array.size)
            null_vars.append(
                f"{var_name} ({null_count}/{total_count} NaNs, {null_count / total_count:.1%})"
            )

    if not null_vars:
        return None
    return "Validation failed. The following variables contain null values:\n" + "\n".join(
        null_vars
    )


def validates(
    schema: type[pa.DatasetModel],
) -> Callable[[Callable[P, xr.Dataset]], Callable[P, xr.Dataset]]:
    """Validate a function's returned `xarray.Dataset` against `schema`.

    Apply this decorator to the public entrypoint of each provider package to guarantee that all
    returned datasets satisfy the target schema.
    """

    def decorator(func: Callable[P, xr.Dataset]) -> Callable[P, xr.Dataset]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> xr.Dataset:
            ds = func(*args, **kwargs)
            try:
                return schema.validate(ds)
            except Exception as exc:
                message = _describe_null_variables(ds, schema, exc)
                if message is not None:
                    raise ValueError(message) from exc
                raise

        return wrapper

    return decorator
