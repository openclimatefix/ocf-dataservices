import functools
from collections.abc import Callable
from typing import ParamSpec

import pandera.xarray as pa
import xarray as xr

P = ParamSpec("P")


def _describe_null_variables(ds: xr.Dataset) -> str | None:
    """Build a human-readable summary of which variables contain nulls, or None if there are none.

    Pandera doesn't always surface the exact failing variable/value in its top-level exception, so
    on validation failure we check each variable ourselves to give a clearer error message.
    """
    null_vars = []
    for var_name, data_array in ds.data_vars.items():
        if bool(data_array.isnull().any()):
            null_count = int(data_array.isnull().sum().item())
            total_count = int(data_array.size)
            null_vars.append(f"{var_name} ({null_count}/{total_count} NaNs, {null_count / total_count:.1%})")

    if not null_vars:
        return None
    return "Validation failed. The following variables contain null values:\n" + "\n".join(null_vars)


def validates(schema: type[pa.DatasetModel]) -> Callable[[Callable[P, xr.Dataset]], Callable[P, xr.Dataset]]:
    """Decorator that validates a function's returned Dataset against `schema` before returning it.

    Applying this at a provider package's public entrypoint (its download/convert/process function)
    guarantees every caller receives data that has passed schema validation, rather than relying on
    the caller to remember to invoke `Schema.validate()` themselves.
    """

    def decorator(func: Callable[P, xr.Dataset]) -> Callable[P, xr.Dataset]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> xr.Dataset:
            ds = func(*args, **kwargs)
            try:
                return schema.validate(ds)
            except Exception as exc:
                message = _describe_null_variables(ds)
                if message is not None:
                    raise ValueError(message) from exc
                raise

        return wrapper

    return decorator
