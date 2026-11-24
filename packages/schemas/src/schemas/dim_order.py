from collections.abc import Sequence

import xarray as xr


def enforce_dim_order(
    ds: xr.Dataset,
    dims: Sequence[str],
    keep_vars: Sequence[str] | None = None,
) -> xr.Dataset:
    """Rebuild `ds` with an explicit dimension order.

    Enforces a deterministic dimension order on coordinates and data variables so that downstream
    positional Zarr chunk and shard specifications match the dataset dimensions.

    Args:
        ds: Input dataset to reorder.
        dims: Target dimension order, as declared by a schema's `dims()` method.
        keep_vars: Optional list of data variable names to retain in the returned dataset.
    """
    new_coords: dict[str, xr.Variable] = {d: ds.coords[d].variable for d in dims if d in ds.coords}
    for c in ds.coords:
        c_str = str(c)
        if c_str not in new_coords:
            new_coords[c_str] = ds.coords[c].variable

    ordered = xr.Dataset(
        data_vars={
            str(name): da.transpose(*[d for d in dims if d in da.dims]).variable
            for name, da in ds.data_vars.items()
        },
        coords=new_coords,
    )
    ordered = ordered.transpose(*dims)

    if keep_vars is not None:
        ordered = ordered[[v for v in keep_vars if v in ordered.data_vars]]

    if list(ordered.dims) != list(dims):
        raise ValueError(
            f"Failed to enforce dimension order. Expected {list(dims)}, got {list(ordered.dims)}"
        )

    return ordered
