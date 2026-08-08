from collections.abc import Sequence

import xarray as xr


def enforce_dim_order(
    ds: xr.Dataset,
    dims: Sequence[str],
    keep_vars: Sequence[str] | None = None,
) -> xr.Dataset:
    """Rebuild `ds` with an explicit dimension order.

    xarray's dimension insertion order for coordinates/data_vars is otherwise implicit and can
    silently drift from what downstream IOManagers (e.g. zarr chunk/shard specs, which are keyed
    positionally by dimension order) expect.

    Args:
        ds: The dataset to reorder.
        dims: The dimension order to enforce, e.g. as declared by a schema's `dims()`.
        keep_vars: If given, restrict the output to these data variables (in this order, filtered
            to those actually present).
    """
    new_coords = {d: ds.coords[d].variable for d in dims if d in ds.coords}
    for c in ds.coords:
        if c not in new_coords:
            new_coords[c] = ds.coords[c].variable

    ordered = xr.Dataset(
        data_vars={
            name: da.transpose(*[d for d in dims if d in da.dims]).variable
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
