import datetime as dt
import fcntl
import re
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import icechunk
import numpy as np
import xarray as xr
import zarr.codecs
from schemas.base import NwpDatasetSchema

SPATIAL_DIMS = ("latitude", "longitude", "y_laea", "x_laea")
STATS = ("mean", "std", "min", "max")


@dataclass(frozen=True)
class WriteResult:
    """Summary of an Icechunk write operation."""

    store_path: str
    commit_hash: str
    skip_write: bool
    size_bytes: int
    written_dims: dict[str, int]
    spatial_stats: xr.Dataset | None
    temporal_stats: xr.Dataset | None


def _compute_stats_reduction(
    ds: xr.Dataset,
    reduce_dims: list[str],
    append_dim: str,
) -> xr.Dataset | None:
    if not reduce_dims:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        parts = [
            ds.mean(dim=reduce_dims, skipna=True),
            ds.std(dim=reduce_dims, skipna=True),
            ds.min(dim=reduce_dims, skipna=True),
            ds.max(dim=reduce_dims, skipna=True),
        ]
    stat_coord = xr.DataArray(list(STATS), dims="stat", name="stat")
    reduced = xr.concat(parts, dim=stat_coord)
    remaining_dims = [str(d) for d in ds.dims if d not in reduce_dims]
    target_dims = [append_dim, "stat"] + [d for d in remaining_dims if d != append_dim]
    return reduced.transpose(*target_dims).astype(np.float32)


def _write_stats_group(
    store: icechunk.IcechunkStore,
    reduced: xr.Dataset | None,
    group: str,
    append_dim: str,
    deduplicate: bool,
) -> bool:
    if reduced is None:
        return False
    group_exists = False
    existing_group = None
    try:
        existing_group = xr.open_zarr(store, group=group, consolidated=False)
        group_exists = bool(existing_group.data_vars)
    except Exception:  # noqa: BLE001
        group_exists = False

    if deduplicate and group_exists and existing_group is not None:
        existing_times = existing_group[append_dim].values.astype("datetime64[ns]")
        incoming_times = reduced[append_dim].values.astype("datetime64[ns]")
        is_new = ~np.isin(incoming_times, existing_times)
        if not is_new.any():
            return False
        reduced_to_write = reduced.isel({append_dim: is_new}).sortby(append_dim)
        reduced_to_write.to_zarr(
            store,
            group=group,
            mode="a",
            append_dim=append_dim,
            write_empty_chunks=False,
        )
        return True

    reduced_to_write = reduced.sortby(append_dim) if deduplicate else reduced
    reduced_to_write.to_zarr(
        store,
        group=group,
        zarr_format=3,
        mode="w-",
        write_empty_chunks=False,
        encoding={
            var: {
                "dtype": "float32",
                "chunks": tuple(
                    1 if d == append_dim else reduced_to_write.coords[d].size
                    for d in reduced_to_write.dims
                ),
                "compressors": zarr.codecs.BloscCodec(
                    cname="zstd",
                    clevel=3,
                    shuffle="bitshuffle",
                ),
            }
            for var in reduced_to_write.data_vars
        }
        | {
            coord: {"chunks": 10000}
            for coord in reduced_to_write.coords
            if coord not in reduced_to_write.dims
        }
        | {
            append_dim: {
                "dtype": int,
                "units": "nanoseconds since 1970-01-01",
                "calendar": "proleptic_gregorian",
                "chunks": 10000,
            }
        },
    )
    return True


def open_icechunk_repo(
    store_path: str,
    create_if_not_exists: bool = False,
    aws_access_key_id: str = "",
    aws_secret_access_key: str = "",
    aws_region_name: str = "",
    aws_endpoint_url: str | None = None,
) -> tuple[icechunk.Repository, bool]:
    """Open (or optionally create) an Icechunk repository on local disk or S3."""
    result = re.match(
        r"^(?P<protocol>[\w]{2,6}):\/\/(?P<bucket>[\w.-]+)\/(?P<prefix>[\w.\/-]+)$",
        store_path,
    )
    if result:
        match (result.group("protocol"), result.group("bucket"), result.group("prefix")):
            case ("s3", bucket, prefix):
                storage = icechunk.s3_storage(
                    bucket=bucket,
                    prefix=prefix,
                    access_key_id=aws_access_key_id,
                    secret_access_key=aws_secret_access_key,
                    region=aws_region_name,
                    endpoint_url=aws_endpoint_url,
                )
            case (_, _, _):
                raise ValueError(f"Unsupported storage protocol: {result.group('protocol')}")
    elif "://" in store_path:
        raise ValueError(f"Invalid store URI: {store_path}")
    else:
        storage = icechunk.local_filesystem_storage(path=store_path)

    created = False
    if icechunk.Repository.exists(storage):
        repo = icechunk.Repository.open(storage)
    elif create_if_not_exists:
        try:
            repo = icechunk.Repository.create(storage)
            created = True
        except Exception as e:
            raise OSError(f"Failed to create repository at storage: {storage}. Error: {e}") from e
    else:
        raise OSError(f"Repository does not exist at storage: {storage}")
    return repo, created


def _to_utc_datetime64(value: str | dt.datetime | np.datetime64) -> np.datetime64:
    if isinstance(value, np.datetime64):
        return value.astype("datetime64[ns]")
    if isinstance(value, dt.datetime):
        if value.tzinfo is not None:
            value = value.astimezone(dt.UTC).replace(tzinfo=None)
        return np.datetime64(value, "ns")
    parsed = dt.datetime.fromisoformat(value)
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(dt.UTC).replace(tzinfo=None)
    return np.datetime64(parsed, "ns")


def existing_partition(
    store_path: str,
    append_dim: str,
    partition_key: str | dt.datetime | np.datetime64,
    aws_access_key_id: str = "",
    aws_secret_access_key: str = "",
    aws_region_name: str = "",
    aws_endpoint_url: str | None = None,
) -> xr.Dataset | None:
    """Return the slice for `partition_key` if it already exists in `store_path`, else None."""
    target_date = _to_utc_datetime64(partition_key)
    try:
        repo, _ = open_icechunk_repo(
            store_path,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_region_name=aws_region_name,
            aws_endpoint_url=aws_endpoint_url,
        )
    except OSError:
        return None

    try:
        session = repo.readonly_session(branch="main")
        ds = xr.open_zarr(session.store, consolidated=False)
        if append_dim in ds.dims and target_date in ds[append_dim].values.astype("datetime64[ns]"):
            return ds.sel({append_dim: target_date})
        return None
    except Exception:  # noqa: BLE001
        return None


def has_partition(
    store_path: str,
    append_dim: str,
    partition_key: str | dt.datetime | np.datetime64,
    aws_access_key_id: str = "",
    aws_secret_access_key: str = "",
    aws_region_name: str = "",
    aws_endpoint_url: str | None = None,
) -> bool:
    """Return True if `partition_key` already exists along `append_dim` in `store_path`."""
    return (
        existing_partition(
            store_path=store_path,
            append_dim=append_dim,
            partition_key=partition_key,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_region_name=aws_region_name,
            aws_endpoint_url=aws_endpoint_url,
        )
        is not None
    )


@contextmanager
def _local_write_lock(path: str):
    if re.match(r"^[\w]{2,6}://", path):
        yield
        return
    lock_path = Path(path).with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def write_to_icechunk(
    ds: xr.Dataset,
    schema: type[NwpDatasetSchema],
    store_path: str,
    deduplicate: bool = True,
    keep_mantissa_bits: int | None = 14,
    write_stats: bool = True,
    commit_message: str | None = None,
    aws_access_key_id: str = "",
    aws_secret_access_key: str = "",
    aws_region_name: str = "",
    aws_endpoint_url: str | None = None,
) -> WriteResult:
    """Write or append an xarray Dataset to an Icechunk repository using `schema`'s layout."""
    for key, spec in (("chunks", schema._chunks), ("shards", schema._shards)):
        if list(ds.dims) != list(spec.keys()):
            raise ValueError(
                f"Dataset dimensions {list(ds.dims)} do not match supplied {key} keys "
                f"{list(spec.keys())}"
            )

    append_dim = schema.append_dim()
    spatial_stats: xr.Dataset | None = None
    temporal_stats: xr.Dataset | None = None
    if write_stats:
        spatial_dims = [str(d) for d in ds.dims if d in SPATIAL_DIMS]
        temporal_dims = [str(d) for d in ds.dims if d != append_dim and d not in SPATIAL_DIMS]
        spatial_stats = _compute_stats_reduction(ds, spatial_dims, append_dim)
        temporal_stats = _compute_stats_reduction(ds, temporal_dims, append_dim)

    obj = ds
    if keep_mantissa_bits is not None:
        mask = np.int32(~((1 << (23 - keep_mantissa_bits)) - 1))

        def apply_veltkamp(da: xr.DataArray) -> xr.DataArray:
            if da.dtype == np.float32:
                if hasattr(da.data, "map_blocks"):
                    new_data = da.data.map_blocks(
                        lambda x: np.bitwise_and(x.view(np.int32), mask).view(np.float32),
                        dtype=np.float32,
                    )
                else:
                    new_data = np.bitwise_and(da.values.view(np.int32), mask).view(np.float32)
                return da.copy(data=new_data)
            return da

        obj = obj.map(apply_veltkamp, keep_attrs=True)

    with _local_write_lock(store_path):
        repo, _was_created = open_icechunk_repo(
            store_path,
            create_if_not_exists=True,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_region_name=aws_region_name,
            aws_endpoint_url=aws_endpoint_url,
        )
        session = repo.writable_session(branch="main")

        dataset_exists = False
        existing_ds = None
        try:
            existing_ds = xr.open_zarr(session.store, consolidated=False)
            dataset_exists = True
        except Exception:  # noqa: BLE001
            dataset_exists = False

        obj_to_write = obj
        skip_write = False

        if deduplicate and dataset_exists and existing_ds is not None:
            existing_times = existing_ds[append_dim].values.astype("datetime64[ns]")
            incoming_times = obj[append_dim].values.astype("datetime64[ns]")
            is_new = ~np.isin(incoming_times, existing_times)

            if not is_new.any():
                skip_write = True
            else:
                obj_to_write = obj.isel({append_dim: is_new}).sortby(append_dim)
                obj_to_write.to_zarr(
                    session.store,
                    mode="a",
                    append_dim=append_dim,
                    write_empty_chunks=False,
                )
        else:
            obj.to_zarr(
                session.store,
                zarr_format=3,
                mode="w-",
                write_empty_chunks=False,
                encoding={
                    var: {
                        "dtype": "float32",
                        "chunks": tuple(
                            obj.coords[k].size if v == -1 else v
                            for k, v in schema._chunks.items()
                        ),
                        "shards": tuple(
                            obj.coords[k].size if v == -1 else v
                            for k, v in schema._shards.items()
                        ),
                        "compressors": zarr.codecs.BloscCodec(
                            cname="zstd",
                            clevel=3,
                            shuffle="bitshuffle",
                        ),
                    }
                    for var in obj.data_vars
                }
                | {coord: {"chunks": 10000} for coord in obj.coords if coord not in obj.dims}
                | {
                    append_dim: {
                        "dtype": int,
                        "units": "nanoseconds since 1970-01-01",
                        "calendar": "proleptic_gregorian",
                        "chunks": 10000,
                    }
                },
            )

        wrote_spatial = _write_stats_group(
            session.store,
            spatial_stats,
            "stats_spatial",
            append_dim,
            deduplicate,
        )
        wrote_temporal = _write_stats_group(
            session.store,
            temporal_stats,
            "stats_temporal",
            append_dim,
            deduplicate,
        )

        if not skip_write or wrote_spatial or wrote_temporal:
            msg = commit_message or f"icechunk write at {dt.datetime.now(dt.UTC)}"
            commit_hash = str(session.commit(msg))
        else:
            commit_hash = "N/A (Skipped duplicates)"

    size_bytes = obj_to_write.nbytes if not skip_write else 0
    written_dims = (
        {str(k): int(v) for k, v in obj_to_write.sizes.items()}
        if not skip_write
        else {**{str(k): int(v) for k, v in obj.sizes.items()}, append_dim: 0}
    )

    return WriteResult(
        store_path=store_path,
        commit_hash=commit_hash,
        skip_write=skip_write,
        size_bytes=size_bytes,
        written_dims=written_dims,
        spatial_stats=spatial_stats,
        temporal_stats=temporal_stats,
    )
