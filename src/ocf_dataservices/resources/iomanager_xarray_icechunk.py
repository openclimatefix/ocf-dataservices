import base64
import datetime as dt
import fcntl
import html
import re
import warnings
from contextlib import contextmanager
from pathlib import Path

import dagster as dg
import icechunk
import numpy as np
import xarray as xr
import zarr.codecs

_SPATIAL_DIMS = ("latitude", "longitude", "y_laea", "x_laea")
_STATS = ("mean", "std", "min", "max")


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
    stat_coord = xr.DataArray(list(_STATS), dims="stat", name="stat")
    reduced = xr.concat(parts, dim=stat_coord)
    remaining_dims = [str(d) for d in ds.dims if d not in reduce_dims]
    target_dims = [append_dim, "stat"] + [d for d in remaining_dims if d != append_dim]
    return reduced.transpose(*target_dims).astype(np.float32)


def _write_stats_group(
    store: icechunk.IcechunkStore,
    reduced: xr.Dataset | None,
    group: str,
    append_dim: str,
    has_partition_key: bool,
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

    if has_partition_key and group_exists and existing_group is not None:
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

    reduced_to_write = reduced.sortby(append_dim) if has_partition_key else reduced
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


def _render_var_svg(
    var: str,
    step_hours: np.ndarray,
    mean_s: np.ndarray,
    std_s: np.ndarray,
    min_s: np.ndarray,
    max_s: np.ndarray,
    grid_2d: np.ndarray | None,
) -> str:
    w, h = 520, 140
    lx0, lx1, ly0, ly1 = 44, 290, 24, 116
    vmin = float(np.nanmin(min_s)) if np.isfinite(min_s).any() else 0.0
    vmax = float(np.nanmax(max_s)) if np.isfinite(max_s).any() else 1.0
    span = (vmax - vmin) or 1.0
    n = max(len(step_hours), 1)

    def px(i: int) -> float:
        return lx0 + (lx1 - lx0) * (i / max(n - 1, 1))

    def py(v: float) -> float:
        val = np.clip((float(v) - vmin) / span, 0.0, 1.0) if np.isfinite(v) else 0.5
        return ly1 - (ly1 - ly0) * val

    idx = [i for i in range(n) if np.isfinite(mean_s[i])]
    minmax_pts = " ".join(
        [f"{px(i):.1f},{py(max_s[i]):.1f}" for i in idx]
        + [f"{px(i):.1f},{py(min_s[i]):.1f}" for i in reversed(idx)]
    )
    std_pts = " ".join(
        [f"{px(i):.1f},{py(mean_s[i] + std_s[i]):.1f}" for i in idx]
        + [f"{px(i):.1f},{py(mean_s[i] - std_s[i]):.1f}" for i in reversed(idx)]
    )
    line_pts = " ".join(f"{px(i):.1f},{py(mean_s[i]):.1f}" for i in idx)

    rects: list[str] = []
    if grid_2d is not None and grid_2d.ndim == 2 and grid_2d.size > 0:
        sy = max(1, grid_2d.shape[0] // 20)
        sx = max(1, grid_2d.shape[1] // 20)
        small = grid_2d[::sy, ::sx]
        ny, nx = small.shape
        gx0, gy0, gw, gh = 330.0, 24.0, 170.0, 92.0
        cw, ch = gw / nx, gh / ny
        gmin = float(np.nanmin(small)) if np.isfinite(small).any() else 0.0
        gmax = float(np.nanmax(small)) if np.isfinite(small).any() else 1.0
        gspan = (gmax - gmin) or 1.0
        for r in range(ny):
            for c in range(nx):
                val = small[r, c]
                if not np.isfinite(val):
                    continue
                t = float(np.clip((val - gmin) / gspan, 0.0, 1.0))
                red = int(40 + 200 * t)
                grn = int(80 + 100 * (1.0 - abs(2.0 * t - 1.0)))
                blu = int(220 - 180 * t)
                rects.append(
                    f'<rect x="{gx0 + c * cw:.1f}" y="{gy0 + r * ch:.1f}" '
                    f'width="{cw + 0.4:.1f}" height="{ch + 0.4:.1f}" '
                    f'fill="rgb({red},{grn},{blu})"/>'
                )

    t_max = float(step_hours[-1]) if len(step_hours) else 0.0
    title = html.escape(var)
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" style="font-family:sans-serif;font-size:10px;background:#fff">'
        f'<text x="8" y="14" font-weight="bold" font-size="11" fill="#111">{title}</text>'
        f'<rect x="{lx0}" y="{ly0}" width="{lx1 - lx0}" height="{ly1 - ly0}" '
        f'fill="#f8fafc" stroke="#cbd5e1"/>'
        f'<polygon points="{minmax_pts}" fill="#93c5fd" fill-opacity="0.35"/>'
        f'<polygon points="{std_pts}" fill="#3b82f6" fill-opacity="0.35"/>'
        f'<polyline points="{line_pts}" fill="none" stroke="#1d4ed8" stroke-width="1.5"/>'
        f'<text x="{lx0 - 3}" y="{ly0 + 8}" text-anchor="end" fill="#475569">{vmax:.3g}</text>'
        f'<text x="{lx0 - 3}" y="{ly1}" text-anchor="end" fill="#475569">{vmin:.3g}</text>'
        f'<text x="{lx0}" y="{ly1 + 12}" fill="#475569">0h</text>'
        f'<text x="{lx1}" y="{ly1 + 12}" text-anchor="end" fill="#475569">{t_max:.0f}h</text>'
        f'<text x="330" y="14" fill="#475569">Spatial Mean</text>'
        + "".join(rects)
        + "</svg>"
    )
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")


def _build_stats_metadata(
    spatial_stats: xr.Dataset | None,
    temporal_stats: xr.Dataset | None,
) -> dict[str, dg.MetadataValue]:
    if spatial_stats is None:
        return {}

    records: list[dg.TableRecord] = []
    md_parts: list[str] = ["### Dataset Statistics Overview\n"]
    extra_dims = [d for d in spatial_stats.dims if d not in ("stat", "step")]
    step_vals = (
        spatial_stats.coords["step"].values / np.timedelta64(1, "h")
        if "step" in spatial_stats.coords
        else np.arange(1, dtype=np.float64)
    )

    for var in spatial_stats.data_vars:
        var_name = str(var)
        da = spatial_stats[var_name]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_s = da.sel(stat="mean").mean(dim=extra_dims, skipna=True).values.ravel()
            std_s = da.sel(stat="std").mean(dim=extra_dims, skipna=True).values.ravel()
            min_s = da.sel(stat="min").min(dim=extra_dims, skipna=True).values.ravel()
            max_s = da.sel(stat="max").max(dim=extra_dims, skipna=True).values.ravel()

        records.append(
            dg.TableRecord(
                {
                    "variable": var_name,
                    "mean": round(float(np.nanmean(mean_s)), 4)
                    if np.isfinite(mean_s).any()
                    else 0.0,
                    "std": round(float(np.nanmean(std_s)), 4)
                    if np.isfinite(std_s).any()
                    else 0.0,
                    "min": round(float(np.nanmin(min_s)), 4)
                    if np.isfinite(min_s).any()
                    else 0.0,
                    "max": round(float(np.nanmax(max_s)), 4)
                    if np.isfinite(max_s).any()
                    else 0.0,
                }
            )
        )

        grid_2d = None
        if temporal_stats is not None and var_name in temporal_stats.data_vars:
            t_da = temporal_stats[var_name].sel(stat="mean")
            t_extra = [d for d in t_da.dims if d not in _SPATIAL_DIMS]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                t_2d = t_da.mean(dim=t_extra, skipna=True) if t_extra else t_da
            if t_2d.ndim == 2:
                y_dim, x_dim = str(t_2d.dims[0]), str(t_2d.dims[1])
                grid_2d = t_2d.values
                if (
                    t_2d.coords[y_dim].size > 1
                    and t_2d.coords[y_dim].values[0] < t_2d.coords[y_dim].values[-1]
                ):
                    grid_2d = grid_2d[::-1, :]
                if (
                    t_2d.coords[x_dim].size > 1
                    and t_2d.coords[x_dim].values[0] > t_2d.coords[x_dim].values[-1]
                ):
                    grid_2d = grid_2d[:, ::-1]

        uri = _render_var_svg(var_name, step_vals, mean_s, std_s, min_s, max_s, grid_2d)
        md_parts.append(f"![{var_name}]({uri})")

    return {
        "stats_summary": dg.MetadataValue.table(records=records),
        "stats_overview": dg.MetadataValue.md("\n\n".join(md_parts)),
    }


class XarrayIcechunkIOManager(dg.ConfigurableIOManager):
    path: str = ""
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region_name: str = ""
    aws_endpoint_url: str | None = None
    keep_mantissa_bits: int | None = 14

    def _get_store_path(self, asset_key: dg.AssetKey) -> str:
        return "/".join([self.path] + list(asset_key.path))

    def _get_repo(
        self, store_path: str, create_if_not_exists: bool = False
    ) -> tuple[icechunk.Repository, bool]:
        result = re.match(
            r"^(?P<protocol>[\w]{2,6}):\/\/(?P<bucket>[\w-]+)\/(?P<prefix>[\w.\/-]+)$",
            store_path,
        )
        if result:
            match (result.group("protocol"), result.group("bucket"), result.group("prefix")):
                case ("s3", bucket, prefix):
                    storage = icechunk.s3_storage(
                        bucket=bucket,
                        prefix=prefix,
                        access_key_id=self.aws_access_key_id,
                        secret_access_key=self.aws_secret_access_key,
                        region=self.aws_region_name,
                        endpoint_url=self.aws_endpoint_url,
                    )
                case (_, _, _):
                    raise ValueError(f"Unsupported storage protocol: {result.group('protocol')}")
        else:
            storage = icechunk.local_filesystem_storage(
                path=store_path,
            )

        created: bool = False
        if icechunk.Repository.exists(storage):
            repo = icechunk.Repository.open(storage)
        elif create_if_not_exists:
            try:
                repo = icechunk.Repository.create(storage)
                created = True
            except Exception as e:  # noqa: BLE001
                raise OSError(f"Failed to create repository at storage: {storage}. Error: {e}")
        else:
            raise OSError(f"Repository does not exist at storage: {storage}")
        return repo, created

    def existing_partition(
        self,
        asset_key: dg.AssetKey,
        append_dim: str,
        partition_key: str,
    ) -> xr.Dataset | None:
        """Gets a partition if it already exists, otherwise returns None."""
        store_path = self._get_store_path(asset_key)
        try:
            repo, _ = self._get_repo(store_path)
        except OSError:
            # The repo doesn't exist, so the partition can't exist either
            return None

        try:
            session = repo.readonly_session(branch="main")
            ds = xr.open_zarr(session.store, consolidated=False)

            # Convert string "YYYY-MM-DD" to numpy datetime64 to match xarray indices
            target_date = np.datetime64(partition_key)

            # Check if the dimension exists and if the target date is in it
            if append_dim in ds.dims and target_date in ds[append_dim].values:
                return ds.sel({append_dim: target_date})
            return None
        except Exception:  # noqa: BLE001
            # If the repo, store, or dimension doesn't exist yet, it's not materialized
            return None

    def handle_output(self, context: dg.OutputContext, obj: object) -> None:
        if not isinstance(obj, xr.Dataset):
            raise dg.DagsterInvariantViolationError(
                f"XarrayIcechunkIOManager can only handle xr.Dataset objects, got {type(obj)}"
            )

        metadata = context.definition_metadata or {}
        schema = metadata["schema"]
        for key, spec in (("chunks", schema._chunks), ("shards", schema._shards)):
            if list(obj.dims) != list(spec.keys()):
                raise dg.DagsterInvariantViolationError(
                    f"Dataset dimensions {list(obj.dims)} do not match supplied {key} keys "
                    + f"{list(spec.keys())}"
                )

        append_dim = schema.append_dim()
        spatial_dims = [str(d) for d in obj.dims if d in _SPATIAL_DIMS]
        temporal_dims = [
            str(d) for d in obj.dims if d != append_dim and d not in _SPATIAL_DIMS
        ]
        spatial_stats = _compute_stats_reduction(obj, spatial_dims, append_dim)
        temporal_stats = _compute_stats_reduction(obj, temporal_dims, append_dim)

        keep_bits = metadata.get("keep_mantissa_bits", self.keep_mantissa_bits)
        if keep_bits is not None:
            mask = np.int32(~((1 << (23 - keep_bits)) - 1))

            def apply_veltkamp(da: xr.DataArray) -> xr.DataArray:
                if da.dtype == np.float32:
                    if hasattr(da.data, "map_blocks"):
                        # Support Dask-backed datasets without computing them into memory
                        new_data = da.data.map_blocks(
                            lambda x: np.bitwise_and(x.view(np.int32), mask).view(np.float32),
                            dtype=np.float32,
                        )
                    else:
                        # In-memory NumPy datasets
                        new_data = np.bitwise_and(da.values.view(np.int32), mask).view(np.float32)
                    return da.copy(data=new_data)
                return da

            obj = obj.map(apply_veltkamp, keep_attrs=True)

        store_path = self._get_store_path(context.asset_key)

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

        with _local_write_lock(store_path):
            repo, _was_created = self._get_repo(store_path, create_if_not_exists=True)
            session = repo.writable_session(branch="main")
    
            dataset_exists = False
            existing_ds = None
            try:
                # Check if a valid zarr dataset actually exists in the store.
                # (An icechunk repo might exist without a zarr dataset inside it if a previous run failed).
                existing_ds = xr.open_zarr(session.store, consolidated=False)
                dataset_exists = True
            except Exception:  # noqa: BLE001
                dataset_exists = False
    
            obj_to_write = obj
            skip_write = False

            if context.has_partition_key and dataset_exists:
                existing_times = existing_ds[append_dim].values.astype("datetime64[ns]")
                incoming_times = obj[append_dim].values.astype("datetime64[ns]")
                is_new = ~np.isin(incoming_times, existing_times)
                
                if not is_new.any():
                    skip_write = True
                else:
                    obj_to_write = obj.isel({append_dim: is_new})
                    obj_to_write = obj_to_write.sortby(append_dim)
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
                                [
                                    obj.coords[k].size if v == -1 else v
                                    for k, v in schema._chunks.items()
                                ]
                            ),
                            "shards": tuple(
                                [
                                    obj.coords[k].size if v == -1 else v
                                    for k, v in schema._shards.items()
                                ]
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
                        schema.append_dim(): {
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
                context.has_partition_key,
            )
            wrote_temporal = _write_stats_group(
                session.store,
                temporal_stats,
                "stats_temporal",
                append_dim,
                context.has_partition_key,
            )

            if not skip_write or wrote_spatial or wrote_temporal:
                commit_hash = session.commit(
                    f"dagster materialization: {context.asset_key} at {dt.datetime.now(dt.UTC)}"
                )
            else:
                commit_hash = "N/A (Skipped duplicates)"

        size_bytes = obj_to_write.nbytes if not skip_write else 0
        written_dims = dict(obj_to_write.dims) if not skip_write else {**dict(obj.dims), schema.append_dim(): 0}
        
        metadata = {
            "store_path": dg.MetadataValue.path(store_path),
            "partition_key": context.partition_key if context.has_partition_key else "N/A",
            "data_vars": list(obj.data_vars),
            "dims": written_dims,
            "icechunk_commit": dg.MetadataValue.text(str(commit_hash)),
            "size_in_memory_bytes": dg.MetadataValue.int(size_bytes),
            "size_human_readable": dg.MetadataValue.text(
                f"{size_bytes / (1024 * 1024):.2f} MB"
            ),
        } | _build_stats_metadata(spatial_stats, temporal_stats)
        
        if skip_write:
            metadata["duplicate_partition_skipped"] = dg.MetadataValue.bool(True)
            
        context.add_output_metadata(metadata)

    def load_input(self, context: dg.InputContext) -> xr.Dataset:
        store_path = self._get_store_path(context.asset_key)
        repo, _ = self._get_repo(store_path)
        session = repo.readonly_session(branch="main")
        ds = xr.open_zarr(session.store, consolidated=False)

        if context.has_partition_key:
            if context.upstream_output is None:
                raise dg.DagsterInvariantViolationError(
                    "Input context has a partition key but no upstream output. "
                    "This is unexpected and likely indicates a misconfiguration."
                )
            append_dim: str = context.upstream_output.definition_metadata["schema"].append_dim()
            # Filter to the specific partition
            partition_key = context.partition_key
            ds = ds.sel({append_dim: partition_key})

        return ds
