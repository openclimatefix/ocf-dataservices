import base64
import datetime as dt
import html
import warnings

import dagster as dg
import icechunk
import numpy as np
import xarray as xr
from storage.store import (
    SPATIAL_DIMS,
    open_icechunk_repo,
    write_to_icechunk,
)
from storage.store import (
    existing_partition as store_existing_partition,
)


def _render_var_svg(
    var: str,
    grid_2d: np.ndarray | None,
    vmin: float,
    vmax: float,
) -> str:
    w, h = 240, 140
    rects: list[str] = []
    if grid_2d is not None and grid_2d.ndim == 2 and grid_2d.size > 0:
        sy = max(1, grid_2d.shape[0] // 20)
        sx = max(1, grid_2d.shape[1] // 20)
        small = grid_2d[::sy, ::sx]
        ny, nx = small.shape
        gx0, gy0, gw, gh = 16.0, 24.0, 170.0, 92.0
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

    title = html.escape(var)
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" style="font-family:sans-serif;font-size:10px;background:#fff">'
        f'<text x="16" y="14" font-weight="bold" font-size="11" fill="#111">{title}</text>'
        f'<text x="192" y="34" fill="#475569">{vmax:.3g}</text>'
        f'<text x="192" y="114" fill="#475569">{vmin:.3g}</text>'
        + "".join(rects)
        + "</svg>"
    )
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")


def _build_stats_metadata(
    stats: xr.Dataset | None,
) -> dict[str, dg.MetadataValue]:
    if stats is None:
        return {}

    records: list[dg.TableRecord] = []
    md_parts: list[str] = ["### Dataset Statistics Overview\n"]

    for var in stats.data_vars:
        var_name = str(var)
        da = stats[var_name]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_val = float(da.sel(stat="mean").mean(skipna=True).values)
            std_val = float(da.sel(stat="std").mean(skipna=True).values)
            min_val = float(da.sel(stat="min").min(skipna=True).values)
            max_val = float(da.sel(stat="max").max(skipna=True).values)

        records.append(
            dg.TableRecord(
                {
                    "variable": var_name,
                    "mean": round(mean_val, 4) if np.isfinite(mean_val) else 0.0,
                    "std": round(std_val, 4) if np.isfinite(std_val) else 0.0,
                    "min": round(min_val, 4) if np.isfinite(min_val) else 0.0,
                    "max": round(max_val, 4) if np.isfinite(max_val) else 0.0,
                }
            )
        )

        grid_2d = None
        t_da = da.sel(stat="mean")
        t_extra = [d for d in t_da.dims if d not in SPATIAL_DIMS]
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

        uri = _render_var_svg(
            var_name,
            grid_2d,
            min_val if np.isfinite(min_val) else 0.0,
            max_val if np.isfinite(max_val) else 1.0,
        )
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
        return open_icechunk_repo(
            store_path=store_path,
            create_if_not_exists=create_if_not_exists,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            aws_region_name=self.aws_region_name,
            aws_endpoint_url=self.aws_endpoint_url,
        )

    def existing_partition(
        self,
        asset_key: dg.AssetKey,
        append_dim: str,
        partition_key: str,
    ) -> xr.Dataset | None:
        """Gets a partition if it already exists, otherwise returns None."""
        store_path = self._get_store_path(asset_key)
        return store_existing_partition(
            store_path=store_path,
            append_dim=append_dim,
            partition_key=partition_key,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            aws_region_name=self.aws_region_name,
            aws_endpoint_url=self.aws_endpoint_url,
        )

    def handle_output(self, context: dg.OutputContext, obj: object) -> None:
        if not isinstance(obj, xr.Dataset):
            raise dg.DagsterInvariantViolationError(
                f"XarrayIcechunkIOManager can only handle xr.Dataset objects, got {type(obj)}"
            )

        metadata = context.definition_metadata or {}
        schema = metadata["schema"]
        keep_bits = metadata.get("keep_mantissa_bits", self.keep_mantissa_bits)
        store_path = self._get_store_path(context.asset_key)

        try:
            result = write_to_icechunk(
                ds=obj,
                schema=schema,
                store_path=store_path,
                deduplicate=context.has_partition_key,
                keep_mantissa_bits=keep_bits,
                write_stats=True,
                commit_message=(
                    f"dagster materialization: {context.asset_key} at {dt.datetime.now(dt.UTC)}"
                ),
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                aws_region_name=self.aws_region_name,
                aws_endpoint_url=self.aws_endpoint_url,
            )
        except ValueError as exc:
            raise dg.DagsterInvariantViolationError(str(exc)) from exc

        output_metadata = {
            "store_path": dg.MetadataValue.path(store_path),
            "partition_key": context.partition_key if context.has_partition_key else "N/A",
            "data_vars": list(obj.data_vars),
            "dims": result.written_dims,
            "icechunk_commit": dg.MetadataValue.text(str(result.commit_hash)),
            "size_in_memory_bytes": dg.MetadataValue.int(result.size_bytes),
            "size_human_readable": dg.MetadataValue.text(
                f"{result.size_bytes / (1024 * 1024):.2f} MB"
            ),
        } | _build_stats_metadata(result.stats)

        if result.skip_write:
            output_metadata["duplicate_partition_skipped"] = dg.MetadataValue.bool(True)

        context.add_output_metadata(output_metadata)

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
            partition_key = context.partition_key
            ds = ds.sel({append_dim: partition_key})

        return ds
