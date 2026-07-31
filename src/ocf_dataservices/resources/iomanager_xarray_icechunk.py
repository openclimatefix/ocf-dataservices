import datetime as dt
import re

import dagster as dg
import icechunk
import numpy as np
import xarray as xr
import zarr.codecs


class XarrayIcechunkIOManager(dg.ConfigurableIOManager):
    path: str = ""
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region_name: str = ""
    aws_endpoint_url: str | None = None

    def _get_store_path(self, asset_key: dg.AssetKey) -> str:
        return "/".join([self.path] + list(asset_key.path))

    def _get_repo(self, store_path: str, create_if_not_exists: bool = False) -> tuple[icechunk.Repository, bool]:
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
            except Exception as e:
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
            repo = self._get_repo(store_path)
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
        except Exception:
            # If the repo, store, or dimension doesn't exist yet, it's not materialized
            return None

    def handle_output(self, context: dg.OutputContext, obj: object) -> None:
        if not isinstance(obj, xr.Dataset):
            raise dg.DagsterInvariantViolationError(
                f"XarrayIcechunkIOManager can only handle xr.Dataset objects, got {type(obj)}"
            )

        metadata = context.definition_metadata or {}
        for key in ["chunks", "shards"]:
            if list(obj.dims) != list(metadata[key].keys()):
                raise dg.DagsterInvariantViolationError(
                    f"Dataset dimensions {list(obj.dims)} do not match supplied {key} keys " + \
                    f"{list(metadata[key].keys())}"
                )

        store_path = self._get_store_path(context.asset_key)
        repo, was_created = self._get_repo(store_path, create_if_not_exists=True)
        session = repo.writable_session(branch="main")

        if context.has_partition_key and not was_created:
            obj.to_zarr(
                session.store,
                mode="a",
                append_dim=metadata["append_dim"],
                write_empty_chunks=False,
            )
        else:
            obj.to_zarr(
                session.store,
                zarr_format=3,
                mode="w-",
                write_empty_chunks=False,
                encoding={var: {
                    "dtype": "float32",
                    "chunks": tuple([
                        obj.coords[k].size if v == -1 else v
                        for k, v in metadata["chunks"].items()
                    ]),
                    "shards": tuple([
                        obj.coords[k].size if v == -1 else v
                        for k, v in metadata["shards"].items()
                    ]),
                    "compressors": zarr.codecs.BloscCodec(
                        cname="zstd",
                        clevel=3,
                        shuffle=zarr.codecs.BloscShuffle.bitshuffle,
                    ),
                } for var in obj.data_vars} | {
                    coord: {"chunks": 10000}
                    for coord in obj.coords if coord not in obj.dims
                } | {
                    metadata["append_dim"]: {
                        "dtype": int,
                        "units": "nanoseconds since 1970-01-01",
                        "calendar": "proleptic_gregorian",
                        "chunks": 10000,
                    }
                }
            )

        session.commit(f"dagster materialization: {context.asset_key} at {dt.datetime.now(dt.UTC)}")

        context.add_output_metadata({
            "store_path": store_path,
            "partition_key": context.partition_key if context.has_partition_key else "N/A",
            "data_vars": list(obj.data_vars),
            "dims": dict(obj.dims),
        })

    def load_input(self, context: dg.InputContext) -> xr.Dataset:
        store_path = self._get_store_path(context.asset_key)
        repo = self._get_repo(store_path)
        session = repo.readonly_session(branch="main")
        ds = xr.open_zarr(session.store, consolidated=False)

        if context.has_partition_key:
            if context.upstream_output is None:
                raise dg.DagsterInvariantViolationError(
                    "Input context has a partition key but no upstream output. "
                    "This is unexpected and likely indicates a misconfiguration."
                )
            append_dim: str = context.upstream_output.definition_metadata["append_dim"]
            # Filter to the specific partition
            partition_key = context.partition_key
            ds = ds.sel({append_dim: partition_key})

        return ds

