import datetime as dt
import fcntl
import re
from contextlib import contextmanager
from pathlib import Path

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
                append_dim = schema.append_dim()
                
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
    
            if not skip_write:
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
        }
        
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
