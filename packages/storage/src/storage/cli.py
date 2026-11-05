import argparse
import datetime as dt
import logging
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import xarray as xr
from schemas.base import NwpDatasetSchema

from .store import has_partition, write_to_icechunk

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProductSpec:
    """Configuration for a single runnable product in a `*-data` package."""

    schema: type[NwpDatasetSchema]
    fetch: Callable[[dt.datetime], xr.Dataset]
    description: str = ""


def _parse_utc_datetime(raw: str) -> dt.datetime:
    parsed = dt.datetime.fromisoformat(raw)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=dt.UTC)
    return parsed.astimezone(dt.UTC)


def _resolve_init_times(args: argparse.Namespace) -> list[dt.datetime]:
    if args.init_time:
        if args.start or args.end:
            raise ValueError("Specify either --init-time or --start/--end, not both.")
        return [_parse_utc_datetime(args.init_time)]

    if args.start and args.end:
        start = _parse_utc_datetime(args.start)
        end = _parse_utc_datetime(args.end)
        if end < start:
            raise ValueError(f"--end ({end.isoformat()}) must be >= --start ({start.isoformat()}).")
        if args.step_hours <= 0:
            raise ValueError("--step-hours must be positive.")
        step = dt.timedelta(hours=args.step_hours)
        times: list[dt.datetime] = []
        current = start
        while current <= end:
            times.append(current)
            current += step
        return times

    raise ValueError("Must provide --init-time (or INIT_TIME) or both --start and --end.")


def build_parser(package_name: str, products: Mapping[str, ProductSpec]) -> argparse.ArgumentParser:
    """Build the unified CLI argument parser for a `*-data` package."""
    product_names = list(products.keys())
    default_product = os.getenv("PRODUCT") or (
        product_names[0] if len(product_names) == 1 else None
    )

    parser = argparse.ArgumentParser(
        prog=package_name,
        description=f"Standalone downloader and Icechunk writer for {package_name}.",
    )
    parser.add_argument(
        "--product",
        choices=product_names,
        default=default_product,
        required=default_product is None,
        help="Product to fetch and write (or set PRODUCT env var).",
    )
    parser.add_argument(
        "--store-path",
        default=os.getenv("STORE_PATH"),
        required=not bool(os.getenv("STORE_PATH")),
        help="Target Icechunk store path or s3://bucket/prefix (or set STORE_PATH env var).",
    )
    parser.add_argument(
        "--init-time",
        default=os.getenv("INIT_TIME"),
        help="Single forecast init_time in ISO-8601 format, e.g. 2026-04-08T00:00 (or INIT_TIME).",
    )
    parser.add_argument(
        "--start",
        default=os.getenv("START_TIME"),
        help="Start init_time (inclusive) in ISO-8601 format for range runs.",
    )
    parser.add_argument(
        "--end",
        default=os.getenv("END_TIME"),
        help="End init_time (inclusive) in ISO-8601 format for range runs.",
    )
    parser.add_argument(
        "--step-hours",
        type=int,
        default=int(os.getenv("STEP_HOURS") or 6),
        help="Step interval in hours between --start and --end (default: 6).",
    )
    parser.add_argument(
        "--write-stats",
        action="store_true",
        help="Also compute and write stats_spatial and stats_temporal Zarr groups (default: off).",
    )
    return parser


def run_cli(
    package_name: str,
    products: Mapping[str, ProductSpec],
    argv: Sequence[str] | None = None,
) -> None:
    """Parse CLI args, check Icechunk, fetch/validate data, and write to Icechunk."""
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    parser = build_parser(package_name, products)
    args = parser.parse_args(argv)

    try:
        init_times = _resolve_init_times(args)
    except ValueError as exc:
        parser.error(str(exc))
        return

    spec = products[args.product]
    store_path: str = args.store_path
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID", "")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    aws_region_name = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION", "")
    aws_endpoint_url = os.getenv("AWS_ENDPOINT_URL")

    for init_time in init_times:
        iso_time = init_time.strftime("%Y-%m-%dT%H:%M")
        if has_partition(
            store_path=store_path,
            append_dim=spec.schema.append_dim(),
            partition_key=init_time,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_region_name=aws_region_name,
            aws_endpoint_url=aws_endpoint_url,
        ):
            logger.info(
                "Partition %s already exists in %s; skipping.",
                iso_time,
                store_path,
            )
            continue

        logger.info("Fetching %s (%s) for init_time=%s", package_name, args.product, iso_time)
        ds = spec.fetch(init_time)

        result = write_to_icechunk(
            ds=ds,
            schema=spec.schema,
            store_path=store_path,
            deduplicate=True,
            write_stats=args.write_stats,
            commit_message=f"{package_name}:{args.product} {iso_time}",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_region_name=aws_region_name,
            aws_endpoint_url=aws_endpoint_url,
        )
        logger.info(
            "Wrote %s (%s) init_time=%s to %s (commit=%s, bytes=%d)",
            package_name,
            args.product,
            iso_time,
            store_path,
            result.commit_hash,
            result.size_bytes,
        )
