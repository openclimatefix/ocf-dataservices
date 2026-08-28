# schemas

Shared [pandera](https://pandera.readthedocs.io/) field builders and a common base model
(`NwpDatasetSchema`) for validating NWP `xr.Dataset` objects produced by the provider packages
(`dynamical-data`, `ecmwfmars-data`, `ecmwflive-data`).

## What a schema validates

- **Shape**: the declared dimensions and coordinates are present, with the declared dtype.
- **Nullability**: which variables are allowed to contain nulls, and how many.
- **Physical unit bounds**: e.g. a temperature field's plausible range, a percentage field's 0-100
  range, `step` within a forecast horizon.

`latitude`/`longitude` fields are checked against global bounds (-90/90, -180/180). This is a basic
sanity check on the coordinate values themselves, not a check that the data covers any particular
region.

## What a schema does not validate

**Geographic region.** Each asset covers a specific bounding box (`bbox_nwse`), but that box is not
encoded in the schema. Region selection is a provider-package data-selection concern, applied by
slicing/filtering the data *before* schema validation runs — see e.g. `open_it` in
`dynamical_data.ecmwf_ens.download`, or the `bbox_nwse` slicing in `ecmwflive_data.processing`.
Passing schema validation confirms the data is physically plausible and correctly shaped; it does
**not** confirm the data covers the region a particular asset expects. If you need to guarantee
region coverage, add an assertion in the provider package's slicing logic, not in the schema.

## Validation contract

Every provider package's public entrypoint function (e.g. `download`, `convert_to_dataset`,
`process_ecmwf_live_uk_india`) validates its own return value — decorate it with
`@schemas.validates(YourSchema)` rather than validating at the Dagster asset call site. This
guarantees any caller of that function receives already-validated data, instead of relying on every
call site remembering to validate.

## Shared helpers

- `schemas.base.NwpDatasetSchema` — base `pa.DatasetModel` with common `Config`, plus
  `dims()`/`append_dim()` classmethods so dimension order and the Icechunk append dimension are
  declared once, on the schema, rather than duplicated at each call site.
- `schemas.dim_order.enforce_dim_order` — rebuilds a dataset with an explicit dimension order, for
  use before returning data ready to validate/write.
- `schemas.validation.validates` — decorator that validates a function's returned dataset against a
  schema before returning it.
