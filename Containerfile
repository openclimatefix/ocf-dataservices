# syntax=docker/dockerfile:1

FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim AS builder

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

# Full workspace: uv needs every member's pyproject.toml present to resolve uv.lock, even
# though --no-editable below only ends up installing the root project's actual dependencies.
COPY pyproject.toml uv.lock README.md ./
COPY packages/ packages/
COPY src/ src/

# --no-editable installs every workspace package as a regular wheel into .venv, so .venv is
# fully self-contained and portable into the runtime stage with no source tree alongside it
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim AS runtime

ARG GIT_SHA

LABEL org.opencontainers.image.source="https://github.com/openclimatefix/ocf-dataservices" \
      org.opencontainers.image.revision="${GIT_SHA}"

ENV GIT_SHA="${GIT_SHA}" \
    PATH="/app/.venv/bin:${PATH}"

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY uv.lock ./

# Use standard exec form
CMD ["dagster", "api", "grpc", "-m", "ocf_dataservices.definitions", "-h", "0.0.0.0", "-p", "4000"]
