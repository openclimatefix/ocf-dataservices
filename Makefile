export DAGSTER_HOME=${HOME}/.local/share/dagster

define DAGSTER_YAML
storage:
  sqlite:
    base_dir: "dagster_history"

run_coordinator:
  module: dagster.core.run_coordinator
  class: QueuedRunCoordinator

concurrency:
  pools:
    default_limit: 1  # Used to limit concurrency of the ecmwf_ens asset

run_monitoring:
  # Without this, a crashed/killed run can leak its concurrency-pool slot (e.g. the pool
  # above) forever, since nothing else frees a slot held by a run that never reached a
  # normal finally-block exit. This lets the daemon self-heal: any run finished (in any
  # terminal status) for longer than the threshold has its slots freed automatically.
  enabled: true
  free_slots_after_run_end_seconds: 300

python_logs:
  managed_python_loggers:
    - nged_data
  python_log_level: DEBUG

endef
export DAGSTER_YAML

init:
	@mkdir -p ${DAGSTER_HOME}
	@echo "$$DAGSTER_YAML" > ${DAGSTER_HOME}/dagster.yaml

.PHONY: run
run: init
	@uv run dg dev

.PHONY: lint
lint:
	@uv run ruff check --fix .

.PHONY: test
test:
	@uv run python -m unittest `find packages -name "test_*.py"`

.PHONY: clean
clean:
	@echo "Note: Saved data is not cleaned, this will have to be done manually."
	@rm -rf ${DAGSTER_HOME}
