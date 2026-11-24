import os

from dagster._config import Field, StringSource
from dagster._core.secrets.env_file import get_env_var_dict
from dagster._core.secrets.loader import SecretsLoader
from dagster._serdes import ConfigurableClass
from dagster._serdes.config_class import ConfigurableClassData


class DotEnvSecretsLoader(SecretsLoader, ConfigurableClass):
    """Load environment variables from `{base_dir}/.env` into launched Dagster run containers.

    Replaces Dagster's built-in `EnvFileLoader`, which appends a duplicate `.env` path segment onto
    `base_dir` (`{base_dir}/.env/.env`) and returns an empty dictionary.
    """

    def __init__(self, inst_data: ConfigurableClassData | None = None, base_dir: str | None = None):
        self._inst_data = inst_data
        self._base_dir = base_dir or os.getcwd()

    def get_secrets_for_environment(self, location_name: str | None) -> dict[str, str]:
        return get_env_var_dict(self._base_dir)

    @property
    def inst_data(self):
        return self._inst_data

    @classmethod
    def config_type(cls):
        return {"base_dir": Field(StringSource, is_required=False)}

    @classmethod
    def from_config_value(cls, inst_data: ConfigurableClassData, config_value):
        return cls(inst_data=inst_data, **config_value)
