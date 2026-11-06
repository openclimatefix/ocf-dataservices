import os

from dagster._config import Field, StringSource
from dagster._core.secrets.env_file import get_env_var_dict
from dagster._core.secrets.loader import SecretsLoader
from dagster._serdes import ConfigurableClass
from dagster._serdes.config_class import ConfigurableClassData


class DotEnvSecretsLoader(SecretsLoader, ConfigurableClass):
    """Loads env vars from ``{base_dir}/.env`` into launched run containers.

    Workaround for a bug in dagster's built-in ``EnvFileLoader`` (as of 1.13.17), which
    double-joins the ``.env`` suffix onto ``base_dir`` and ends up looking for
    ``{base_dir}/.env/.env``, silently returning no secrets.
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
