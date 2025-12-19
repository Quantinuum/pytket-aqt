# Copyright Quantinuum
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""AQT config."""

import logging
from dataclasses import dataclass
from getpass import getpass
from typing import Any, ClassVar

from pytket.config import PytketExtConfig
from qiskit_aqt_provider.aqt_provider import AQTProvider

logger = logging.getLogger()


@dataclass
class AQTConfig(PytketExtConfig):
    """Holds config parameters for pytket-aqt."""

    ext_dict_key: ClassVar[str] = "aqt"

    access_token: str | None

    @classmethod
    def from_extension_dict(
        cls: type["AQTConfig"], ext_dict: dict[str, Any]
    ) -> "AQTConfig":
        return cls(ext_dict.get("access_token"))


def set_aqt_config(
    access_token: str | None = None,
) -> None:
    """Set default value for AQT API token."""
    config = AQTConfig.from_default_config_file()
    if access_token is not None:
        config.access_token = access_token
    config.update_default_config_file()


class AQTAccessToken:
    """Holds the aqt access token in memory."""

    _access_token: str | None = None

    @classmethod
    def overwrite(cls, access_token: str) -> None:
        cls._access_token = access_token

    @classmethod
    def retrieve(cls) -> str | None:
        return cls._access_token

    @classmethod
    def reset(cls, access_token: str | None = None) -> None:
        if access_token:
            cls._access_token = access_token
            return
        cls._access_token = getpass(
            prompt="Enter your AQT access token (or press enter for offline use): "
        )
        if cls._access_token == "":
            warning = (
                "No AQT access token provided,"
                " only offline simulators will be available. "
                "Use pytket.extensions.aqt.AQTAccessToken.reset()"
                " to reset token."
            )
            logger.warning(warning)
            cls._access_token = "unspecified"

    @classmethod
    def resolve(cls, access_token: str | None = None) -> str:
        """Resolve access token by looking in several locations

        If `access_token` is set, use it and store in memory for later.
        If not, look first in memory
        and then in the pytket config file for an access token.
        If none available prompt user to
        input access token.
        """
        config = AQTConfig.from_default_config_file()
        access_token_config_file = config.access_token
        access_token_in_mem = AQTAccessToken.retrieve()
        match (access_token, access_token_in_mem, access_token_config_file):
            case (None, None, None):
                # Request access token from user and store in memory
                AQTAccessToken.reset()
            case (a, _, _) if a is not None:
                # Store provided access token in memory (possibly overwrite)
                AQTAccessToken.overwrite(a)
            case (None, a, _) if a is not None:
                # Use memory stored access token (see return)
                pass
            case (None, None, a) if a is not None:
                # Use access token from config file and store in memory
                AQTAccessToken.overwrite(a)
        resolved_token = AQTAccessToken.retrieve()
        assert resolved_token is not None
        return resolved_token


def print_available_devices(access_token: str | None = None) -> None:
    resolved_access_token = AQTAccessToken.resolve(access_token)

    aqt_provider = AQTProvider(access_token=resolved_access_token)
    backends = aqt_provider.backends()
    backends.headers[1] = "Device ID"
    backends.headers[3] = "Device type"
    print(backends)
