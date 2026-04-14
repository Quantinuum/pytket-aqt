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
import importlib.util

MT_KAHYPAR_INSTALLED = importlib.util.find_spec("mtkahypar") is not None
"""Whether or not the mtkahypar package is available"""


class MissingMtKahyparInstallError(Exception):
    def __init__(self) -> None:
        super().__init__("Graph partitioning requires mtkahypar package")
