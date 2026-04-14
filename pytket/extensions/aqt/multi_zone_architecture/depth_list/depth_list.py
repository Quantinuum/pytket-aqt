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
from copy import deepcopy
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from pytket.circuit import Circuit, Command, OpType, Qubit

DepthList: TypeAlias = list[list[tuple[int, int]]]
DepthBlocks: TypeAlias = list[list[set[int]]]
"""
DepthBlocks provides, at each depth, a list of blocks.
   Blocks are disjoint sets of qubits that are connected through gates
   up to the current depth.
   With increasing depth, the number of blocks can only decrease while their
   size can only increase.
"""


@dataclass
class DepthInfo:
    depth_list: DepthList
    depth_blocks: DepthBlocks


def get_2q_gate_pairs_from_circuit(circuit: Circuit) -> list[tuple[int, int]]:
    return get_2q_gate_pairs_from_commands(circuit.get_commands())


def get_2q_gate_pairs_from_commands(commands: list[Command]) -> list[tuple[int, int]]:
    pair_list: list[tuple[int, int]] = []
    for cmd in commands:
        if cmd.op.type == OpType.Barrier:
            continue
        n_args = len(cmd.args)
        if n_args == 1 or cmd.op.type == OpType.Measure:
            continue
        if (
            n_args == 2  # noqa: PLR2004
            and isinstance(cmd.args[0], Qubit)
            and isinstance(cmd.args[1], Qubit)
        ):
            qubit0 = cmd.args[0].index[0]
            qubit1 = cmd.args[1].index[0]
            # always smaller index first
            if qubit0 < qubit1:
                pair_list.append((qubit0, qubit1))
            else:
                pair_list.append((qubit1, qubit0))
    return pair_list


def get_depth_info(n_qubits: int, gate_pairs: list[tuple[int, int]]) -> DepthInfo:  # noqa: PLR0912
    depth_list = get_depth_list(n_qubits, gate_pairs)

    if not depth_list:
        return DepthInfo([], [])

    depth_blocks: DepthBlocks = []
    block_assignment = np.array([-1] * n_qubits, dtype=np.int64)
    unique_pairs_depth_0 = {tuple(sorted(pair)) for pair in depth_list[0]}
    depth_blocks.append([])
    for i, pair in enumerate(unique_pairs_depth_0):
        block_assignment[[pair[0], pair[1]]] = i
        depth_blocks[0].append(set(pair))

    for depth_layer in depth_list[1:]:
        current_blocks = deepcopy(depth_blocks[-1])

        def merge_blocks(
            block0_idx: int, block1_idx: int, current_blocks_loc: list[set[int]]
        ) -> None:
            """Merge two blocks into first and clear second"""
            block1 = current_blocks_loc[block1_idx]
            current_blocks_loc[block0_idx].update(block1)
            block_assignment[list(block1)] = block0_idx
            block1.clear()

        for pair in depth_layer:
            current_block_q0 = block_assignment[pair[0]]
            current_block_q1 = block_assignment[pair[1]]
            match (current_block_q0, current_block_q1):
                case (-1, -1):
                    raise Exception("Should have been at lower depth")
                case (-1, _):
                    current_blocks[current_block_q1].add(pair[0])
                    block_assignment[pair[0]] = current_block_q1
                case (_, -1):
                    current_blocks[current_block_q0].add(pair[1])
                    block_assignment[pair[1]] = current_block_q0
                case (a, b) if a != b:
                    merge_blocks(current_block_q0, current_block_q1, current_blocks)

        # remove empty blocks
        n_remove = 0
        remove_tags: list[int | None] = [None] * len(current_blocks)
        # tag blocks None -> Remove, int = position shift due to removal
        for j, block in enumerate(current_blocks):
            if len(block) == 0:
                n_remove += 1
            else:
                remove_tags[j] = n_remove
        # shift and remove blocks
        for j in range(len(current_blocks) - 1, -1, -1):
            tag = remove_tags[j]
            if tag == 0:
                # nothing left to do
                break
            if tag is None:
                current_blocks.pop(j)
            else:
                block_assignment[list(current_blocks[j])] = j - tag
        depth_blocks.append(current_blocks)
    return DepthInfo(depth_list, depth_blocks)


def get_depth_list(n_qubits: int, gate_pairs: list[tuple[int, int]]) -> DepthList:
    depth_list: DepthList = []
    current_depth_per_qubit: list[int] = [0] * n_qubits
    for pair in gate_pairs:
        qubit0 = pair[0]
        qubit1 = pair[1]
        depth = max(current_depth_per_qubit[qubit0], current_depth_per_qubit[qubit1])
        assert len(depth_list) >= depth
        if depth > 0 and (
            (qubit0, qubit1) in depth_list[depth - 1]
            or (
                qubit1,
                qubit0,
            )
            in depth_list[depth - 1]
        ):
            depth_list[depth - 1].append((qubit0, qubit1))
            continue
        if len(depth_list) > depth:
            depth_list[depth].append((qubit0, qubit1))
        else:
            depth_list.append([(qubit0, qubit1)])
        current_depth_per_qubit[qubit0] = depth + 1
        current_depth_per_qubit[qubit1] = depth + 1
    return depth_list


def get_initial_depth_info(circuit: Circuit) -> DepthInfo:
    """From a given Circuit get the Depth list used to determine gate priority.

    Used for the initial placement of ions based on partitioning algorithms
    """
    n_qubits = circuit.n_qubits
    gate_pairs = get_2q_gate_pairs_from_circuit(circuit)
    return get_depth_info(n_qubits, gate_pairs)


def depth_info_from_command_list(n_qubits: int, commands: list[Command]) -> DepthInfo:
    """From a given list of Circuit Commands get the Depth list used to determine gate priority.

    Used for the initial placement of ions based on partitioning algorithms
    """
    gate_pairs = get_2q_gate_pairs_from_commands(commands)
    return get_depth_info(n_qubits, gate_pairs)
