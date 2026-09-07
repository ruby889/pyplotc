#!/usr/bin/env python3
"""Convert rt_control file telemetry (.bin or .txt) to CSV.

Binary layout is versioned via the CTLG header `version` field (bumped on the C
side when SaveData changes). Add a new entry to VERSION_SPECS when supporting a
new layout; keep older specs so historical .bin files remain readable.
"""

from __future__ import annotations

import argparse
import csv
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

MAGIC = b"CTLG"
CURRENT_VERSION = 1

# Capacity limits baked into the on-wire record for each supported layout.
MAX_ARM_SIZE = 2
MAX_ARM_JOINT_SIZE = 7
MAX_GRIPPER_SIZE = 2
MAX_GRIPPER_JOINT_SIZE = 6

HEADER_STRUCT = struct.Struct("<4sHH2B B 2B II")


def _record_struct_v1() -> struct.Struct:
    parts: List[str] = []
    parts.extend(["i"] * 7)
    parts.append("4x")  # align uint64_t after int fields
    parts.append("Q")
    parts.extend(["Q"] * (MAX_ARM_SIZE * MAX_ARM_JOINT_SIZE))
    parts.extend(["Q"] * (MAX_GRIPPER_SIZE * MAX_GRIPPER_JOINT_SIZE))
    parts.append("i")
    parts.append("4x")  # align long after int button_state
    parts.extend(["q"] * 7)
    parts.extend(["f"] * (MAX_ARM_SIZE * MAX_GRIPPER_JOINT_SIZE * 2))
    parts.extend(["f"] * (MAX_ARM_SIZE * MAX_ARM_JOINT_SIZE * 4))
    parts.extend(["d"] * (MAX_ARM_SIZE * MAX_ARM_JOINT_SIZE * 8))
    parts.extend(["d"] * (MAX_ARM_SIZE * 7))
    parts.extend(["f"] * (MAX_ARM_SIZE * MAX_ARM_JOINT_SIZE * 6))
    parts.extend(["f"] * (MAX_ARM_SIZE * 6 * 3))
    return struct.Struct("<" + "".join(parts))


RECORD_STRUCT_V1 = _record_struct_v1()
RECORD_STRUCT = RECORD_STRUCT_V1  # alias for current layout


@dataclass(frozen=True)
class TelemetryHeader:
    version: int
    arm_size: int
    arm_joint_size: Tuple[int, int]
    gripper_size: int
    gripper_joint_size: Tuple[int, int]
    record_size: int


@dataclass
class SaveData:
    simulation: int
    target_type: int
    actuator_mode: int
    control_strategy: int
    branch_state: int
    system_state: int
    plan_result: int
    system_diagnostic_flags: int
    arm_diagnostic_flags: List[List[int]]
    gripper_diagnostic_flags: List[List[int]]
    button_state: int
    communication_cost: int
    hardware_communication_cost: int
    running_cost: int
    robotics_total_cost: int
    motor_communication_cost: int
    gripper_communication_cost: int
    button_communication_cost: int
    gripper_target: List[List[float]]
    gripper_position: List[List[float]]
    joint_gravity: List[List[float]]
    filtered_joint_vel: List[List[float]]
    friction: List[List[float]]
    gravity_kd_effect: List[List[float]]
    motor_target: List[List[float]]
    joint_position: List[List[float]]
    joint_velocity: List[List[float]]
    joint_acceleration: List[List[float]]
    joint_torque: List[List[float]]
    motor_velocity: List[List[float]]
    motor_torque: List[List[float]]
    motor_position: List[List[float]]
    tool_pose: List[List[float]]
    target_before_interpolation: List[List[float]]
    sliced_tar: List[List[float]]
    task_target_before_interpolation: List[List[float]]
    task_sliced_tar: List[List[float]]
    task_pose: List[List[float]]
    pid_cmd: List[List[float]]
    guard_pos_cmd: List[List[float]]
    sat_cmd: List[List[float]]
    jump_cmd: List[List[float]]


@dataclass(frozen=True)
class TelemetryVersionSpec:
    """Per-version binary codec registered in VERSION_SPECS."""

    version: int
    record_struct: struct.Struct
    parse: Callable[[Sequence[object]], object]
    flatten: Callable[[object, TelemetryHeader], List[object]]
    columns: Callable[[Optional[TelemetryHeader]], List[str]]


def read_binary_header(data: bytes) -> TelemetryHeader:
    if len(data) < HEADER_STRUCT.size:
        raise ValueError("file too small for telemetry header")
    magic, version, arm_size, arm_joint0, arm_joint1, gripper_size, gri_joint0, gri_joint1, record_size, _reserved = (
        HEADER_STRUCT.unpack_from(data, 0)
    )
    if magic != MAGIC:
        raise ValueError(f"invalid telemetry magic: {magic!r}")
    return TelemetryHeader(
        version=version,
        arm_size=arm_size,
        arm_joint_size=(arm_joint0, arm_joint1),
        gripper_size=gripper_size,
        gripper_joint_size=(gri_joint0, gri_joint1),
        record_size=record_size,
    )


def _take_matrix(values: Sequence[object], idx: int, rows: int, cols: int) -> Tuple[List[List[object]], int]:
    matrix: List[List[object]] = []
    for _ in range(rows):
        matrix.append(list(values[idx : idx + cols]))
        idx += cols
    return matrix, idx


def parse_save_data_v1(values: Sequence[object]) -> SaveData:
    idx = 0
    ints = values[idx : idx + 7]
    idx += 7
    system_diagnostic_flags = values[idx]
    idx += 1
    arm_diagnostic_flags, idx = _take_matrix(values, idx, MAX_ARM_SIZE, MAX_ARM_JOINT_SIZE)
    gripper_diagnostic_flags, idx = _take_matrix(values, idx, MAX_GRIPPER_SIZE, MAX_GRIPPER_JOINT_SIZE)
    button_state = values[idx]
    idx += 1
    costs = values[idx : idx + 7]
    idx += 7
    gripper_target, idx = _take_matrix(values, idx, MAX_ARM_SIZE, MAX_GRIPPER_JOINT_SIZE)
    gripper_position, idx = _take_matrix(values, idx, MAX_ARM_SIZE, MAX_GRIPPER_JOINT_SIZE)
    joint_gravity, idx = _take_matrix(values, idx, MAX_ARM_SIZE, MAX_ARM_JOINT_SIZE)
    filtered_joint_vel, idx = _take_matrix(values, idx, MAX_ARM_SIZE, MAX_ARM_JOINT_SIZE)
    friction, idx = _take_matrix(values, idx, MAX_ARM_SIZE, MAX_ARM_JOINT_SIZE)
    gravity_kd_effect, idx = _take_matrix(values, idx, MAX_ARM_SIZE, MAX_ARM_JOINT_SIZE)
    motor_target, idx = _take_matrix(values, idx, MAX_ARM_SIZE, MAX_ARM_JOINT_SIZE)
    joint_position, idx = _take_matrix(values, idx, MAX_ARM_SIZE, MAX_ARM_JOINT_SIZE)
    joint_velocity, idx = _take_matrix(values, idx, MAX_ARM_SIZE, MAX_ARM_JOINT_SIZE)
    joint_acceleration, idx = _take_matrix(values, idx, MAX_ARM_SIZE, MAX_ARM_JOINT_SIZE)
    joint_torque, idx = _take_matrix(values, idx, MAX_ARM_SIZE, MAX_ARM_JOINT_SIZE)
    motor_velocity, idx = _take_matrix(values, idx, MAX_ARM_SIZE, MAX_ARM_JOINT_SIZE)
    motor_torque, idx = _take_matrix(values, idx, MAX_ARM_SIZE, MAX_ARM_JOINT_SIZE)
    motor_position, idx = _take_matrix(values, idx, MAX_ARM_SIZE, MAX_ARM_JOINT_SIZE)
    tool_pose, idx = _take_matrix(values, idx, MAX_ARM_SIZE, 7)
    target_before_interpolation, idx = _take_matrix(values, idx, MAX_ARM_SIZE, MAX_ARM_JOINT_SIZE)
    sliced_tar, idx = _take_matrix(values, idx, MAX_ARM_SIZE, MAX_ARM_JOINT_SIZE)
    task_target_before_interpolation, idx = _take_matrix(values, idx, MAX_ARM_SIZE, 6)
    task_sliced_tar, idx = _take_matrix(values, idx, MAX_ARM_SIZE, 6)
    task_pose, idx = _take_matrix(values, idx, MAX_ARM_SIZE, 6)
    pid_cmd, idx = _take_matrix(values, idx, MAX_ARM_SIZE, MAX_ARM_JOINT_SIZE)
    guard_pos_cmd, idx = _take_matrix(values, idx, MAX_ARM_SIZE, MAX_ARM_JOINT_SIZE)
    sat_cmd, idx = _take_matrix(values, idx, MAX_ARM_SIZE, MAX_ARM_JOINT_SIZE)
    jump_cmd, idx = _take_matrix(values, idx, MAX_ARM_SIZE, MAX_ARM_JOINT_SIZE)

    return SaveData(
        simulation=ints[0],
        target_type=ints[1],
        actuator_mode=ints[2],
        control_strategy=ints[3],
        branch_state=ints[4],
        system_state=ints[5],
        plan_result=ints[6],
        system_diagnostic_flags=system_diagnostic_flags,
        arm_diagnostic_flags=arm_diagnostic_flags,
        gripper_diagnostic_flags=gripper_diagnostic_flags,
        button_state=button_state,
        communication_cost=costs[0],
        hardware_communication_cost=costs[1],
        running_cost=costs[2],
        robotics_total_cost=costs[3],
        motor_communication_cost=costs[4],
        gripper_communication_cost=costs[5],
        button_communication_cost=costs[6],
        gripper_target=gripper_target,
        gripper_position=gripper_position,
        joint_gravity=joint_gravity,
        filtered_joint_vel=filtered_joint_vel,
        friction=friction,
        gravity_kd_effect=gravity_kd_effect,
        motor_target=motor_target,
        joint_position=joint_position,
        joint_velocity=joint_velocity,
        joint_acceleration=joint_acceleration,
        joint_torque=joint_torque,
        motor_velocity=motor_velocity,
        motor_torque=motor_torque,
        motor_position=motor_position,
        tool_pose=tool_pose,
        target_before_interpolation=target_before_interpolation,
        sliced_tar=sliced_tar,
        task_target_before_interpolation=task_target_before_interpolation,
        task_sliced_tar=task_sliced_tar,
        task_pose=task_pose,
        pid_cmd=pid_cmd,
        guard_pos_cmd=guard_pos_cmd,
        sat_cmd=sat_cmd,
        jump_cmd=jump_cmd,
    )


def column_names_v1(header: Optional[TelemetryHeader] = None) -> List[str]:
    names = [
        "running_cost",
        "robotics_total_cost",
        "hardware_communication_cost",
        "communication_cost",
        "motor_communication_cost",
        "gripper_communication_cost",
        "button_communication_cost",
        "actuator_mode",
        "control_strategy",
        "branch_state",
        "system_state",
        "plan_result",
        "system_diagnostic_flags",
        "button_state",
        "simulation",
        "target_type",
    ]

    arm_count = header.arm_size if header else MAX_ARM_SIZE
    gri_count = header.gripper_size if header else MAX_GRIPPER_SIZE

    for arm_i in range(arm_count):
        joint_count = header.arm_joint_size[arm_i] if header else MAX_ARM_JOINT_SIZE
        for j in range(joint_count):
            names.extend(
                [
                    f"arm{arm_i}_j{j}_motor_target",
                    f"arm{arm_i}_j{j}_joint_position",
                    f"arm{arm_i}_j{j}_joint_velocity",
                    f"arm{arm_i}_j{j}_joint_acceleration",
                    f"arm{arm_i}_j{j}_joint_torque",
                    f"arm{arm_i}_j{j}_motor_position",
                    f"arm{arm_i}_j{j}_motor_velocity",
                    f"arm{arm_i}_j{j}_motor_torque",
                ]
            )
        for j in range(7):
            names.append(f"arm{arm_i}_tool_pose{j}")
        for j in range(joint_count):
            names.extend(
                [
                    f"arm{arm_i}_j{j}_target_before_interpolation",
                    f"arm{arm_i}_j{j}_sliced_tar",
                    f"arm{arm_i}_j{j}_pid_cmd",
                    f"arm{arm_i}_j{j}_guard_pos_cmd",
                    f"arm{arm_i}_j{j}_sat_cmd",
                    f"arm{arm_i}_j{j}_jump_cmd",
                ]
            )
        for j in range(6):
            names.extend(
                [
                    f"arm{arm_i}_task_target_before_interpolation{j}",
                    f"arm{arm_i}_task_sliced_tar{j}",
                    f"arm{arm_i}_task_pose{j}",
                ]
            )
        for j in range(joint_count):
            names.extend(
                [
                    f"arm{arm_i}_j{j}_joint_gravity",
                    f"arm{arm_i}_j{j}_friction",
                    f"arm{arm_i}_j{j}_gravity_kd_effect",
                    f"arm{arm_i}_j{j}_filtered_joint_vel",
                    f"arm{arm_i}_j{j}_arm_diagnostic_flags",
                ]
            )

    for gri_i in range(gri_count):
        joint_count = header.gripper_joint_size[gri_i] if header else MAX_GRIPPER_JOINT_SIZE
        for j in range(joint_count):
            names.extend(
                [
                    f"gripper{gri_i}_j{j}_target",
                    f"gripper{gri_i}_j{j}_position",
                    f"gripper{gri_i}_j{j}_diagnostic_flags",
                ]
            )
    return names


def flatten_record_v1(record: object, header: TelemetryHeader) -> List[object]:
    assert isinstance(record, SaveData)
    row: List[object] = [
        record.running_cost,
        record.robotics_total_cost,
        record.hardware_communication_cost,
        record.communication_cost,
        record.motor_communication_cost,
        record.gripper_communication_cost,
        record.button_communication_cost,
        record.actuator_mode,
        record.control_strategy,
        record.branch_state,
        record.system_state,
        record.plan_result,
        record.system_diagnostic_flags,
        record.button_state,
        record.simulation,
        record.target_type,
    ]

    for arm_i in range(header.arm_size):
        joint_count = header.arm_joint_size[arm_i]
        for j in range(joint_count):
            row.extend(
                [
                    record.motor_target[arm_i][j],
                    record.joint_position[arm_i][j],
                    record.joint_velocity[arm_i][j],
                    record.joint_acceleration[arm_i][j],
                    record.joint_torque[arm_i][j],
                    record.motor_position[arm_i][j],
                    record.motor_velocity[arm_i][j],
                    record.motor_torque[arm_i][j],
                ]
            )
        for j in range(7):
            row.append(record.tool_pose[arm_i][j])
        for j in range(joint_count):
            row.extend(
                [
                    record.target_before_interpolation[arm_i][j],
                    record.sliced_tar[arm_i][j],
                    record.pid_cmd[arm_i][j],
                    record.guard_pos_cmd[arm_i][j],
                    record.sat_cmd[arm_i][j],
                    record.jump_cmd[arm_i][j],
                ]
            )
        for j in range(6):
            row.extend(
                [
                    record.task_target_before_interpolation[arm_i][j],
                    record.task_sliced_tar[arm_i][j],
                    record.task_pose[arm_i][j],
                ]
            )
        for j in range(joint_count):
            row.extend(
                [
                    record.joint_gravity[arm_i][j],
                    record.friction[arm_i][j],
                    record.gravity_kd_effect[arm_i][j],
                    record.filtered_joint_vel[arm_i][j],
                    record.arm_diagnostic_flags[arm_i][j],
                ]
            )

    for gri_i in range(header.gripper_size):
        joint_count = header.gripper_joint_size[gri_i]
        for j in range(joint_count):
            row.extend(
                [
                    record.gripper_target[gri_i][j],
                    record.gripper_position[gri_i][j],
                    record.gripper_diagnostic_flags[gri_i][j],
                ]
            )
    return row


# ---------------------------------------------------------------------------
# Version registry
# When C bumps TELEMETRY_VERSION / SaveData layout, add a new *_vN codec here
# and register it. Do not mutate older specs.
# ---------------------------------------------------------------------------
VERSION_SPECS: Dict[int, TelemetryVersionSpec] = {
    1: TelemetryVersionSpec(
        version=1,
        record_struct=RECORD_STRUCT_V1,
        parse=parse_save_data_v1,
        flatten=flatten_record_v1,
        columns=column_names_v1,
    ),
}


def supported_versions() -> List[int]:
    return sorted(VERSION_SPECS)


def get_version_spec(version: int) -> TelemetryVersionSpec:
    spec = VERSION_SPECS.get(version)
    if spec is None:
        raise ValueError(
            f"unsupported telemetry version {version}; "
            f"supported versions: {supported_versions()}"
        )
    return spec


def resolve_version_spec(header: TelemetryHeader) -> TelemetryVersionSpec:
    """Select codec by header.version and verify record_size."""
    spec = get_version_spec(header.version)
    if header.record_size != spec.record_struct.size:
        raise ValueError(
            f"record_size mismatch for version {header.version}: "
            f"file={header.record_size}, parser={spec.record_struct.size}"
        )
    return spec


# Backward-compatible aliases for the current layout.
parse_save_data = parse_save_data_v1
flatten_record = flatten_record_v1


def column_names(header: Optional[TelemetryHeader] = None) -> List[str]:
    """Return flat column names for a header (dispatches by version)."""
    if header is None:
        return get_version_spec(CURRENT_VERSION).columns(None)
    return resolve_version_spec(header).columns(header)


def iter_binary_records(path: Path, downsample: int = 1) -> Tuple[TelemetryHeader, Iterator[List[object]]]:
    data = path.read_bytes()
    header = read_binary_header(data)
    spec = resolve_version_spec(header)
    offset = HEADER_STRUCT.size

    def _records() -> Iterator[List[object]]:
        nonlocal offset
        record_index = 0
        while offset + header.record_size <= len(data):
            values = spec.record_struct.unpack_from(data, offset)
            offset += header.record_size
            if record_index % downsample == 0:
                yield spec.flatten(spec.parse(values), header)
            record_index += 1

    return header, _records()


def convert_text(path: Path, downsample: int = 1) -> Tuple[List[str], Iterator[List[str]]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        raise ValueError("empty telemetry text file")

    header_line = lines[0]
    if header_line.startswith("#"):
        columns = header_line.lstrip("#").strip().split()
        data_lines = lines[1:]
    else:
        columns = column_names()
        data_lines = lines

    def _rows() -> Iterator[List[str]]:
        for index, line in enumerate(data_lines):
            if not line.strip():
                continue
            if index % downsample != 0:
                continue
            yield line.split()

    return columns, _rows()


def write_csv(output_path: Path, columns: List[str], rows: Iterable[Sequence[object]]) -> int:
    count = 0
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(columns)
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert rt_control telemetry files to CSV.")
    parser.add_argument("input", type=Path, help="Input .bin or .txt telemetry file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output CSV path (default: input with .csv extension)",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=1,
        help="Keep every N-th record (default: 1)",
    )
    args = parser.parse_args(argv)

    if args.downsample < 1:
        parser.error("--downsample must be >= 1")

    input_path = args.input
    output_path = args.output or input_path.with_suffix(".csv")

    if input_path.suffix == ".bin":
        header, rows = iter_binary_records(input_path, downsample=args.downsample)
        count = write_csv(output_path, column_names(header), rows)
    elif input_path.suffix == ".txt":
        columns, rows = convert_text(input_path, downsample=args.downsample)
        count = write_csv(output_path, columns, rows)
    else:
        parser.error("input must be a .bin or .txt telemetry file")

    print(f"Wrote {count} rows to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
