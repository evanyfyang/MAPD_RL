#!/usr/bin/env python3
"""
Generate KIVAONLINE task files for multiple frequencies.

The generated task format follows LNS-wPBS `KivaSystemOnline::load_tasks`:
    line 1: "<task_num> <task_frequency> <task_release_period>"
    each next line: "<release_time> <pickup_endpoint_idx> <delivery_endpoint_idx> 0 0"
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path
from typing import Iterable


FREQUENCIES = [10, 20, 30, 40, 50]


def read_map_meta(map_path: Path) -> tuple[int, int]:
    """Read endpoint count and task_num guess from map file/name."""
    lines = map_path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise ValueError(f"Invalid map file: {map_path}")

    endpoint_num = int(lines[1].strip())

    # Filename style: kiva-100-1000-50.map -> use the middle number as task_num.
    task_num = 1000
    match = re.match(r"^kiva-\d+-(\d+)-\d+$", map_path.stem)
    if match:
        task_num = int(match.group(1))

    return endpoint_num, task_num


def build_output_task_name(map_path: Path, frequency: int) -> str:
    """
    Build output file name by replacing the final '-<freq>' part.
    Example: kiva-100-1000-50.map -> kiva-100-1000-20.task
    """
    stem_parts = map_path.stem.split("-")
    if len(stem_parts) >= 2 and stem_parts[-1].isdigit():
        stem_parts[-1] = str(frequency)
        return "-".join(stem_parts) + ".task"
    return f"{map_path.stem}-f{frequency}.task"


def generate_tasks(
    endpoint_num: int,
    task_num: int,
    frequency: int,
    rng: random.Random,
    release_period: int = 1,
) -> list[str]:
    """Generate task file lines."""
    lines = [f"{task_num} {frequency} {release_period}"]
    for i in range(task_num):
        release_time = i // frequency
        pickup = rng.randrange(endpoint_num)
        delivery = rng.randrange(endpoint_num)
        while delivery == pickup:
            delivery = rng.randrange(endpoint_num)
        lines.append(f"{release_time} {pickup} {delivery} 0 0")
    return lines


def generate_for_map(
    map_path: Path,
    output_dir: Path,
    frequencies: Iterable[int],
    seed: int,
    dry_run: bool,
) -> None:
    endpoint_num, task_num = read_map_meta(map_path)

    for frequency in frequencies:
        rng = random.Random(f"{seed}:{map_path.name}:{frequency}")
        out_name = build_output_task_name(map_path, frequency)
        out_path = output_dir / out_name
        lines = generate_tasks(endpoint_num, task_num, frequency, rng)
        if dry_run:
            print(
                f"[DRY-RUN] {out_path} | endpoints={endpoint_num}, "
                f"tasks={task_num}, freq={frequency}"
            )
            continue
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(
            f"[OK] {out_path} | endpoints={endpoint_num}, "
            f"tasks={task_num}, freq={frequency}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate kiva task files for frequencies 10/20/30/40/50."
    )
    parser.add_argument(
        "--maps",
        nargs="+",
        default=[
            "/local-scratchg/yifan/2024/MAPD/LNS-wPBS/maps/Instances/large/kiva-100-1000-50.map",
            "/local-scratchg/yifan/2024/MAPD/LNS-wPBS/maps/Instances/large/kiva-200-1000-50.map",
        ],
        help="Input map file paths.",
    )
    parser.add_argument(
        "--output-dir",
        default="/local-scratchg/yifan/2024/MAPD/LNS-wPBS/maps/Instances/large",
        help="Directory to save generated .task files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed for reproducible generation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print target outputs without writing files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    maps = [Path(p) for p in args.maps]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for map_path in maps:
        if not map_path.exists():
            raise FileNotFoundError(f"Map file does not exist: {map_path}")
        generate_for_map(
            map_path=map_path,
            output_dir=output_dir,
            frequencies=FREQUENCIES,
            seed=args.seed,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
