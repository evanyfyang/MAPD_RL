#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import random
from typing import List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按给定map与task_frequency生成长任务序列(.task)"
    )
    parser.add_argument(
        "--map_path",
        type=str,
        required=True,
        help="包含agent起始位信息的map文件路径",
    )
    parser.add_argument(
        "--num_tasks",
        type=int,
        default=6000,
        help="每个frequency生成的任务数量，默认6000",
    )
    parser.add_argument(
        "--frequencies",
        type=str,
        default="2,5,10",
        help='任务频率列表，逗号分隔，例如 "2,5,10"',
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="随机种子基数（不同frequency会在此基础上偏移）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出目录，默认与map同目录",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="输出文件名前缀，默认取map文件名(不含扩展名)",
    )
    parser.add_argument(
        "--use_all_endpoints",
        action="store_true",
        help="若传入则使用全部num_e端点；默认按env_gnn风格使用(num_e - num_agents)",
    )
    return parser.parse_args()


def read_map_meta(map_path: str) -> Tuple[int, int]:
    with open(map_path, "r", encoding="utf-8") as f:
        _grid_size = f.readline().strip()  # e.g. "21,35"
        num_e_line = f.readline().strip()
        num_r_line = f.readline().strip()
    num_e = int(num_e_line)
    num_r = int(num_r_line)
    return num_e, num_r


def parse_frequency_list(raw: str) -> List[float]:
    items = [x.strip() for x in raw.split(",") if x.strip()]
    if not items:
        raise ValueError("frequencies不能为空")
    freqs: List[float] = []
    for x in items:
        v = float(x)
        if v <= 0:
            raise ValueError(f"frequency必须>0，收到: {v}")
        freqs.append(v)
    return freqs


def release_time_for_task(task_idx: int, task_frequency: float, task_release_period: int) -> int:
    if task_release_period > 1:
        return task_release_period * task_idx
    return int(task_idx / task_frequency)


def build_tasks(
    rng: random.Random, num_tasks: int, endpoint_upper: int, task_frequency: float
) -> Tuple[int, List[Tuple[int, int, int]]]:
    # 对于你当前频率(2/5/10)，release_period=1
    if task_frequency < 1:
        task_release_period = int(1 / task_frequency)
    else:
        task_release_period = 1

    tasks: List[Tuple[int, int, int]] = []
    for i in range(num_tasks):
        pickup = rng.randint(0, endpoint_upper - 1)
        delivery = rng.randint(0, endpoint_upper - 1)
        while delivery == pickup:
            delivery = rng.randint(0, endpoint_upper - 1)
        release_time = release_time_for_task(i, task_frequency, task_release_period)
        tasks.append((release_time, pickup, delivery))
    return task_release_period, tasks


def write_task_file(
    out_path: str, num_tasks: int, task_frequency: float, task_release_period: int, tasks: List[Tuple[int, int, int]]
) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        # 与现有.task格式保持一致
        f.write(f"{num_tasks} {task_frequency:g} {task_release_period}\n")
        for release_time, pickup, delivery in tasks:
            f.write(f"{release_time}\t{pickup}\t{delivery}\t0\t0\n")


def main() -> None:
    args = parse_args()

    num_e, num_r = read_map_meta(args.map_path)
    frequencies = parse_frequency_list(args.frequencies)

    output_dir = args.output_dir if args.output_dir else os.path.dirname(os.path.abspath(args.map_path))
    os.makedirs(output_dir, exist_ok=True)

    base_name = args.prefix if args.prefix else os.path.splitext(os.path.basename(args.map_path))[0]

    if args.use_all_endpoints:
        endpoint_upper = num_e
    else:
        # 按env_gnn思路：endpoint_num = len(e_map) - agent_num
        endpoint_upper = num_e - num_r

    if endpoint_upper <= 1:
        raise ValueError(
            f"可用端点数量过小: {endpoint_upper} (num_e={num_e}, num_r={num_r}, use_all={args.use_all_endpoints})"
        )

    print(f"map: {args.map_path}")
    print(f"num_e={num_e}, num_r={num_r}, endpoint_upper={endpoint_upper}")
    print(f"frequencies={frequencies}, num_tasks={args.num_tasks}")
    print(f"output_dir={output_dir}")

    for idx, freq in enumerate(frequencies):
        rng = random.Random(args.seed + idx * 100003)
        release_period, tasks = build_tasks(
            rng=rng,
            num_tasks=args.num_tasks,
            endpoint_upper=endpoint_upper,
            task_frequency=freq,
        )
        out_name = f"{base_name}-n{args.num_tasks}-f{freq:g}.task"
        out_path = os.path.join(output_dir, out_name)
        write_task_file(
            out_path=out_path,
            num_tasks=args.num_tasks,
            task_frequency=freq,
            task_release_period=release_period,
            tasks=tasks,
        )
        print(f"[OK] {out_path}")


if __name__ == "__main__":
    main()

