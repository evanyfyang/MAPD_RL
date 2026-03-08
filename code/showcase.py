import argparse
import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


def load_map(map_path: str) -> Tuple[np.ndarray, Tuple[int, int]]:
    with open(map_path, "r", encoding="utf-8") as f:
        grid_size = tuple(map(int, f.readline().strip().split(",")))
        height, width = grid_size
        lines = []
        for line in f:
            s = line.rstrip("\n")
            if len(s) == width:
                lines.append(s)

    grid = np.zeros((height, width), dtype=np.float32)
    for x, line in enumerate(lines):
        for y, ch in enumerate(line):
            if ch == "@":
                grid[x, y] = -1
    return grid, grid_size


def _task_map(status: Dict) -> Dict[int, Dict]:
    out = {}
    for t in status.get("tasks", []):
        out[int(t["task_id"])] = t
    for t in status.get("delivering_tasks", []):
        out[int(t["task_id"])] = t
    return out


def _agent_indices(status: Dict) -> Tuple[List[int], List[int]]:
    free = [int(a["idx"]) for a in status.get("agents_free", [])]
    delivering = [int(a["idx"]) for a in status.get("agents_delivering", [])]
    return free, delivering


def _paths_by_agent(status: Dict) -> Dict[int, List[List[int]]]:
    out = {}
    for p in status.get("paths", []):
        idx = int(p["idx"])
        out[idx] = [pt["loc"] for pt in p.get("path", [])]
    return out


def _assigned_task_ids(status: Dict) -> Dict[int, int]:
    assigned = {}
    seqs = status.get("agent_task_sequences", [])
    free_agents = {int(a["idx"]) for a in status.get("agents_free", [])}
    for agent_idx in free_agents:
        if agent_idx < len(seqs) and len(seqs[agent_idx]) > 0:
            assigned[agent_idx] = int(seqs[agent_idx][0])
    return assigned


def _truncate_path(path: List[List[int]], pickup, delivery) -> List[List[int]]:
    if not path:
        return path
    pickup_idx = None
    delivery_idx = None
    if pickup is not None:
        for i, loc in enumerate(path):
            if loc == pickup:
                pickup_idx = i
                break
    if delivery is not None:
        start = pickup_idx if pickup_idx is not None else 0
        for i in range(start, len(path)):
            if path[i] == delivery:
                delivery_idx = i
                break
    if delivery_idx is not None:
        return path[:delivery_idx + 1]
    if pickup_idx is not None:
        return path[:pickup_idx + 1]
    return []


def _plot_case(ax, grid: np.ndarray, status: Dict, title: str, skip_agents=None) -> None:
    obstacle = (grid == -1).astype(np.float32)
    ax.imshow(obstacle, cmap="gray_r", origin="upper", interpolation="none")
    ax.set_title(title)
    ax.set_aspect("equal")
    height, width = grid.shape
    ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
    ax.grid(which="minor", color="#d0d0d0", linewidth=0.5)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)

    task_map = _task_map(status)
    paths = _paths_by_agent(status)
    free_agents, delivering_agents = _agent_indices(status)
    assigned = _assigned_task_ids(status)
    delivering_map = {}
    for k, v in status.get("agent_task_pair", {}).items():
        try:
            delivering_map[int(k)] = int(v[0])
        except Exception:
            continue

    assigned_task_ids = set(assigned.values())
    task_points = []
    for tid in assigned_task_ids:
        task = task_map.get(tid)
        if not task:
            continue
        pickup = task.get("pickup")
        delivery = task.get("delivery")
        if pickup is not None:
            task_points.append(("pickup", pickup))
        if delivery is not None:
            task_points.append(("delivery", delivery))

    if skip_agents is None:
        skip_agents = set()
    agent_colors = list(mcolors.TABLEAU_COLORS.values())
    for agent_idx in delivering_agents + free_agents:
        if agent_idx in skip_agents:
            continue
        color = agent_colors[agent_idx % len(agent_colors)]
        full_path = paths.get(agent_idx, [])
        task_id = delivering_map.get(agent_idx) if agent_idx in delivering_agents else assigned.get(agent_idx)
        task = task_map.get(task_id) if task_id is not None else None
        pickup = task.get("pickup") if task else None
        delivery = task.get("delivery") if task else None
        path = _truncate_path(full_path, pickup, delivery)
        if len(path) >= 2:
            xs = [p[1] for p in path]
            ys = [p[0] for p in path]
            ax.plot(xs, ys, color=color, linewidth=1.5, alpha=0.9)
        if len(full_path) > 0:
            x, y = full_path[0][1], full_path[0][0]
            ax.scatter([x], [y], c=color, s=36, marker="o", edgecolors="black", linewidths=0.5)

    for kind, loc in task_points:
        x, y = loc[1], loc[0]
        if kind == "pickup":
            ax.scatter([x], [y], c="green", s=36, marker="^", edgecolors="black", linewidths=0.5)
        else:
            ax.scatter([x], [y], c="orange", s=36, marker="s", edgecolors="black", linewidths=0.5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, help="Path to showcase case json")
    parser.add_argument("--map", required=True, help="Path to map file")
    parser.add_argument("--out_dir", default=None, help="Output directory")
    args = parser.parse_args()

    with open(args.case, "r", encoding="utf-8") as f:
        payload = json.load(f)

    grid, _ = load_map(args.map)
    base_dir = args.out_dir or os.path.dirname(args.case)
    os.makedirs(base_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(args.case))[0]
    expert_path = os.path.join(base_dir, f"{base_name}_expert.png")
    model_path = os.path.join(base_dir, f"{base_name}_model.png")

    expert_paths = _paths_by_agent(payload["expert_status"])
    model_paths = _paths_by_agent(payload["model_status"])
    shared_agents = {
        idx for idx, path in expert_paths.items()
        if idx in model_paths and path == model_paths[idx]
    }

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    _plot_case(ax, grid, payload["expert_status"], "expert", skip_agents=shared_agents)
    fig.tight_layout(pad=0.1)
    fig.savefig(expert_path, dpi=150, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    _plot_case(ax, grid, payload["model_status"], "model", skip_agents=shared_agents)
    fig.tight_layout(pad=0.1)
    fig.savefig(model_path, dpi=150, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    main()
