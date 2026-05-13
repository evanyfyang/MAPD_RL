import heapq
from collections import deque
from itertools import combinations, permutations, product

import math
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import Circle, FancyArrowPatch, PathPatch, Rectangle, RegularPolygon
from matplotlib.path import Path


def build_world():
    rows, cols = 5, 13
    obstacles = {(2, c) for c in range(3, 10)}

    endpoints = set()
    for c in range(3, 10):
        endpoints.add((1, c))
        endpoints.add((3, c))
    for r in range(1, 4):
        endpoints.add((r, 1))
        endpoints.add((r, 11))

    agents = [(1, 1), (3, 1), (1, 11), (3, 11), (1, 6)]

    tasks = [
        {"pickup": (3, 3), "delivery": (3, 9), "color": "#E67E22"},
        {"pickup": (1, 3), "delivery": (1, 8), "color": "#1ABC9C"},
        {"pickup": (3, 5), "delivery": (3, 4), "color": "#8E44AD"},
    ]

    return rows, cols, obstacles, endpoints, agents, tasks


def in_bounds(rows, cols, pos):
    r, c = pos
    return 0 <= r < rows and 0 <= c < cols


def compute_all_pairs_distances(rows, cols, obstacles):
    distances = {}
    for r in range(rows):
        for c in range(cols):
            start = (r, c)
            if start in obstacles:
                continue
            dist = {start: 0}
            queue = deque([start])
            while queue:
                cr, cc = queue.popleft()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    nxt = (nr, nc)
                    if not in_bounds(rows, cols, nxt):
                        continue
                    if nxt in obstacles or nxt in dist:
                        continue
                    dist[nxt] = dist[(cr, cc)] + 1
                    queue.append(nxt)
            distances[start] = dist
    return distances


def neighbors(rows, cols, pos, blocked, allow_wait=True):
    r, c = pos
    candidates = [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]
    if allow_wait:
        candidates.append((r, c))
    valid = []
    for nxt in candidates:
        if not in_bounds(rows, cols, nxt):
            continue
        if nxt in blocked:
            continue
        valid.append(nxt)
    return valid


def compute_individual_costs(agents, tasks, distances):
    costs = []
    for agent in agents:
        row = []
        for task in tasks:
            pickup = task["pickup"]
            delivery = task["delivery"]
            dist_ap = distances[agent].get(pickup)
            dist_pd = distances[pickup].get(delivery)
            if dist_ap is None or dist_pd is None:
                row.append(float("inf"))
            else:
                row.append(dist_ap + dist_pd)
        costs.append(row)
    return costs


def hungarian_like_assignment(agents, tasks, distances, rows, cols, obstacles):
    costs = compute_individual_costs(agents, tasks, distances)
    agent_count = len(agents)
    task_count = len(tasks)

    best_total = float("inf")
    best_assignments = []
    for agent_idxs in combinations(range(agent_count), task_count):
        for perm_tasks in permutations(range(task_count)):
            total = 0
            for idx, task_idx in enumerate(perm_tasks):
                total += costs[agent_idxs[idx]][task_idx]
            if total < best_total:
                best_total = total
                best_assignments = [{agent_idxs[i]: perm_tasks[i] for i in range(task_count)}]
            elif total == best_total:
                best_assignments.append({agent_idxs[i]: perm_tasks[i] for i in range(task_count)})

    if len(best_assignments) == 1:
        return best_assignments[0], best_total

    worst_collision_cost = -1
    worst_assignment = best_assignments[0]
    for assignment in best_assignments:
        assigned_agents, assigned_tasks, idle_agents, _ = assignment_to_ordered_lists(assignment, agents, tasks)
        static_blocked = set(obstacles) | set(idle_agents)
        cost = mapf_sum_of_costs(rows, cols, obstacles, assigned_agents, assigned_tasks, static_blocked, distances)
        if cost is None:
            continue
        if cost > worst_collision_cost:
            worst_collision_cost = cost
            worst_assignment = assignment
    return worst_assignment, best_total


def assignment_to_ordered_lists(assignment, agents, tasks):
    agent_ids = sorted(assignment.keys())
    assigned_agents = [agents[i] for i in agent_ids]
    assigned_tasks = [(tasks[assignment[i]]["pickup"], tasks[assignment[i]]["delivery"]) for i in agent_ids]
    idle_agents = [agents[i] for i in range(len(agents)) if i not in assignment]
    return assigned_agents, assigned_tasks, idle_agents, agent_ids


def mapf_plan(rows, cols, obstacles, agent_starts, tasks, static_blocked, distances):
    if not agent_starts:
        return 0, []

    starts = tuple(agent_starts)
    stages = []
    for i, start in enumerate(agent_starts):
        pickup, _ = tasks[i]
        stages.append(1 if start == pickup else 0)
    stages = tuple(stages)

    def heuristic(positions, stage_tuple):
        total = 0
        for i, pos in enumerate(positions):
            pickup, delivery = tasks[i]
            stage = stage_tuple[i]
            if stage == 0:
                dist_ap = distances[pos].get(pickup, float("inf"))
                dist_pd = distances[pickup].get(delivery, float("inf"))
                total += dist_ap + dist_pd
            elif stage == 1:
                total += distances[pos].get(delivery, float("inf"))
        return total

    init = (starts, stages)
    init_h = heuristic(*init)
    if init_h == float("inf"):
        return None

    open_heap = [(init_h, 0, init)]
    best_g = {init: 0}
    parents = {init: None}

    while open_heap:
        _, g, state = heapq.heappop(open_heap)
        if g != best_g.get(state):
            continue
        positions, stage_tuple = state
        if all(stage == 2 for stage in stage_tuple):
            path_states = []
            cursor = state
            while cursor is not None:
                path_states.append(cursor)
                cursor = parents[cursor]
            path_states.reverse()
            paths = [[] for _ in range(len(agent_starts))]
            for positions, _ in path_states:
                for i, pos in enumerate(positions):
                    paths[i].append(pos)
            return g, paths

        active_count = sum(1 for stage in stage_tuple if stage < 2)
        move_options = []
        for i, pos in enumerate(positions):
            if stage_tuple[i] == 2:
                move_options.append([pos])
            else:
                move_options.append(neighbors(rows, cols, pos, obstacles | static_blocked, allow_wait=True))

        for moves in product(*move_options):
            if len(set(moves)) < len(moves):
                continue
            collision = False
            for i in range(len(moves)):
                for j in range(i + 1, len(moves)):
                    if positions[i] == moves[j] and positions[j] == moves[i]:
                        collision = True
                        break
                if collision:
                    break
            if collision:
                continue

            new_stages = list(stage_tuple)
            for i, new_pos in enumerate(moves):
                pickup, delivery = tasks[i]
                if new_stages[i] == 0 and new_pos == pickup:
                    new_stages[i] = 1
                elif new_stages[i] == 1 and new_pos == delivery:
                    new_stages[i] = 2

            new_state = (tuple(moves), tuple(new_stages))
            new_g = g + active_count
            if new_g < best_g.get(new_state, float("inf")):
                best_g[new_state] = new_g
                parents[new_state] = state
                h = heuristic(*new_state)
                if h == float("inf"):
                    continue
                heapq.heappush(open_heap, (new_g + h, new_g, new_state))

    return None, None


def mapf_sum_of_costs(rows, cols, obstacles, agent_starts, tasks, static_blocked, distances):
    cost, _ = mapf_plan(rows, cols, obstacles, agent_starts, tasks, static_blocked, distances)
    return cost


def find_collision_optimal_assignment(rows, cols, obstacles, agents, tasks, distances):
    agent_count = len(agents)
    task_count = len(tasks)
    best_cost = float("inf")
    best_assignment = None
    for agent_idxs in combinations(range(agent_count), task_count):
        for perm_tasks in permutations(range(task_count)):
            assignment = {agent_idxs[i]: perm_tasks[i] for i in range(task_count)}
            assigned_agents, assigned_tasks, idle_agents, _ = assignment_to_ordered_lists(assignment, agents, tasks)
            static_blocked = set(obstacles) | set(idle_agents)
            cost = mapf_sum_of_costs(rows, cols, obstacles, assigned_agents, assigned_tasks, static_blocked, distances)
            if cost is None:
                continue
            if cost < best_cost:
                best_cost = cost
                best_assignment = assignment
    return best_assignment, best_cost


def build_task_color_map(assignment, agent_colors, task_count):
    task_colors = ["#888888"] * task_count
    for agent_idx, task_idx in assignment.items():
        task_colors[task_idx] = agent_colors[agent_idx]
    return task_colors


def adjust_color(hex_color, factor):
    r, g, b = mcolors.to_rgb(hex_color)
    r = min(max(r * factor, 0.0), 1.0)
    g = min(max(g * factor, 0.0), 1.0)
    b = min(max(b * factor, 0.0), 1.0)
    return (r, g, b)


def build_edge_offsets(paths, agent_ids, offset_step):
    edge_agents = {}
    for agent_idx, path in zip(agent_ids, paths):
        for i in range(len(path) - 1):
            p0, p1 = path[i], path[i + 1]
            if p0 == p1:
                continue
            key = tuple(sorted([p0, p1]))
            edge_agents.setdefault(key, []).append(agent_idx)

    edge_offsets = {}
    for key, agents in edge_agents.items():
        agents_sorted = sorted(agents)
        count = len(agents_sorted)
        offsets = []
        for i in range(count):
            offset_index = i - (count - 1) / 2
            offsets.append(offset_index * offset_step)
        edge_offsets[key] = dict(zip(agents_sorted, offsets))
    return edge_offsets


def build_agent_points(path, agent_idx, edge_offsets, cell_size):
    segments = []
    for i in range(len(path) - 1):
        p0, p1 = path[i], path[i + 1]
        if p0 == p1:
            continue
        key = tuple(sorted([p0, p1]))
        offset = edge_offsets.get(key, {}).get(agent_idx, 0.0)
        dr = p1[0] - p0[0]
        dc = p1[1] - p0[1]
        axis = "h" if dr == 0 else "v"
        segments.append({"p0": p0, "p1": p1, "dr": dr, "dc": dc, "axis": axis, "offset": offset})

    if not segments:
        return []

    turn_offset = cell_size * 0.18
    overlap_offset = cell_size * 0.16
    edge_counts = {}
    for seg in segments:
        key = tuple(sorted([seg["p0"], seg["p1"]]))
        edge_counts[key] = edge_counts.get(key, 0) + 1
    edge_seen = {}

    def point_for(cell, axis, offset):
        x = (cell[1] + 0.5) * cell_size + (offset if axis == "v" else 0.0)
        y = (cell[0] + 0.5) * cell_size + (offset if axis == "h" else 0.0)
        return x, y

    points = [point_for(segments[0]["p0"], segments[0]["axis"], segments[0]["offset"])]

    for i, seg in enumerate(segments):
        end_cell = seg["p1"]
        end_offset = seg["offset"]
        key = tuple(sorted([seg["p0"], seg["p1"]]))
        if edge_counts.get(key, 0) > 1:
            seen = edge_seen.get(key, 0)
            edge_seen[key] = seen + 1
            end_offset += overlap_offset if seen % 2 == 0 else -overlap_offset
        if i < len(segments) - 1:
            nxt = segments[i + 1]
            if seg["axis"] == nxt["axis"] and seg["dr"] == -nxt["dr"] and seg["dc"] == -nxt["dc"]:
                offset_in = seg["offset"] + turn_offset
                offset_out = nxt["offset"] - turn_offset
                points.append(point_for(end_cell, seg["axis"], offset_in))
                points.append(point_for(end_cell, nxt["axis"], offset_out))
                continue
        points.append(point_for(end_cell, seg["axis"], end_offset))

    return points


def build_smooth_path(points, corner_radius):
    if len(points) < 2:
        return None

    cleaned = [points[0]]
    for p in points[1:]:
        if p != cleaned[-1]:
            cleaned.append(p)
    if len(cleaned) < 2:
        return None

    def normalize(vec):
        vx, vy = vec
        length = (vx * vx + vy * vy) ** 0.5
        if length == 0:
            return 0.0, 0.0
        return vx / length, vy / length

    vertices = [cleaned[0]]
    codes = [Path.MOVETO]

    for i in range(1, len(cleaned) - 1):
        p_prev = cleaned[i - 1]
        p = cleaned[i]
        p_next = cleaned[i + 1]
        v1 = (p[0] - p_prev[0], p[1] - p_prev[1])
        v2 = (p_next[0] - p[0], p_next[1] - p[1])
        n1 = normalize(v1)
        n2 = normalize(v2)
        if n1 == n2:
            vertices.append(p)
            codes.append(Path.LINETO)
            continue

        len1 = (v1[0] * v1[0] + v1[1] * v1[1]) ** 0.5
        len2 = (v2[0] * v2[0] + v2[1] * v2[1]) ** 0.5
        radius = min(corner_radius, len1 * 0.45, len2 * 0.45)

        p1 = (p[0] - n1[0] * radius, p[1] - n1[1] * radius)
        p2 = (p[0] + n2[0] * radius, p[1] + n2[1] * radius)
        vertices.append(p1)
        codes.append(Path.LINETO)
        vertices.append(p)
        codes.append(Path.CURVE3)
        vertices.append(p2)
        codes.append(Path.CURVE3)

    vertices.append(cleaned[-1])
    codes.append(Path.LINETO)
    return Path(vertices, codes)


def draw_case(output_path: str, assignment, paths, path_agent_ids) -> None:
    rows, cols, obstacles, endpoints, agents, tasks = build_world()
    cell_size = 1.0

    fig, ax = plt.subplots(figsize=(cols, rows))
    ax.set_aspect("equal")
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.axis("off")

    for r in range(rows):
        for c in range(cols):
            x, y = c * cell_size, r * cell_size
            if (r, c) in obstacles:
                color = "#111111"
            elif (r, c) in endpoints:
                color = "#B0B0B0"
            else:
                color = "white"
            rect = Rectangle((x, y), cell_size, cell_size, facecolor=color, edgecolor="black", linewidth=2.2)
            ax.add_patch(rect)

    agent_colors = ["#E74C3C", "#3498DB", "#2ECC71", "#F1C40F", "#9B59B6"]
    center_offset = cell_size * 0.5
    agent_radius = cell_size * 0.42

    offset_step = cell_size * 0.22
    edge_offsets = build_edge_offsets(paths, path_agent_ids, offset_step)
    for agent_idx, path in zip(path_agent_ids, paths):
        base_color = agent_colors[agent_idx]
        line_color = adjust_color(base_color, 0.82)
        points = build_agent_points(path, agent_idx, edge_offsets, cell_size)
        curve = build_smooth_path(points, corner_radius=cell_size * 0.28)
        if curve is None:
            continue
        ax.add_patch(PathPatch(curve, facecolor="none", edgecolor=line_color, linewidth=3.2,
                               capstyle="round", joinstyle="round", zorder=4))
        if len(points) >= 2:
            p0 = points[-2]
            p1 = points[-1]
            dx = p1[0] - p0[0]
            dy = p1[1] - p0[1]
            angle = math.atan2(dy, dx)
            head_size = cell_size * 0.16
            back_offset = cell_size * 0.09
            hx = p1[0] - math.cos(angle) * back_offset
            hy = p1[1] - math.sin(angle) * back_offset
            head = RegularPolygon(
                (hx, hy),
                numVertices=3,
                radius=head_size,
                orientation=angle - math.pi / 2,
                facecolor=line_color,
                edgecolor="none",
                linewidth=0.0,
                zorder=5,
            )
            ax.add_patch(head)
    for (r, c), color in zip(agents, agent_colors):
        cx = c * cell_size + center_offset
        cy = r * cell_size + center_offset
        circle = Circle((cx, cy), radius=agent_radius, facecolor=color, edgecolor="black", linewidth=2.2, zorder=3)
        ax.add_patch(circle)

    task_colors = build_task_color_map(assignment, agent_colors, len(tasks))
    shape_radius = cell_size * 0.35
    square_size = cell_size * 0.64
    for task_idx, task in enumerate(tasks):
        pr, pc = task["pickup"]
        dr, dc = task["delivery"]
        color = task_colors[task_idx]

        px = pc * cell_size + center_offset
        py = pr * cell_size + center_offset
        triangle = RegularPolygon((px, py), numVertices=3, radius=shape_radius, orientation=0.0,
                                  facecolor=color, edgecolor="black", linewidth=2.0, zorder=3)
        ax.add_patch(triangle)

        dx = dc * cell_size + center_offset - square_size / 2
        dy = dr * cell_size + center_offset - square_size / 2
        square = Rectangle((dx, dy), square_size, square_size, facecolor=color, edgecolor="black", linewidth=2.0, zorder=3)
        ax.add_patch(square)

    plt.tight_layout(pad=0.2)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def run_assignment_evaluation():
    rows, cols, obstacles, _, agents, tasks = build_world()
    distances = compute_all_pairs_distances(rows, cols, obstacles)

    hungarian_assignment, _ = hungarian_like_assignment(agents, tasks, distances, rows, cols, obstacles)
    assigned_agents, assigned_tasks, idle_agents, agent_ids = assignment_to_ordered_lists(
        hungarian_assignment, agents, tasks
    )
    static_blocked = set(obstacles) | set(idle_agents)
    cost_case1, paths_case1 = mapf_plan(
        rows, cols, obstacles, assigned_agents, assigned_tasks, static_blocked, distances
    )

    best_assignment, cost_case2 = find_collision_optimal_assignment(
        rows, cols, obstacles, agents, tasks, distances
    )

    assigned_agents2, assigned_tasks2, idle_agents2, agent_ids2 = assignment_to_ordered_lists(
        best_assignment, agents, tasks
    )
    static_blocked2 = set(obstacles) | set(idle_agents2)
    cost_case2_verified, paths_case2 = mapf_plan(
        rows, cols, obstacles, assigned_agents2, assigned_tasks2, static_blocked2, distances
    )

    print("Case 1 (Hungarian ignore-collision assignment) collision-aware total length:", cost_case1)
    print("Case 2 (collision-optimal assignment) collision-aware total length:", cost_case2_verified)

    draw_case("case_hungarian.png", assignment=hungarian_assignment,
              paths=paths_case1, path_agent_ids=agent_ids)
    draw_case("case_optimal.png", assignment=best_assignment,
              paths=paths_case2, path_agent_ids=agent_ids2)


if __name__ == "__main__":
    run_assignment_evaluation()
