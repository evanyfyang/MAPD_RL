#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import random
import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

from model.spmpnn_pretrain import RingRegressionHead, SPMPNNGridEncoder


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class GridMap:
    def __init__(self, map_path=None, blocked_mask=None, source_name=None):
        self.map_path = map_path
        self.source_name = source_name or map_path or "custom_map"
        self.height = 0
        self.width = 0
        self.grid_chars = []
        self.blocked_mask = None
        self.free_indices = []
        if blocked_mask is not None:
            self._init_from_mask(blocked_mask)
        else:
            self._read_map()

    def _init_from_mask(self, blocked_mask):
        blocked_mask = np.asarray(blocked_mask, dtype=np.bool_)
        if blocked_mask.ndim != 2:
            raise ValueError("blocked_mask must be a 2D array")
        self.height, self.width = blocked_mask.shape
        self.blocked_mask = blocked_mask
        self.grid_chars = []
        for r in range(self.height):
            row = []
            for c in range(self.width):
                row.append("@" if self.blocked_mask[r, c] else ".")
            self.grid_chars.append("".join(row))
        self._build_free_indices()

    def _build_free_indices(self):
        self.free_indices = []
        for r in range(self.height):
            for c in range(self.width):
                if not self.blocked_mask[r, c]:
                    self.free_indices.append(self.rc_to_idx(r, c))
        if len(self.free_indices) == 0:
            raise ValueError("No free cells found in map.")

    def _read_map(self):
        with open(self.map_path, "r", encoding="utf-8") as f:
            first = f.readline().strip()
            h, w = first.split(",")
            self.height, self.width = int(h), int(w)
            _ = f.readline()
            _ = f.readline()
            _ = f.readline()
            lines = [line.strip() for line in f if line.strip()]

        if len(lines) != self.height:
            raise ValueError(
                f"Map rows mismatch: expected {self.height}, got {len(lines)} from {self.map_path}"
            )
        for row in lines:
            if len(row) != self.width:
                raise ValueError(
                    f"Map cols mismatch: expected {self.width}, got {len(row)} in {self.map_path}"
                )

        self.grid_chars = lines
        blocked = np.zeros((self.height, self.width), dtype=np.bool_)
        for r in range(self.height):
            for c in range(self.width):
                blocked[r, c] = self.grid_chars[r][c] == "@"
        self.blocked_mask = blocked
        self._build_free_indices()

    def rc_to_idx(self, r, c):
        return r * self.width + c

    def idx_to_rc(self, idx):
        return idx // self.width, idx % self.width

    @property
    def num_nodes(self):
        return self.height * self.width


def build_map_pool(map_paths, obstacle_drop_prob_min, obstacle_drop_prob_max, variants_per_map):
    pool = []
    for mp in map_paths:
        base = GridMap(map_path=mp, source_name=os.path.basename(mp))
        pool.append(base)

        blocked_pos = np.argwhere(base.blocked_mask)
        if blocked_pos.shape[0] == 0 or int(variants_per_map) <= 0:
            continue
        for vidx in range(int(variants_per_map)):
            p = random.uniform(float(obstacle_drop_prob_min), float(obstacle_drop_prob_max))
            mask = base.blocked_mask.copy()
            for rr, cc in blocked_pos:
                if random.random() < p:
                    mask[rr, cc] = False
            name = f"{os.path.basename(mp)}#drop{vidx}"
            pool.append(GridMap(blocked_mask=mask, source_name=name))
    return pool


class SPDistanceCache:
    """
    Precompute shortest-path helpers on static grid:
    - distance edges for encoder (dist_1..dist_k)
    - neighbors_by_hop for ring/heat labels
    - ring_size for fast ratio normalization
    """

    def __init__(
        self,
        grid_map,
        max_distance=3,
        max_hop=8,
        rings=None,
        cache_dir=None,
        verbose=False,
    ):
        self.grid_map = grid_map
        self.max_distance = int(max_distance)
        self.max_hop = int(max_hop)
        self.rings = rings if rings is not None else []
        self.cache_dir = cache_dir
        self.verbose = bool(verbose)

        self.adj = self._build_adj()
        self.distance_edges = {}
        self.neighbors_by_hop = {}
        self.ring_size = {}
        self.loaded_from_cache = False
        self._precompute_or_load()

    def _build_adj(self):
        adj = [[] for _ in range(self.grid_map.num_nodes)]
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for r in range(self.grid_map.height):
            for c in range(self.grid_map.width):
                if self.grid_map.blocked_mask[r, c]:
                    continue
                u = self.grid_map.rc_to_idx(r, c)
                for dr, dc in dirs:
                    nr, nc = r + dr, c + dc
                    if (
                        0 <= nr < self.grid_map.height
                        and 0 <= nc < self.grid_map.width
                        and not self.grid_map.blocked_mask[nr, nc]
                    ):
                        v = self.grid_map.rc_to_idx(nr, nc)
                        adj[u].append(v)
        return adj

    def _cache_key(self):
        bm = np.ascontiguousarray(self.grid_map.blocked_mask.astype(np.uint8))
        digest = hashlib.md5(bm.tobytes()).hexdigest()
        ring_str = "_".join([f"{a}-{b}" for a, b in self.rings]) if self.rings else "norings"
        return (
            f"ringcache_{self.grid_map.source_name}_"
            f"h{self.grid_map.height}_w{self.grid_map.width}_"
            f"d{self.max_distance}_mh{self.max_hop}_{ring_str}_{digest}.pt"
        )

    def _cache_path(self):
        if self.cache_dir is None:
            return None
        os.makedirs(self.cache_dir, exist_ok=True)
        return os.path.join(self.cache_dir, self._cache_key())

    def _precompute_or_load(self):
        cache_path = self._cache_path()
        if cache_path is not None and os.path.exists(cache_path):
            data = torch.load(cache_path, map_location="cpu")
            self.distance_edges = data["distance_edges"]
            self.neighbors_by_hop = data["neighbors_by_hop"]
            self.ring_size = data["ring_size"]
            self.loaded_from_cache = True
            if self.verbose:
                print(f"[SPCache] loaded: {cache_path}")
            return

        t0 = time.time()
        self._precompute()
        if cache_path is not None:
            torch.save(
                {
                    "distance_edges": self.distance_edges,
                    "neighbors_by_hop": self.neighbors_by_hop,
                    "ring_size": self.ring_size,
                },
                cache_path,
            )
            if self.verbose:
                print(f"[SPCache] saved: {cache_path}")
        if self.verbose:
            print(f"[SPCache] computed {self.grid_map.source_name} in {time.time() - t0:.2f}s")

    def _precompute(self):
        dist_edge_lists = {f"dist_{d}": [] for d in range(1, self.max_distance + 1)}

        for src in self.grid_map.free_indices:
            q = deque([src])
            visited = {src: 0}
            hop_dict = {d: [] for d in range(1, self.max_hop + 1)}

            while q:
                cur = q.popleft()
                dcur = visited[cur]
                if dcur >= self.max_hop:
                    continue
                for nb in self.adj[cur]:
                    if nb in visited:
                        continue
                    nd = dcur + 1
                    visited[nb] = nd
                    q.append(nb)
                    if nd <= self.max_hop:
                        hop_dict[nd].append(nb)
                    if nd <= self.max_distance:
                        dist_edge_lists[f"dist_{nd}"].append([src, nb])
            self.neighbors_by_hop[src] = hop_dict

        for d in range(1, self.max_distance + 1):
            key = f"dist_{d}"
            edges = dist_edge_lists[key]
            if len(edges) == 0:
                self.distance_edges[key] = torch.zeros((2, 0), dtype=torch.long)
            else:
                self.distance_edges[key] = torch.tensor(edges, dtype=torch.long).t().contiguous()

        for src in self.grid_map.free_indices:
            rs = []
            for lo, hi in self.rings:
                c = 0
                for d in range(lo, hi + 1):
                    c += len(self.neighbors_by_hop[src].get(d, []))
                rs.append(c)
            self.ring_size[src] = rs

    def get_nodes_in_ring(self, src, ring_idx):
        lo, hi = self.rings[ring_idx]
        out = []
        for d in range(lo, hi + 1):
            out.extend(self.neighbors_by_hop[src].get(d, []))
        return out


def build_adaptive_rings(R):
    R = int(max(1, R))
    rings = []
    if R >= 1:
        rings.append((1, 1))
    if R >= 2:
        rings.append((2, 2))
    if R >= 3:
        rings.append((3, min(4, R)))

    start = 5
    while start <= R:
        end = min(R, 2 * start - 2)
        rings.append((start, end))
        start = end + 1
    return rings


def shortest_path(adj, src, dst):
    if src == dst:
        return [src]
    q = deque([src])
    parent = {src: -1}
    while q:
        cur = q.popleft()
        for nb in adj[cur]:
            if nb in parent:
                continue
            parent[nb] = cur
            if nb == dst:
                path = [dst]
                x = dst
                while parent[x] != -1:
                    x = parent[x]
                    path.append(x)
                path.reverse()
                return path
            q.append(nb)
    return []


class RandomStateGenerator:
    def __init__(
        self,
        grid_map,
        min_agents=10,
        max_agents=50,
        min_tasks=20,
        max_tasks=100,
        clip_max=4.0,
        delivering_fraction=0.15,
    ):
        self.grid_map = grid_map
        self.min_agents = int(min_agents)
        self.max_agents = int(max_agents)
        self.min_tasks = int(min_tasks)
        self.max_tasks = int(max_tasks)
        self.clip_max = float(clip_max)
        self.delivering_fraction = float(delivering_fraction)

    def _sample_cells(self, free_indices, n, rng, replace=False):
        n_free = len(free_indices)
        if not replace and n <= n_free:
            return rng.sample(free_indices, n)
        return [rng.choice(free_indices) for _ in range(n)]

    def sample_state(self, grid_map=None, rng=None):
        if rng is None:
            rng = random
        gm = grid_map if grid_map is not None else self.grid_map

        num_agents = rng.randint(self.min_agents, self.max_agents)
        num_tasks = rng.randint(self.min_tasks, self.max_tasks)

        blocked_vec = np.zeros((gm.num_nodes,), dtype=np.float32)
        for idx in range(gm.num_nodes):
            r, c = gm.idx_to_rc(idx)
            blocked_vec[idx] = 1.0 if gm.blocked_mask[r, c] else 0.0

        agent_occ = np.zeros((gm.num_nodes,), dtype=np.float32)
        pickup_count = np.zeros((gm.num_nodes,), dtype=np.float32)
        delivery_count = np.zeros((gm.num_nodes,), dtype=np.float32)

        agent_cells = self._sample_cells(
            gm.free_indices, num_agents, rng=rng, replace=(num_agents > len(gm.free_indices))
        )
        for cell in agent_cells:
            agent_occ[cell] += 1.0

        free_pickup_cells = []
        free_delivery_cells = []
        for _ in range(num_tasks):
            p = rng.choice(gm.free_indices)
            d = rng.choice(gm.free_indices)
            while d == p:
                d = rng.choice(gm.free_indices)
            free_pickup_cells.append(p)
            free_delivery_cells.append(d)
            pickup_count[p] += 1.0
            delivery_count[d] += 1.0

        delivering_cnt = int(round(num_agents * self.delivering_fraction))
        delivering_cnt = max(0, min(num_agents, delivering_cnt))
        delivering_agent_indices = rng.sample(list(range(num_agents)), delivering_cnt) if delivering_cnt > 0 else []
        delivering_paths = []
        delivering_targets = []
        for ai in delivering_agent_indices:
            start = agent_cells[ai]
            target = rng.choice(gm.free_indices)
            while target == start:
                target = rng.choice(gm.free_indices)
            delivering_targets.append((start, target))

        state = {
            "grid_name": gm.source_name,
            "blocked_vec": blocked_vec,
            "agent_occ_raw": agent_occ,
            "pickup_count_raw": pickup_count,
            "delivery_count_raw": delivery_count,
            "free_cells": gm.free_indices,
            "agent_cells": agent_cells,
            "free_pickup_cells": free_pickup_cells,
            "free_delivery_cells": free_delivery_cells,
            "delivering_targets": delivering_targets,
            "delivering_paths": delivering_paths,
            "node_features": None,
        }
        return state


def build_delivering_paths_and_delivery_counts(state, dist_cache, rng):
    gm = dist_cache.grid_map
    delivery_count = state["delivery_count_raw"].copy()
    paths = []
    for start, target in state["delivering_targets"]:
        path = shortest_path(dist_cache.adj, start, target)
        if len(path) == 0:
            continue
        if len(path) > 2:
            keep = rng.randint(max(2, len(path) // 2), len(path))
            path = path[:keep]
        paths.append(path)
        delivery_count[target] += 1.0
    state["delivering_paths"] = paths
    state["delivery_count_raw"] = delivery_count
    return state


def build_heatmap(delivering_paths, dist_cache, R_heat):
    n = dist_cache.grid_map.num_nodes
    H = np.zeros((n,), dtype=np.float32)
    R_heat = int(max(1, R_heat))
    for path in delivering_paths:
        if len(path) == 0:
            continue
        pw = 1.0 / float(len(path))
        for u in path:
            H[u] += pw
            hop_map = dist_cache.neighbors_by_hop.get(u, {})
            for d in range(1, R_heat + 1):
                val = max(0.0, 1.0 - float(d) / float(R_heat))
                if val <= 0:
                    continue
                for v in hop_map.get(d, []):
                    H[v] += pw * val
    H = np.log1p(H)
    m = float(np.max(H))
    if m > 0:
        H = H / m
    return H.astype(np.float32)


def build_node_features(state, clip_max, heatmap):
    pickup_norm = np.minimum(state["pickup_count_raw"], float(clip_max)) / float(clip_max)
    delivery_norm = np.minimum(state["delivery_count_raw"], float(clip_max)) / float(clip_max)
    node_features = np.stack(
        [
            state["blocked_vec"],
            state["agent_occ_raw"],
            pickup_norm,
            delivery_norm,
            heatmap,
        ],
        axis=-1,
    ).astype(np.float32)
    state["node_features"] = node_features
    return state


def sample_centers(free_cells, special_cells, num_centers):
    target = int(num_centers)
    if target <= 0:
        return []
    half = target // 2
    chosen = []
    if len(special_cells) > 0:
        n_special = min(half, len(special_cells))
        chosen.extend(random.sample(special_cells, n_special))
    remain = target - len(chosen)
    if remain > 0:
        free_pool = [x for x in free_cells if x not in set(chosen)]
        if len(free_pool) >= remain:
            chosen.extend(random.sample(free_pool, remain))
        elif len(free_pool) > 0:
            chosen.extend(free_pool)
            still = target - len(chosen)
            if still > 0:
                chosen.extend(random.choices(free_cells, k=still))
        else:
            chosen.extend(random.choices(free_cells, k=remain))
    return chosen


def build_ring_labels_for_centers(state, centers, dist_cache, rings, heatmap):
    labels = []
    for c in centers:
        row = np.zeros((len(rings), 4), dtype=np.float32)
        for ridx, _ in enumerate(rings):
            nodes = dist_cache.get_nodes_in_ring(c, ridx)
            denom = max(1, len(nodes))
            if len(nodes) == 0:
                row[ridx, :] = 0.0
                continue
            row[ridx, 0] = float(np.sum(state["agent_occ_raw"][nodes])) / float(denom)
            row[ridx, 1] = float(np.sum(state["pickup_count_raw"][nodes])) / float(denom)
            row[ridx, 2] = float(np.sum(state["delivery_count_raw"][nodes])) / float(denom)
            row[ridx, 3] = float(np.mean(heatmap[nodes]))
        labels.append(row)
    return np.stack(labels, axis=0) if len(labels) > 0 else np.zeros((0, len(rings), 4), dtype=np.float32)


class RandomStateIterableDataset(IterableDataset):
    def __init__(self, map_pool, min_agents, max_agents, min_tasks, max_tasks, clip_max, delivering_fraction, base_seed):
        super().__init__()
        self.map_pool = map_pool
        self.min_agents = int(min_agents)
        self.max_agents = int(max_agents)
        self.min_tasks = int(min_tasks)
        self.max_tasks = int(max_tasks)
        self.clip_max = float(clip_max)
        self.delivering_fraction = float(delivering_fraction)
        self.base_seed = int(base_seed)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        wid = 0 if worker_info is None else int(worker_info.id)
        seed = self.base_seed + wid * 100003 + int(time.time()) % 100000
        rng = random.Random(seed)
        generator = RandomStateGenerator(
            grid_map=self.map_pool[0],
            min_agents=self.min_agents,
            max_agents=self.max_agents,
            min_tasks=self.min_tasks,
            max_tasks=self.max_tasks,
            clip_max=self.clip_max,
            delivering_fraction=self.delivering_fraction,
        )
        while True:
            gm = rng.choice(self.map_pool)
            yield generator.sample_state(grid_map=gm, rng=rng)


def gather_batch(encoder, ring_head, batch_states, dist_cache_by_name, dist_edges_dev_by_name, rings, args, device):
    ring_losses = []
    abs_err_collect = []
    centers_total = 0

    for st in batch_states:
        dist_cache = dist_cache_by_name[st["grid_name"]]
        st = build_delivering_paths_and_delivery_counts(st, dist_cache, random)
        heatmap = build_heatmap(st["delivering_paths"], dist_cache, args.R_heat)
        st = build_node_features(st, clip_max=args.clip_max, heatmap=heatmap)

        special_mask = (st["agent_occ_raw"] > 0) | (st["pickup_count_raw"] > 0) | (st["delivery_count_raw"] > 0)
        special_cells = [idx for idx in st["free_cells"] if special_mask[idx]]
        centers = sample_centers(st["free_cells"], special_cells, args.centers_per_graph)
        centers_total += len(centers)
        if len(centers) == 0:
            continue

        labels_np = build_ring_labels_for_centers(st, centers, dist_cache, rings, heatmap)
        if labels_np.shape[0] == 0:
            continue

        x = torch.from_numpy(st["node_features"]).to(device=device)
        h = encoder(x, dist_edges_dev_by_name[st["grid_name"]])
        center_idx = torch.tensor(centers, dtype=torch.long, device=device)
        z = h[center_idx]
        pred = ring_head(z)
        target = torch.from_numpy(labels_np).to(device=device)

        per_elem = F.smooth_l1_loss(pred, target, reduction="none")
        ring_w = torch.ones((1, len(rings), 1), device=device)
        for ridx in range(min(args.near_weighted_rings, len(rings))):
            ring_w[:, ridx, :] = float(args.near_ring_weight)
        loss = (per_elem * ring_w).mean()
        ring_losses.append(loss)

        with torch.no_grad():
            abs_err_collect.append(torch.abs(pred - target).detach().cpu())

    if len(ring_losses) == 0:
        return None
    batch_loss = torch.stack(ring_losses).mean()

    abs_err = torch.cat(abs_err_collect, dim=0)  # [num_centers, num_rings, 4]
    mae_task = abs_err.mean(dim=(0, 1)).numpy().tolist()  # 4
    mae_ring = abs_err.mean(dim=(0, 2)).numpy().tolist()  # num_rings
    mae_all = float(abs_err.mean().item())
    return {
        "loss": batch_loss,
        "mae_all": mae_all,
        "mae_task": mae_task,
        "mae_ring": mae_ring,
        "centers_total": centers_total,
    }


def run_epoch(mode, encoder, ring_head, generator, map_pool, dist_cache_by_name, dist_edges_dev_by_name, optimizer, rings, args, device):
    train_mode = mode == "train"
    encoder.train(train_mode)
    ring_head.train(train_mode)

    graphs_per_epoch = args.train_graphs_per_epoch if train_mode else args.val_graphs_per_epoch
    num_batches = max(1, int(np.ceil(graphs_per_epoch / args.batch_size_graph)))

    sum_loss = 0.0
    sum_mae = 0.0
    sum_task = np.zeros((4,), dtype=np.float64)
    sum_ring = np.zeros((len(rings),), dtype=np.float64)
    valid_batches = 0
    centers_total = 0

    data_iter = None
    if int(args.data_num_workers) > 0:
        ds = RandomStateIterableDataset(
            map_pool=map_pool,
            min_agents=args.min_agents,
            max_agents=args.max_agents,
            min_tasks=args.min_tasks,
            max_tasks=args.max_tasks,
            clip_max=args.clip_max,
            delivering_fraction=args.delivering_fraction,
            base_seed=args.seed + (0 if train_mode else 99991),
        )
        loader = DataLoader(
            ds,
            batch_size=args.batch_size_graph,
            num_workers=int(args.data_num_workers),
            collate_fn=lambda x: x,
            prefetch_factor=max(2, int(args.prefetch_batches)),
            persistent_workers=False,
        )
        data_iter = iter(loader)

    for _ in range(num_batches):
        if data_iter is None:
            batch_states = []
            for _ in range(args.batch_size_graph):
                gm = random.choice(map_pool)
                st = generator.sample_state(grid_map=gm, rng=random)
                batch_states.append(st)
        else:
            batch_states = next(data_iter)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train_mode):
            out = gather_batch(
                encoder=encoder,
                ring_head=ring_head,
                batch_states=batch_states,
                dist_cache_by_name=dist_cache_by_name,
                dist_edges_dev_by_name=dist_edges_dev_by_name,
                rings=rings,
                args=args,
                device=device,
            )
            if out is None:
                continue
            loss = out["loss"]
            if train_mode and torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(ring_head.parameters()),
                    args.grad_clip,
                )
                optimizer.step()

        valid_batches += 1
        sum_loss += float(loss.item())
        sum_mae += float(out["mae_all"])
        sum_task += np.array(out["mae_task"], dtype=np.float64)
        sum_ring += np.array(out["mae_ring"], dtype=np.float64)
        centers_total += int(out["centers_total"])

    if valid_batches == 0:
        return {
            "ring_loss": float("nan"),
            "mae_all": float("nan"),
            "mae_task": [float("nan")] * 4,
            "mae_ring": [float("nan")] * len(rings),
            "centers_total": 0,
        }
    return {
        "ring_loss": sum_loss / valid_batches,
        "mae_all": sum_mae / valid_batches,
        "mae_task": (sum_task / valid_batches).tolist(),
        "mae_ring": (sum_ring / valid_batches).tolist(),
        "centers_total": centers_total,
    }


def save_checkpoint(path, encoder, ring_head, optimizer, epoch, metrics, args, rings):
    ckpt = {
        "encoder": encoder.state_dict(),
        "ring_head": ring_head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "rings": rings,
        "args": vars(args),
    }
    torch.save(ckpt, path)


def parse_args():
    parser = argparse.ArgumentParser("SP-MPNN ring+heat convergence pretraining")
    parser.add_argument("--map_path", type=str, default="/local-scratchg/yifan/2024/MAPD/MAPD_RL/code/mapf_solver/maps/Instances/small/kiva-small.map")
    parser.add_argument("--map_paths", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="./pretrain_spmpnn_logs")
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_distance", type=int, default=3)
    parser.add_argument("--use_adaptive_rings", action="store_true", default=True)
    parser.add_argument("--disable_adaptive_rings", action="store_true", default=False)

    parser.add_argument("--min_agents", type=int, default=10)
    parser.add_argument("--max_agents", type=int, default=50)
    parser.add_argument("--min_tasks", type=int, default=20)
    parser.add_argument("--max_tasks", type=int, default=100)
    parser.add_argument("--clip_max", type=float, default=4.0)
    parser.add_argument("--centers_per_graph", type=int, default=48)
    parser.add_argument("--delivering_fraction", type=float, default=0.15)
    parser.add_argument("--R_heat", type=int, default=3)
    parser.add_argument("--near_ring_weight", type=float, default=1.5)
    parser.add_argument("--near_weighted_rings", type=int, default=2)

    parser.add_argument("--variants_per_map", type=int, default=3)
    parser.add_argument("--obstacle_drop_prob_min", type=float, default=0.01)
    parser.add_argument("--obstacle_drop_prob_max", type=float, default=0.08)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size_graph", type=int, default=24)
    parser.add_argument("--train_graphs_per_epoch", type=int, default=256)
    parser.add_argument("--val_graphs_per_epoch", type=int, default=64)
    parser.add_argument("--data_num_workers", type=int, default=4)
    parser.add_argument("--prefetch_batches", type=int, default=4)
    parser.add_argument("--spcache_dir", type=str, default="./spcache")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--early_stop_patience", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    os.makedirs(args.save_dir, exist_ok=True)

    if args.map_paths.strip():
        map_paths = [p.strip() for p in args.map_paths.split(",") if p.strip()]
    else:
        map_paths = [args.map_path]
    if len(map_paths) == 0:
        raise ValueError("No map paths provided.")

    R = int(args.max_distance) * int(args.num_layers)
    use_adaptive = bool(args.use_adaptive_rings) and (not bool(args.disable_adaptive_rings))
    if use_adaptive:
        rings = build_adaptive_rings(R)
    else:
        rings = [(1, 2), (3, 4), (5, 6), (7, min(8, R))]
        rings = [r for r in rings if r[0] <= R]
        rings[-1] = (rings[-1][0], min(rings[-1][1], R))

    with open(os.path.join(args.save_dir, "config.json"), "w", encoding="utf-8") as f:
        cfg = dict(vars(args))
        cfg["effective_R"] = R
        cfg["rings"] = rings
        json.dump(cfg, f, ensure_ascii=True, indent=2)

    map_pool = build_map_pool(
        map_paths=map_paths,
        obstacle_drop_prob_min=args.obstacle_drop_prob_min,
        obstacle_drop_prob_max=args.obstacle_drop_prob_max,
        variants_per_map=args.variants_per_map,
    )

    dist_cache_by_name = {}
    dist_edges_dev_by_name = {}
    total_maps = len(map_pool)
    for idx, gm in enumerate(map_pool, start=1):
        print(f"[Precompute] ({idx}/{total_maps}) start: {gm.source_name}")
        cache = SPDistanceCache(
            grid_map=gm,
            max_distance=args.max_distance,
            max_hop=max(R, args.R_heat),
            rings=rings,
            cache_dir=args.spcache_dir,
            verbose=True,
        )
        dist_cache_by_name[gm.source_name] = cache
        dist_edges_dev_by_name[gm.source_name] = {k: v.to(device) for k, v in cache.distance_edges.items()}
        source = "cache" if cache.loaded_from_cache else "fresh"
        print(f"[Precompute] ({idx}/{total_maps}) done: {gm.source_name} ({source})")

    print(f"Map pool size={len(map_pool)}, base_maps={len(map_paths)}, R={R}, rings={rings}, R_heat={args.R_heat}")

    generator = RandomStateGenerator(
        grid_map=map_pool[0],
        min_agents=args.min_agents,
        max_agents=args.max_agents,
        min_tasks=args.min_tasks,
        max_tasks=args.max_tasks,
        clip_max=args.clip_max,
        delivering_fraction=args.delivering_fraction,
    )

    encoder = SPMPNNGridEncoder(
        input_dim=5,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        max_distance=args.max_distance,
        dropout=args.dropout,
    ).to(device)
    ring_head = RingRegressionHead(
        hidden_dim=args.hidden_dim,
        num_rings=len(rings),
        head_hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(ring_head.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_loss = float("inf")
    no_improve = 0
    metrics_path = os.path.join(args.save_dir, "metrics.jsonl")

    for epoch in range(1, args.epochs + 1):
        train_m = run_epoch(
            mode="train",
            encoder=encoder,
            ring_head=ring_head,
            generator=generator,
            map_pool=map_pool,
            dist_cache_by_name=dist_cache_by_name,
            dist_edges_dev_by_name=dist_edges_dev_by_name,
            optimizer=optimizer,
            rings=rings,
            args=args,
            device=device,
        )
        with torch.no_grad():
            val_m = run_epoch(
                mode="val",
                encoder=encoder,
                ring_head=ring_head,
                generator=generator,
                map_pool=map_pool,
                dist_cache_by_name=dist_cache_by_name,
                dist_edges_dev_by_name=dist_edges_dev_by_name,
                optimizer=optimizer,
                rings=rings,
                args=args,
                device=device,
            )

        row = {
            "epoch": epoch,
            "train_ring_loss": train_m["ring_loss"],
            "val_ring_loss": val_m["ring_loss"],
            "train_mae_all": train_m["mae_all"],
            "val_mae_all": val_m["mae_all"],
            "train_mae_task": train_m["mae_task"],
            "val_mae_task": val_m["mae_task"],
            "train_mae_ring": train_m["mae_ring"],
            "val_mae_ring": val_m["mae_ring"],
            "train_centers_total": train_m["centers_total"],
            "val_centers_total": val_m["centers_total"],
        }
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

        print(
            f"[Epoch {epoch:03d}] "
            f"train_ring_loss={train_m['ring_loss']:.4f} val_ring_loss={val_m['ring_loss']:.4f} "
            f"train_mae={train_m['mae_all']:.4f} val_mae={val_m['mae_all']:.4f} "
            f"train_task_mae={train_m['mae_task']} val_task_mae={val_m['mae_task']}"
        )

        save_checkpoint(
            os.path.join(args.save_dir, "last.pt"),
            encoder=encoder,
            ring_head=ring_head,
            optimizer=optimizer,
            epoch=epoch,
            metrics=row,
            args=args,
            rings=rings,
        )

        cur_val = val_m["ring_loss"]
        if np.isfinite(cur_val) and cur_val < best_val_loss:
            best_val_loss = cur_val
            no_improve = 0
            save_checkpoint(
                os.path.join(args.save_dir, "best_ring.pt"),
                encoder=encoder,
                ring_head=ring_head,
                optimizer=optimizer,
                epoch=epoch,
                metrics=row,
                args=args,
                rings=rings,
            )
        else:
            no_improve += 1

        if no_improve >= args.early_stop_patience:
            print(f"Early stop at epoch {epoch}, patience={args.early_stop_patience}")
            break

    print(f"Finished. Checkpoints/logs saved in: {args.save_dir}")


if __name__ == "__main__":
    main()
    raise SystemExit(0)
#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import random
import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

from model.spmpnn_pretrain import (
    ContextPredictionHead,
    OccupancySummaryHead,
    SPMPNNGridEncoder,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class GridMap:
    def __init__(self, map_path=None, blocked_mask=None, source_name=None):
        self.map_path = map_path
        self.source_name = source_name or map_path or "custom_map"
        self.height = 0
        self.width = 0
        self.grid_chars = []
        self.blocked_mask = None
        self.free_indices = []
        if blocked_mask is not None:
            self._init_from_mask(blocked_mask)
        else:
            self._read_map()

    def _init_from_mask(self, blocked_mask):
        blocked_mask = np.asarray(blocked_mask, dtype=np.bool_)
        if blocked_mask.ndim != 2:
            raise ValueError("blocked_mask must be a 2D array")
        self.height, self.width = blocked_mask.shape
        self.blocked_mask = blocked_mask
        self.grid_chars = []
        for r in range(self.height):
            row = []
            for c in range(self.width):
                row.append("@" if self.blocked_mask[r, c] else ".")
            self.grid_chars.append("".join(row))
        self._build_free_indices()

    def _build_free_indices(self):
        self.free_indices = []
        for r in range(self.height):
            for c in range(self.width):
                if not self.blocked_mask[r, c]:
                    self.free_indices.append(self.rc_to_idx(r, c))
        if len(self.free_indices) == 0:
            raise ValueError("No free cells found in map.")

    def _read_map(self):
        with open(self.map_path, "r", encoding="utf-8") as f:
            first = f.readline().strip()
            h, w = first.split(",")
            self.height, self.width = int(h), int(w)
            _ = f.readline()
            _ = f.readline()
            _ = f.readline()
            lines = [line.strip() for line in f if line.strip()]

        if len(lines) != self.height:
            raise ValueError(
                f"Map rows mismatch: expected {self.height}, got {len(lines)} from {self.map_path}"
            )
        for row in lines:
            if len(row) != self.width:
                raise ValueError(
                    f"Map cols mismatch: expected {self.width}, got {len(row)} in {self.map_path}"
                )

        self.grid_chars = lines
        blocked = np.zeros((self.height, self.width), dtype=np.bool_)
        for r in range(self.height):
            for c in range(self.width):
                blocked[r, c] = self.grid_chars[r][c] == "@"
        self.blocked_mask = blocked
        self._build_free_indices()

    def rc_to_idx(self, r, c):
        return r * self.width + c

    def idx_to_rc(self, idx):
        return idx // self.width, idx % self.width

    @property
    def num_nodes(self):
        return self.height * self.width


def build_map_pool(map_paths, obstacle_drop_prob_min, obstacle_drop_prob_max, variants_per_map):
    """
    Build pool of maps:
    - each base map itself
    - plus obstacle-drop variants
    """
    pool = []
    for mp in map_paths:
        base = GridMap(map_path=mp, source_name=os.path.basename(mp))
        pool.append(base)

        blocked_pos = np.argwhere(base.blocked_mask)
        if blocked_pos.shape[0] == 0 or variants_per_map <= 0:
            continue
        for vidx in range(int(variants_per_map)):
            p = random.uniform(float(obstacle_drop_prob_min), float(obstacle_drop_prob_max))
            mask = base.blocked_mask.copy()
            for rr, cc in blocked_pos:
                if random.random() < p:
                    mask[rr, cc] = False
            name = f"{os.path.basename(mp)}#drop{vidx}"
            pool.append(GridMap(blocked_mask=mask, source_name=name))
    return pool


class SPDistanceCache:
    """
    Precompute:
    - exact-distance directed edges for dist_1..dist_k
    - exact-distance node lists (for ring sampling)
    """

    def __init__(
        self,
        grid_map,
        max_distance=3,
        max_ring_distance=5,
        cache_dir=None,
        verbose=False,
    ):
        self.grid_map = grid_map
        self.max_distance = int(max_distance)
        self.max_ring_distance = int(max_ring_distance)
        self.cache_dir = cache_dir
        self.verbose = bool(verbose)
        self.adj = self._build_adj()
        self.distance_edges = {}
        self.exact_dist_nodes = {}  # src -> {d: [nodes]}
        self.loaded_from_cache = False
        self._precompute_or_load()

    def _build_adj(self):
        adj = [[] for _ in range(self.grid_map.num_nodes)]
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for r in range(self.grid_map.height):
            for c in range(self.grid_map.width):
                if self.grid_map.blocked_mask[r, c]:
                    continue
                u = self.grid_map.rc_to_idx(r, c)
                for dr, dc in dirs:
                    nr, nc = r + dr, c + dc
                    if (
                        0 <= nr < self.grid_map.height
                        and 0 <= nc < self.grid_map.width
                        and not self.grid_map.blocked_mask[nr, nc]
                    ):
                        v = self.grid_map.rc_to_idx(nr, nc)
                        adj[u].append(v)
        return adj

    def _cache_key(self):
        bm = np.ascontiguousarray(self.grid_map.blocked_mask.astype(np.uint8))
        digest = hashlib.md5(bm.tobytes()).hexdigest()
        return (
            f"spcache_{self.grid_map.source_name}_"
            f"h{self.grid_map.height}_w{self.grid_map.width}_"
            f"d{self.max_distance}_r{self.max_ring_distance}_{digest}.pt"
        )

    def _cache_path(self):
        if self.cache_dir is None:
            return None
        os.makedirs(self.cache_dir, exist_ok=True)
        return os.path.join(self.cache_dir, self._cache_key())

    def _precompute_or_load(self):
        cache_path = self._cache_path()
        if cache_path is not None and os.path.exists(cache_path):
            data = torch.load(cache_path, map_location="cpu")
            self.distance_edges = data["distance_edges"]
            self.exact_dist_nodes = data["exact_dist_nodes"]
            self.loaded_from_cache = True
            if self.verbose:
                print(f"[SPCache] loaded: {cache_path}")
            return

        t0 = time.time()
        self._precompute()
        if cache_path is not None:
            torch.save(
                {
                    "distance_edges": self.distance_edges,
                    "exact_dist_nodes": self.exact_dist_nodes,
                },
                cache_path,
            )
            if self.verbose:
                print(f"[SPCache] saved: {cache_path}")
        if self.verbose:
            dt = time.time() - t0
            print(f"[SPCache] computed {self.grid_map.source_name} in {dt:.2f}s")

    def _precompute(self):
        dist_edge_lists = {f"dist_{d}": [] for d in range(1, self.max_distance + 1)}
        max_hop = max(self.max_distance, self.max_ring_distance)

        for src in self.grid_map.free_indices:
            q = deque([src])
            visited = {src: 0}
            exact_nodes = {d: [] for d in range(1, max_hop + 1)}

            while q:
                cur = q.popleft()
                dcur = visited[cur]
                if dcur >= max_hop:
                    continue
                for nb in self.adj[cur]:
                    if nb in visited:
                        continue
                    nd = dcur + 1
                    visited[nb] = nd
                    q.append(nb)
                    exact_nodes[nd].append(nb)
                    if nd <= self.max_distance:
                        dist_edge_lists[f"dist_{nd}"].append([src, nb])

            self.exact_dist_nodes[src] = exact_nodes

        for d in range(1, self.max_distance + 1):
            key = f"dist_{d}"
            edges = dist_edge_lists[key]
            if len(edges) == 0:
                self.distance_edges[key] = torch.zeros((2, 0), dtype=torch.long)
            else:
                self.distance_edges[key] = torch.tensor(edges, dtype=torch.long).t().contiguous()


class RandomStateGenerator:
    def __init__(
        self,
        grid_map,
        min_agents=10,
        max_agents=50,
        min_tasks=20,
        max_tasks=100,
        clip_max=4,
    ):
        self.grid_map = grid_map
        self.min_agents = int(min_agents)
        self.max_agents = int(max_agents)
        self.min_tasks = int(min_tasks)
        self.max_tasks = int(max_tasks)
        self.clip_max = float(clip_max)

    def _sample_cells(self, free_indices, n, rng, replace=False):
        n_free = len(free_indices)
        if not replace and n <= n_free:
            return rng.sample(free_indices, n)
        return [rng.choice(free_indices) for _ in range(n)]

    def sample_state(self, grid_map=None, rng=None):
        if rng is None:
            rng = random
        gm = grid_map if grid_map is not None else self.grid_map
        num_agents = rng.randint(self.min_agents, self.max_agents)
        num_tasks = rng.randint(self.min_tasks, self.max_tasks)

        blocked_vec = np.zeros((gm.num_nodes,), dtype=np.float32)
        for idx in range(gm.num_nodes):
            r, c = gm.idx_to_rc(idx)
            blocked_vec[idx] = 1.0 if gm.blocked_mask[r, c] else 0.0

        agent_occ = np.zeros((gm.num_nodes,), dtype=np.float32)
        pickup_count = np.zeros((gm.num_nodes,), dtype=np.float32)
        delivery_count = np.zeros((gm.num_nodes,), dtype=np.float32)

        agent_cells = self._sample_cells(
            gm.free_indices,
            num_agents,
            rng=rng,
            replace=(num_agents > len(gm.free_indices)),
        )
        for cell in agent_cells:
            agent_occ[cell] += 1.0

        pickup_cells = []
        delivery_cells = []
        for _ in range(num_tasks):
            p = rng.choice(gm.free_indices)
            d = rng.choice(gm.free_indices)
            while d == p:
                d = rng.choice(gm.free_indices)
            pickup_cells.append(p)
            delivery_cells.append(d)
            pickup_count[p] += 1.0
            delivery_count[d] += 1.0

        pickup_norm = np.minimum(pickup_count, self.clip_max) / self.clip_max
        delivery_norm = np.minimum(delivery_count, self.clip_max) / self.clip_max

        node_features = np.stack(
            [
                blocked_vec,
                agent_occ,
                pickup_norm,
                delivery_norm,
            ],
            axis=-1,
        ).astype(np.float32)

        special_mask = (agent_occ > 0) | (pickup_count > 0) | (delivery_count > 0)
        special_cells = [idx for idx in gm.free_indices if special_mask[idx]]

        return {
            "node_features": node_features,
            "agent_occ_raw": agent_occ,
            "special_cells": special_cells,
            "free_cells": gm.free_indices,
            "grid_name": gm.source_name,
        }


def sample_centers(free_cells, special_cells, num_centers):
    target = int(num_centers)
    if target <= 0:
        return []
    half = target // 2

    chosen = []
    if len(special_cells) > 0:
        n_special = min(half, len(special_cells))
        chosen.extend(random.sample(special_cells, n_special))

    remain = target - len(chosen)
    if remain > 0:
        free_pool = [x for x in free_cells if x not in set(chosen)]
        if len(free_pool) >= remain:
            chosen.extend(random.sample(free_pool, remain))
        elif len(free_pool) > 0:
            chosen.extend(free_pool)
            still = target - len(chosen)
            if still > 0:
                chosen.extend(random.choices(free_cells, k=still))
        else:
            chosen.extend(random.choices(free_cells, k=remain))
    return chosen


def build_center_records(state, dist_cache, centers, r1, r2):
    records = []
    occ_raw = state["agent_occ_raw"]
    for center in centers:
        ring = []
        exact = dist_cache.exact_dist_nodes.get(center, {})
        for d in range(r1 + 1, r2 + 1):
            ring.extend(exact.get(d, []))
        if len(ring) == 0:
            continue
        occ_sum = float(np.sum(occ_raw[ring]))
        records.append(
            {
                "center": int(center),
                "ring_nodes": ring,
                "occ_sum": occ_sum,
            }
        )
    return records


def sample_graph_bundle(generator, map_pool, dist_cache_by_name):
    gm = random.choice(map_pool)
    st = generator.sample_state(grid_map=gm)
    cache = dist_cache_by_name[gm.source_name]
    return st, cache


class RandomStateIterableDataset(IterableDataset):
    """
    Infinite random state generator for multi-process prefetch.
    """

    def __init__(self, map_pool, min_agents, max_agents, min_tasks, max_tasks, clip_max, base_seed):
        super().__init__()
        self.map_pool = map_pool
        self.min_agents = int(min_agents)
        self.max_agents = int(max_agents)
        self.min_tasks = int(min_tasks)
        self.max_tasks = int(max_tasks)
        self.clip_max = float(clip_max)
        self.base_seed = int(base_seed)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        wid = 0 if worker_info is None else int(worker_info.id)
        seed = self.base_seed + wid * 100003 + int(time.time()) % 100000
        rng = random.Random(seed)

        generator = RandomStateGenerator(
            grid_map=self.map_pool[0],
            min_agents=self.min_agents,
            max_agents=self.max_agents,
            min_tasks=self.min_tasks,
            max_tasks=self.max_tasks,
            clip_max=self.clip_max,
        )
        while True:
            gm = rng.choice(self.map_pool)
            yield generator.sample_state(grid_map=gm, rng=rng)


def estimate_occ_threshold(generator, map_pool, dist_cache_by_name, num_graphs, num_centers, r1, r2):
    all_occ = []
    for _ in range(int(num_graphs)):
        st, dist_cache = sample_graph_bundle(generator, map_pool, dist_cache_by_name)
        centers = sample_centers(st["free_cells"], st["special_cells"], num_centers)
        recs = build_center_records(st, dist_cache, centers, r1, r2)
        all_occ.extend([x["occ_sum"] for x in recs])
    if len(all_occ) == 0:
        return 0.0
    return float(np.median(np.array(all_occ, dtype=np.float32)))


def gather_batch_samples(
    encoder,
    ctx_head,
    occ_head,
    batch_states,
    device,
    num_centers,
    neg_per_pos,
    r1,
    r2,
    occ_threshold,
    occ_task_mode,
    occ_num_bins,
    occ_max_count,
):
    all_ctx_logits = []
    all_ctx_labels = []
    all_occ_logits = []
    all_occ_labels = []
    skipped_empty_ring = 0
    total_centers = 0
    valid_centers = 0

    for st in batch_states:
        dist_edges_dev = st["dist_edges_dev"]
        dist_cache = st["dist_cache"]
        x = torch.from_numpy(st["node_features"]).to(device=device)
        h = encoder(x, dist_edges_dev)

        centers = sample_centers(st["free_cells"], st["special_cells"], num_centers)
        total_centers += len(centers)
        records = build_center_records(st, dist_cache, centers, r1, r2)
        skipped_empty_ring += max(0, len(centers) - len(records))
        if len(records) == 0:
            continue
        valid_centers += len(records)

        z_centers = []
        c_contexts = []
        occ_sums = []
        for rec in records:
            ring_nodes = torch.tensor(rec["ring_nodes"], dtype=torch.long, device=device)
            c = h[ring_nodes].mean(dim=0)
            z = h[rec["center"]]
            z_centers.append(z)
            c_contexts.append(c)
            occ_sums.append(rec["occ_sum"])

        z_centers = torch.stack(z_centers, dim=0)
        c_contexts = torch.stack(c_contexts, dim=0)

        # Context task: positive + negative pairs
        n_valid = z_centers.size(0)
        if n_valid >= 2:
            for i in range(n_valid):
                # Positive
                pos_logit = ctx_head(z_centers[i : i + 1], c_contexts[i : i + 1])
                all_ctx_logits.append(pos_logit)
                all_ctx_labels.append(torch.ones((1,), device=device))

                candidates = [j for j in range(n_valid) if j != i]
                for _ in range(int(neg_per_pos)):
                    j = random.choice(candidates)
                    neg_logit = ctx_head(z_centers[i : i + 1], c_contexts[j : j + 1])
                    all_ctx_logits.append(neg_logit)
                    all_ctx_labels.append(torch.zeros((1,), device=device))

        # Occupancy task
        if occ_task_mode == "binary":
            occ_target = torch.tensor(
                [1 if v > occ_threshold else 0 for v in occ_sums],
                dtype=torch.long,
                device=device,
            )
            occ_logits = occ_head(z_centers)
            all_occ_logits.append(occ_logits)
            all_occ_labels.append(occ_target)
        elif occ_task_mode == "count_bins":
            # Uniform bins over [0, occ_max_count]
            maxv = max(1.0, float(occ_max_count))
            b = max(2, int(occ_num_bins))
            targets = []
            for v in occ_sums:
                ratio = min(max(float(v) / maxv, 0.0), 0.999999)
                idx = int(ratio * b)
                if idx >= b:
                    idx = b - 1
                targets.append(idx)
            occ_target = torch.tensor(targets, dtype=torch.long, device=device)
            occ_logits = occ_head(z_centers)
            all_occ_logits.append(occ_logits)
            all_occ_labels.append(occ_target)
        elif occ_task_mode == "regression":
            # Predict normalized count in [0, 1]
            maxv = max(1.0, float(occ_max_count))
            target = torch.tensor(
                [min(max(float(v) / maxv, 0.0), 1.0) for v in occ_sums],
                dtype=torch.float32,
                device=device,
            )
            pred = occ_head(z_centers).squeeze(-1)
            all_occ_logits.append(pred)
            all_occ_labels.append(target)
        else:
            raise ValueError(f"Unsupported occ_task_mode: {occ_task_mode}")

    stats = {
        "total_centers": total_centers,
        "valid_centers": valid_centers,
        "skipped_empty_ring": skipped_empty_ring,
    }
    return all_ctx_logits, all_ctx_labels, all_occ_logits, all_occ_labels, stats


def run_epoch(
    mode,
    encoder,
    ctx_head,
    occ_head,
    generator,
    map_pool,
    dist_cache_by_name,
    dist_edges_dev_by_name,
    optimizer,
    device,
    args,
    occ_threshold,
    occ_max_count,
):
    train_mode = mode == "train"
    encoder.train(train_mode)
    ctx_head.train(train_mode)
    occ_head.train(train_mode)

    graphs_per_epoch = args.train_graphs_per_epoch if train_mode else args.val_graphs_per_epoch
    num_batches = max(1, int(np.ceil(graphs_per_epoch / args.batch_size_graph)))

    sum_ctx_loss = 0.0
    sum_occ_loss = 0.0
    sum_ctx_acc = 0.0
    sum_occ_acc = 0.0
    ctx_cnt = 0
    occ_cnt = 0
    centers_total = 0
    centers_valid = 0
    centers_skipped = 0

    bce = torch.nn.BCEWithLogitsLoss()
    ce = torch.nn.CrossEntropyLoss()
    huber = torch.nn.SmoothL1Loss()

    data_iter = None
    if int(args.data_num_workers) > 0:
        ds = RandomStateIterableDataset(
            map_pool=map_pool,
            min_agents=args.min_agents,
            max_agents=args.max_agents,
            min_tasks=args.min_tasks,
            max_tasks=args.max_tasks,
            clip_max=args.clip_max,
            base_seed=args.seed + (0 if train_mode else 99991),
        )
        loader = DataLoader(
            ds,
            batch_size=args.batch_size_graph,
            num_workers=int(args.data_num_workers),
            collate_fn=lambda x: x,
            prefetch_factor=max(2, int(args.prefetch_batches)),
            persistent_workers=False,
        )
        data_iter = iter(loader)

    for _ in range(num_batches):
        if data_iter is None:
            batch_states = []
            for _ in range(args.batch_size_graph):
                st, cache = sample_graph_bundle(generator, map_pool, dist_cache_by_name)
                st["dist_cache"] = cache
                st["dist_edges_dev"] = dist_edges_dev_by_name[st["grid_name"]]
                batch_states.append(st)
        else:
            batch_states = next(data_iter)
            for st in batch_states:
                cache = dist_cache_by_name[st["grid_name"]]
                st["dist_cache"] = cache
                st["dist_edges_dev"] = dist_edges_dev_by_name[st["grid_name"]]

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train_mode):
            ctx_logits_list, ctx_labels_list, occ_logits_list, occ_labels_list, st = gather_batch_samples(
                encoder=encoder,
                ctx_head=ctx_head,
                occ_head=occ_head,
                batch_states=batch_states,
                device=device,
                num_centers=args.centers_per_graph,
                neg_per_pos=args.neg_per_pos,
                r1=args.r1,
                r2=args.r2,
                occ_threshold=occ_threshold,
                occ_task_mode=args.occ_task_mode,
                occ_num_bins=args.occ_num_bins,
                occ_max_count=occ_max_count,
            )
            centers_total += st["total_centers"]
            centers_valid += st["valid_centers"]
            centers_skipped += st["skipped_empty_ring"]

            if len(ctx_logits_list) == 0 and len(occ_logits_list) == 0:
                continue

            loss = torch.tensor(0.0, device=device)
            if len(ctx_logits_list) > 0:
                ctx_logits = torch.cat(ctx_logits_list, dim=0)
                ctx_labels = torch.cat(ctx_labels_list, dim=0)
                ctx_loss = bce(ctx_logits, ctx_labels)
                loss = loss + ctx_loss

                with torch.no_grad():
                    ctx_pred = (torch.sigmoid(ctx_logits) >= 0.5).float()
                    ctx_acc = (ctx_pred == ctx_labels).float().mean()
                n_ctx = int(ctx_labels.numel())
                sum_ctx_loss += float(ctx_loss.item()) * n_ctx
                sum_ctx_acc += float(ctx_acc.item()) * n_ctx
                ctx_cnt += n_ctx
            else:
                ctx_loss = None

            if len(occ_logits_list) > 0:
                occ_logits = torch.cat(occ_logits_list, dim=0)
                occ_labels = torch.cat(occ_labels_list, dim=0)
                if args.occ_task_mode in ("binary", "count_bins"):
                    occ_loss = ce(occ_logits, occ_labels)
                    with torch.no_grad():
                        occ_pred = torch.argmax(occ_logits, dim=-1)
                        occ_acc = (occ_pred == occ_labels).float().mean()
                else:
                    occ_loss = huber(occ_logits, occ_labels)
                    with torch.no_grad():
                        mae = torch.abs(occ_logits - occ_labels).mean()
                        occ_acc = 1.0 - mae

                loss = loss + args.lambda_occ * occ_loss
                n_occ = int(occ_labels.numel())
                sum_occ_loss += float(occ_loss.item()) * n_occ
                sum_occ_acc += float(occ_acc.item()) * n_occ
                occ_cnt += n_occ
            else:
                occ_loss = None

            if train_mode and torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(ctx_head.parameters()) + list(occ_head.parameters()),
                    args.grad_clip,
                )
                optimizer.step()

    out = {
        "ctx_loss": (sum_ctx_loss / ctx_cnt) if ctx_cnt > 0 else float("nan"),
        "occ_loss": (sum_occ_loss / occ_cnt) if occ_cnt > 0 else float("nan"),
        "ctx_acc": (sum_ctx_acc / ctx_cnt) if ctx_cnt > 0 else float("nan"),
        "occ_acc": (sum_occ_acc / occ_cnt) if occ_cnt > 0 else float("nan"),
        "ctx_samples": ctx_cnt,
        "occ_samples": occ_cnt,
        "centers_total": centers_total,
        "centers_valid": centers_valid,
        "centers_skipped": centers_skipped,
    }
    return out


def parse_args():
    parser = argparse.ArgumentParser("Grid-only SP-MPNN pretraining")
    parser.add_argument(
        "--map_path",
        type=str,
        default="/local-scratchg/yifan/2024/MAPD/MAPD_RL/code/mapf_solver/maps/Instances/small/kiva-small.map",
    )
    parser.add_argument(
        "--map_paths",
        type=str,
        default="",
        help="Comma-separated map paths. If provided, overrides --map_path.",
    )
    parser.add_argument("--save_dir", type=str, default="./pretrain_spmpnn_logs")
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_distance", type=int, default=3)

    parser.add_argument("--min_agents", type=int, default=10)
    parser.add_argument("--max_agents", type=int, default=50)
    parser.add_argument("--min_tasks", type=int, default=20)
    parser.add_argument("--max_tasks", type=int, default=100)
    parser.add_argument("--clip_max", type=float, default=4.0)

    parser.add_argument("--centers_per_graph", type=int, default=48)
    parser.add_argument("--neg_per_pos", type=int, default=3)
    parser.add_argument("--k_hop", type=int, default=3)
    parser.add_argument("--r1", type=int, default=2)
    parser.add_argument("--r2", type=int, default=5)
    parser.add_argument(
        "--use_effective_hops",
        action="store_true",
        default=True,
        help="Use effective_hops=max_distance*num_layers to set neighborhood range.",
    )
    parser.add_argument(
        "--disable_effective_hops",
        action="store_true",
        default=False,
        help="Disable effective hop scaling and use explicit r1/r2.",
    )
    parser.add_argument(
        "--effective_r1_ratio",
        type=float,
        default=0.5,
        help="r1 = floor(effective_hops * ratio) when --use_effective_hops is enabled.",
    )
    parser.add_argument("--lambda_occ", type=float, default=0.2)
    parser.add_argument(
        "--occ_task_mode",
        type=str,
        default="count_bins",
        choices=["binary", "count_bins", "regression"],
        help="Aux task mode: binary(low/high), count_bins, regression.",
    )
    parser.add_argument("--occ_num_bins", type=int, default=6)

    parser.add_argument("--variants_per_map", type=int, default=3)
    parser.add_argument("--obstacle_drop_prob_min", type=float, default=0.01)
    parser.add_argument("--obstacle_drop_prob_max", type=float, default=0.08)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size_graph", type=int, default=24)
    parser.add_argument("--train_graphs_per_epoch", type=int, default=256)
    parser.add_argument("--val_graphs_per_epoch", type=int, default=64)
    parser.add_argument("--occ_threshold_graphs", type=int, default=128)
    parser.add_argument("--data_num_workers", type=int, default=4)
    parser.add_argument("--prefetch_batches", type=int, default=4)
    parser.add_argument(
        "--spcache_dir",
        type=str,
        default="./spcache",
        help="Directory to store/load precomputed SP distance caches.",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--early_stop_patience", type=int, default=8)
    return parser.parse_args()


def save_checkpoint(path, encoder, ctx_head, occ_head, optimizer, epoch, metrics, args):
    ckpt = {
        "encoder": encoder.state_dict(),
        "ctx_head": ctx_head.state_dict(),
        "occ_head": occ_head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "args": vars(args),
    }
    torch.save(ckpt, path)


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=True, indent=2)

    if args.map_paths.strip():
        map_paths = [p.strip() for p in args.map_paths.split(",") if p.strip()]
    else:
        map_paths = [args.map_path]
    if len(map_paths) == 0:
        raise ValueError("No map paths provided.")

    use_effective_hops = bool(args.use_effective_hops) and (not bool(args.disable_effective_hops))
    if use_effective_hops:
        effective_hops = int(args.max_distance) * int(args.num_layers)
        args.k_hop = effective_hops
        args.r2 = effective_hops
        args.r1 = max(0, int(np.floor(effective_hops * float(args.effective_r1_ratio))))
        if args.r1 >= args.r2:
            args.r1 = max(0, args.r2 - 1)

    map_pool = build_map_pool(
        map_paths=map_paths,
        obstacle_drop_prob_min=args.obstacle_drop_prob_min,
        obstacle_drop_prob_max=args.obstacle_drop_prob_max,
        variants_per_map=args.variants_per_map,
    )

    dist_cache_by_name = {}
    dist_edges_dev_by_name = {}
    total_maps = len(map_pool)
    for idx, gm in enumerate(map_pool, start=1):
        print(f"[Precompute] ({idx}/{total_maps}) start: {gm.source_name}")
        cache = SPDistanceCache(
            grid_map=gm,
            max_distance=args.max_distance,
            max_ring_distance=max(args.r2, args.k_hop),
            cache_dir=args.spcache_dir,
            verbose=True,
        )
        dist_cache_by_name[gm.source_name] = cache
        dist_edges_dev_by_name[gm.source_name] = {k: v.to(device) for k, v in cache.distance_edges.items()}
        source = "cache" if cache.loaded_from_cache else "fresh"
        print(f"[Precompute] ({idx}/{total_maps}) done: {gm.source_name} ({source})")

    # global normalization range for count-like occupancy targets
    occ_max_count = float(max(1, args.max_agents))
    print(
        f"Map pool size={len(map_pool)}, base_maps={len(map_paths)}, "
        f"effective_hops={args.k_hop}, r1={args.r1}, r2={args.r2}, occ_mode={args.occ_task_mode}"
    )

    generator = RandomStateGenerator(
        grid_map=map_pool[0],
        min_agents=args.min_agents,
        max_agents=args.max_agents,
        min_tasks=args.min_tasks,
        max_tasks=args.max_tasks,
        clip_max=args.clip_max,
    )

    encoder = SPMPNNGridEncoder(
        input_dim=4,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        max_distance=args.max_distance,
        dropout=args.dropout,
    ).to(device)
    ctx_head = ContextPredictionHead(hidden_dim=args.hidden_dim).to(device)
    if args.occ_task_mode == "binary":
        occ_out_dim = 2
    elif args.occ_task_mode == "count_bins":
        occ_out_dim = max(2, int(args.occ_num_bins))
    else:
        occ_out_dim = 1
    occ_head = OccupancySummaryHead(
        hidden_dim=args.hidden_dim,
        occ_hidden_dim=32,
        output_dim=occ_out_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(ctx_head.parameters()) + list(occ_head.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_ctx = float("inf")
    no_improve = 0
    metrics_path = os.path.join(args.save_dir, "metrics.jsonl")

    for epoch in range(1, args.epochs + 1):
        occ_threshold = estimate_occ_threshold(
            generator=generator,
            map_pool=map_pool,
            dist_cache_by_name=dist_cache_by_name,
            num_graphs=args.occ_threshold_graphs,
            num_centers=args.centers_per_graph,
            r1=args.r1,
            r2=args.r2,
        )

        train_m = run_epoch(
            mode="train",
            encoder=encoder,
            ctx_head=ctx_head,
            occ_head=occ_head,
            generator=generator,
            map_pool=map_pool,
            dist_cache_by_name=dist_cache_by_name,
            dist_edges_dev_by_name=dist_edges_dev_by_name,
            optimizer=optimizer,
            device=device,
            args=args,
            occ_threshold=occ_threshold,
            occ_max_count=occ_max_count,
        )
        with torch.no_grad():
            val_m = run_epoch(
                mode="val",
                encoder=encoder,
                ctx_head=ctx_head,
                occ_head=occ_head,
                generator=generator,
                map_pool=map_pool,
                dist_cache_by_name=dist_cache_by_name,
                dist_edges_dev_by_name=dist_edges_dev_by_name,
                optimizer=optimizer,
                device=device,
                args=args,
                occ_threshold=occ_threshold,
                occ_max_count=occ_max_count,
            )

        row = {
            "epoch": epoch,
            "occ_threshold": occ_threshold,
            "train_ctx_loss": train_m["ctx_loss"],
            "train_occ_loss": train_m["occ_loss"],
            "train_ctx_acc": train_m["ctx_acc"],
            "train_occ_acc": train_m["occ_acc"],
            "val_ctx_loss": val_m["ctx_loss"],
            "val_occ_loss": val_m["occ_loss"],
            "val_ctx_acc": val_m["ctx_acc"],
            "val_occ_acc": val_m["occ_acc"],
            "train_centers_total": train_m["centers_total"],
            "train_centers_valid": train_m["centers_valid"],
            "train_centers_skipped": train_m["centers_skipped"],
            "val_centers_total": val_m["centers_total"],
            "val_centers_valid": val_m["centers_valid"],
            "val_centers_skipped": val_m["centers_skipped"],
        }
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

        print(
            f"[Epoch {epoch:03d}] "
            f"train_ctx_loss={train_m['ctx_loss']:.4f} train_ctx_acc={train_m['ctx_acc']:.4f} "
            f"train_occ_loss={train_m['occ_loss']:.4f} train_occ_acc={train_m['occ_acc']:.4f} | "
            f"val_ctx_loss={val_m['ctx_loss']:.4f} val_ctx_acc={val_m['ctx_acc']:.4f} "
            f"val_occ_loss={val_m['occ_loss']:.4f} val_occ_acc={val_m['occ_acc']:.4f} "
            f"occ_thr={occ_threshold:.3f}"
        )

        save_checkpoint(
            os.path.join(args.save_dir, "last.pt"),
            encoder,
            ctx_head,
            occ_head,
            optimizer,
            epoch,
            row,
            args,
        )

        cur_val_ctx = val_m["ctx_loss"]
        if np.isfinite(cur_val_ctx) and cur_val_ctx < best_val_ctx:
            best_val_ctx = cur_val_ctx
            no_improve = 0
            save_checkpoint(
                os.path.join(args.save_dir, "best_ctx.pt"),
                encoder,
                ctx_head,
                occ_head,
                optimizer,
                epoch,
                row,
                args,
            )
        else:
            no_improve += 1

        if no_improve >= args.early_stop_patience:
            print(f"Early stop at epoch {epoch}, patience={args.early_stop_patience}")
            break

    print(f"Finished. Checkpoints/logs saved in: {args.save_dir}")


if __name__ == "__main__":
    main()
