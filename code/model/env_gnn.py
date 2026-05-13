import gymnasium as gym
import numpy as np
from mapf_solver.mapf_solver import PBSSolver, Agent, Task, AgentTaskStatus
import random
from scipy.optimize import linear_sum_assignment
import sys
import os
import json
import time
from collections import deque
import torch
from torch_geometric.utils import k_hop_subgraph

class MultiAgentPickupEnv(gym.Env):
    def __init__(self, training=True, grid_path=None, seed=40, 
            solver="PBS", agent_num_lower_bound=10, agent_num_higher_bound=50, eval_data_path=None, task_num=500, pos_reward=False,
            sp_mpnn_max_distance=3, debug_env=False, debug_every=50, nearest_tasks_min_k=100, model_only_eval=False,
            task_truncated_size=1, obs_candidate_task_k=None, eval_max_tasks=None, eval_simulation_time=5000,
            eval_pending_task_cap=None,
            compute_cnn_distance_maps=True):
        super().__init__()
        self.training = training
        self.solver_name = solver
        self.step_count = 0
        self.seed = seed
        self.agent_num = (agent_num_lower_bound, agent_num_higher_bound)
        self.task_num = task_num
        self.pos_reward = pos_reward
        self.sp_mpnn_max_distance = sp_mpnn_max_distance  # 新增参数
        self.nearest_tasks_min_k = max(1, int(nearest_tasks_min_k))
        self.task_truncated_size = max(1, int(task_truncated_size))
        # 基础K（传给solver），solver侧再执行 max(num_agents, K) 逻辑
        self.candidate_task_k = min(self.nearest_tasks_min_k, self.task_num)
        # 观察/构图容量K：默认容纳 max(num_agents, K)，也允许外部显式锁定（兼容旧checkpoint测试）
        if obs_candidate_task_k is None:
            self.obs_candidate_task_k = min(max(self.candidate_task_k, self.agent_num[1]), self.task_num)
        else:
            self.obs_candidate_task_k = min(max(1, int(obs_candidate_task_k)), self.task_num)
        self.model_only_eval = bool(model_only_eval)
        self.eval_max_tasks = None if eval_max_tasks is None else max(1, int(eval_max_tasks))
        self.eval_simulation_time = max(1, int(eval_simulation_time))
        self.eval_pending_task_cap = None if eval_pending_task_cap is None else max(1, int(eval_pending_task_cap))
        self.compute_cnn_distance_maps = bool(compute_cnn_distance_maps)
        # 调试开关与频率
        self.debug_env = debug_env
        self.debug_every = debug_every
        self.grid_path = grid_path

        self.read_grid(self.grid_path)

        self.observation_space = gym.spaces.Dict({
            "env_id": gym.spaces.Discrete(1),
            "free_agents": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(agent_num_higher_bound, 3), dtype=np.float32
            ),
            "delivering_agents": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(agent_num_higher_bound, 3), dtype=np.float32
            ),
            "free_tasks": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(task_num, 4), dtype=np.float32
            ),
            "delivering_tasks": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(task_num, 5), dtype=np.float32
            ),
            "free_agents_num": gym.spaces.Discrete(agent_num_higher_bound),
            "delivering_agents_num": gym.spaces.Discrete(agent_num_higher_bound),
            "free_tasks_num": gym.spaces.Discrete(task_num),
            "delivering_tasks_num": gym.spaces.Discrete(task_num),
            "expert_actions": gym.spaces.Box(
                low=-1, high=np.inf, 
                shape=(agent_num_higher_bound+1, ),
                dtype=np.int32
            ),
            "grid": gym.spaces.Box(low=-1, high=0, shape=(self.grid_size[0], self.grid_size[1]), dtype=np.float32),
            "free_agents_nearest_tasks": gym.spaces.Box(
                low=-1, high=np.inf, shape=(agent_num_higher_bound, self.obs_candidate_task_k, 3), dtype=np.float32
            ),
            "assignable_agent_is_delivering": gym.spaces.Box(
                low=0, high=1, shape=(agent_num_higher_bound, 1), dtype=np.float32
            ),
            "assignable_agent_delivering_task_idx": gym.spaces.Box(
                low=-1, high=task_num, shape=(agent_num_higher_bound, 1), dtype=np.float32
            ),
            "delivering_tasks_nearest_tasks": gym.spaces.Box(
                low=-1, high=np.inf, shape=(task_num, self.obs_candidate_task_k, 3), dtype=np.float32
            ),
            "assignable_agent_a2a": gym.spaces.Box(
                low=-1, high=np.inf, shape=(agent_num_higher_bound, agent_num_higher_bound), dtype=np.float32
            ),
            "free_task_p2d": gym.spaces.Box(
                low=-1, high=np.inf, shape=(task_num, 1), dtype=np.float32
            ),
            # CNN channel maps
            "pickup_distances": gym.spaces.Box(low=0, high=np.inf, shape=(self.grid_size[0], self.grid_size[1]), dtype=np.float32),
            "delivery_distances": gym.spaces.Box(low=0, high=np.inf, shape=(self.grid_size[0], self.grid_size[1]), dtype=np.float32),
            "obstacle_map": gym.spaces.Box(low=0, high=1, shape=(self.grid_size[0], self.grid_size[1]), dtype=np.float32),
            "free_agent_map": gym.spaces.Box(low=0, high=1, shape=(self.grid_size[0], self.grid_size[1]), dtype=np.float32),
            "delivering_agent_map": gym.spaces.Box(low=0, high=1, shape=(self.grid_size[0], self.grid_size[1]), dtype=np.float32),
            "delivering_task_id_map": gym.spaces.Box(low=0, high=1, shape=(self.grid_size[0], self.grid_size[1]), dtype=np.float32),
            "pickup_location_map": gym.spaces.Box(low=0, high=1, shape=(self.grid_size[0], self.grid_size[1]), dtype=np.float32),
            "delivery_location_map": gym.spaces.Box(low=0, high=1, shape=(self.grid_size[0], self.grid_size[1]), dtype=np.float32),
        })
        
        # 新增：预计算的SP-MPNN k-hop距离信息（展平到顶层，避免嵌套Dict）
        # 为每个距离层级添加单独的观察空间条目
        # 关键优化：不再用 O((HW)^2) 的极端上界，而用 O(HW) 的线性上界。
        # 对4连通栅格图，d-hop邻域边数近似与 d*HW 成正比。
        num_grid_nodes = int(self.grid_size[0] * self.grid_size[1])
        self.sp_mpnn_max_edges_by_dist = {}
        for dist in range(1, sp_mpnn_max_distance + 1):
            # 采用保守线性上界（有向边）：16 * d * HW
            # 相比旧版 HW*HW，显著降低观测与前向内存占用。
            max_edges = int(min(num_grid_nodes * num_grid_nodes, max(1, 16 * dist * num_grid_nodes)))
            self.sp_mpnn_max_edges_by_dist[dist] = max_edges
            self.observation_space.spaces[f"sp_mpnn_dist_{dist}"] = gym.spaces.Box(
                low=0, high=self.grid_size[0] * self.grid_size[1], 
                shape=(2, max_edges), 
                dtype=np.int32
            )

        self.task_id_map = {}
        self.free_agent_id_map = {}
        self.delivering_agent_id_map = {}
        self.agent_task_pair = {}
        self.action_space = gym.spaces.MultiDiscrete([task_num] * agent_num_higher_bound)
        self.eval_data_path = eval_data_path
        self.last_task_id = []
        self.last_total_finish_time = 0
        self.agent_num_now = 0
        self.last_finish_time = 0
        self.episode = 0
        self.storage_ready = False
        self.storage_snapshot = None
        self.heuristics = {}
        self._distance_missing = -(self.grid_size[0] + self.grid_size[1])
        self._bfs_dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        if hasattr(self, 'sp_mpnn_max_distance') and self.sp_mpnn_max_distance > 0:
            self.precompute_sp_mpnn_distances()
        else:
            self.sp_mpnn_distance_edges = {}
        

    def _dbg(self, msg, force=False):
        try:
            if not getattr(self, 'debug_env', False):
                return
            every = int(getattr(self, 'debug_every', 50))
            if force or (self.step_count % max(1, every) == 0):
                print(f"[env {self.seed}] {msg}")
                sys.stdout.flush()
        except Exception:
            pass
    
    def read_grid(self, grid_path):
        with open(grid_path, 'r') as f:
            self.grid_size = tuple(map(int, f.readline().strip().split(',')))
            self.num_e = int(f.readline())
            self.num_r = int(f.readline())
            f.readline()  
            self.grid = [line.strip() for line in f]

            grid_np = np.zeros((self.grid_size[0], self.grid_size[1]), dtype=np.float32)
        
            self.e_map, self.r_map = {}, {}
            for x, line in enumerate(self.grid):
                for y, char in enumerate(line):
                    if char == 'e':
                        self.e_map[len(self.e_map)] = (x * self.grid_size[0] + y, x, y)
                    elif char == 'r':
                        self.r_map[len(self.r_map)] = (x * self.grid_size[0] + y, x, y)
                    elif char == '@':
                        grid_np[x, y] = -1
            
            self.grid = grid_np

        args = [
            "--map", grid_path,          
            "--agentNum", str(self.num_r),                      
            "--seed", str(self.seed),               
            "--solver",  self.solver_name,
            "--task_truncated_size", str(self.task_truncated_size),
            "--candidate_task_k", str(self.candidate_task_k),
            "--infer_use_expert_fallback", "false" if self.model_only_eval else "true",
        ]
        if self.eval_pending_task_cap is not None:
            args += ["--pending_task_cap", str(self.eval_pending_task_cap)]
        self.solver = PBSSolver(args)

    def cal_heuristics(self):
        # 兼容旧调用：改为按需缓存，不再全图预计算
        if not isinstance(self.heuristics, dict):
            self.heuristics = {}

    def bfs(self, start, directions=None):
        if directions is None:
            directions = self._bfs_dirs
        distances = np.full(self.grid_size, self._distance_missing, dtype=np.int32)
        queue = deque([start])
        distances[start] = 0

        while queue:
            current = queue.popleft()
            current_distance = distances[current]

            for direction in directions:
                neighbor = (current[0] + direction[0], current[1] + direction[1])

                if (0 <= neighbor[0] < self.grid_size[0] and
                    0 <= neighbor[1] < self.grid_size[1] and
                    self.grid[neighbor] == 0 and
                    distances[neighbor] == self._distance_missing):

                    distances[neighbor] = current_distance + 1
                    queue.append(neighbor)

        return distances

    def _get_dist_map(self, src_loc):
        key = (int(src_loc[0]), int(src_loc[1]))
        if key in self.heuristics:
            return self.heuristics[key]
        if not (0 <= key[0] < self.grid_size[0] and 0 <= key[1] < self.grid_size[1]):
            return None
        if self.grid[key] != 0:
            return None
        dist_map = self.bfs(key)
        self.heuristics[key] = dist_map
        return dist_map

    def _xy_to_loc_id(self, xy):
        x, y = int(xy[0]), int(xy[1])
        return x * self.grid_size[1] + y

    def _get_distance(self, src_loc, dst_loc):
        src = (int(src_loc[0]), int(src_loc[1]))
        dst = (int(dst_loc[0]), int(dst_loc[1]))
        if not (0 <= src[0] < self.grid_size[0] and 0 <= src[1] < self.grid_size[1]):
            return self._distance_missing
        if not (0 <= dst[0] < self.grid_size[0] and 0 <= dst[1] < self.grid_size[1]):
            return self._distance_missing
        # 优先走C++ solver中的heuristics缓存，避免Python重复BFS。
        try:
            if hasattr(self, "solver") and hasattr(self.solver, "get_distance"):
                d = int(self.solver.get_distance(self._xy_to_loc_id(src), self._xy_to_loc_id(dst)))
                if d >= 0:
                    return d
        except Exception:
            pass
        dist_map = self._get_dist_map(src)
        if dist_map is None:
            return self._distance_missing
        return int(dist_map[dst])
    
    def generate_agents_tasks(self):
        agent_num = random.randint(self.agent_num[0]//max(1, int(5-self.episode//2)), 
            self.agent_num[1]//max(1, int(5-self.episode//2)))
        # agent_num = random.randint(self.agent_num[0], self.agent_num[1]-1)
        self.agent_num_now = agent_num
        task_frequencies = [0.2, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        task_frequency = random.choice(task_frequencies)

        task_num_episode = self.task_num / 10 * min(10, self.episode+1)
        
        if task_frequency < 1:
            task_release_time = int(1/task_frequency)
        else:
            task_release_time = 1

        agents = []
        endpoint_num = len(self.e_map)
        while (len(agents) < agent_num):
            ra = random.randint(0, endpoint_num-1)
            if ra not in agents:
                agents.append(ra)

        tasks = []

        endpoint_num = len(self.e_map) - agent_num

        while len(tasks) < task_num_episode:
            pickup = random.randint(0, endpoint_num-1)
            delivery = random.randint(0, endpoint_num-1)
            while delivery == pickup:
                delivery = random.randint(0, endpoint_num-1)
            if task_release_time > 1:
                release_time = task_release_time * len(tasks)
            else:
                release_time = int(len(tasks)/task_frequency)
            tasks.append([release_time, pickup, delivery])

        return agents, tasks, task_frequency, task_release_time
            

    def loc(self, pos):
        return [int(pos/self.grid_size[1]), pos % self.grid_size[1]]
    
    def _capture_storage_snapshot(self):
        self.storage_snapshot = {
            # "state": self.state,
            "free_agent_id_map": dict(self.free_agent_id_map),
            "delivering_agent_id_map": dict(self.delivering_agent_id_map),
            "free_task_id_map": dict(self.free_task_id_map),
            "agent_task_pair": dict(self.agent_task_pair),
            "last_free_agent_num": int(getattr(self, "last_free_agent_num", 0) or 0),
            "last_free_task_num": int(getattr(self, "last_free_task_num", 0) or 0),
            "expert_estimated_finish_time": int(getattr(self, "expert_estimated_finish_time", 0) or 0),
            "task_truncated_size": int(getattr(self, "task_truncated_size", 1) or 1),
        }
        self.storage_ready = True

    def build_state(self, status):
        if getattr(status, "time_limit_reached", False) and status.valid == True:
            done = True
            finish_time = int(getattr(status, "estimated_finish_time", 0) or 0)
            service_time = int(getattr(status, "estimated_service_time", 0) or 0)
            if finish_time <= 0:
                finish_time = int(getattr(status, "finished_flowtime", 0) or 0)
            if service_time <= 0:
                service_time = int(getattr(status, "finished_service_time", 0) or 0)
            return None, finish_time, service_time, done, True

        if status.allFinished == 1 and status.valid == True:
            done = True
            return None, status.finished_flowtime, status.estimated_service_time, done, True
        else:
            done = False

        if status.valid == False:
            done = True
            penalty = self.agent_num_now*(self.grid_size[0]+self.grid_size[1])*2
            return None, penalty, penalty, done, False

        self.task_id_map.clear()
        self.free_agent_id_map.clear()
        self.delivering_agent_id_map.clear()
        self.agent_task_pair.clear()
        self.free_task_id_map.clear()
        self.delivering_task_id_map.clear()

        timestep = status.timestep
        agent_task_pair = status.agent_task_pair
        self.agent_task_pair = agent_task_pair

        delivering_agents = np.zeros((self.agent_num[1], 3), dtype=np.float32)
        free_agents = np.zeros((self.agent_num[1], 3), dtype=np.float32)
        assignable_agent_is_delivering = np.zeros((self.agent_num[1], 1), dtype=np.float32)
        assignable_agent_delivering_task_idx = np.full((self.agent_num[1], 1), -1, dtype=np.float32)
        free_tasks = np.zeros((self.task_num, 4), dtype=np.float32)
        delivering_tasks = np.zeros((self.task_num, 5), dtype=np.float32)
        delivering_tasks_nearest_tasks = np.full((self.task_num, self.obs_candidate_task_k, 3), -1, dtype=np.float32)
        assignable_agent_a2a = np.full((self.agent_num[1], self.agent_num[1]), -1, dtype=np.float32)
        free_task_p2d = np.full((self.task_num, 1), -1, dtype=np.float32)
        
        free_agent_cnt = 0
        delivering_agent_cnt = 0

        delivering_task_agent_map = {}
        all_free_agent_ids = []
        all_delivering_agent_ids = []

        for i in range(len(status.agents_all)):
            # full_loaded = status.agents_all[i].full_loaded
            is_delivering = status.agents_all[i].is_delivering
            location = self.loc(status.agents_all[i].start_location)
            start_timestep = status.agents_all[i].start_timestep
            if is_delivering:
                delivering_agents[delivering_agent_cnt] = np.array(location + [start_timestep], dtype=np.float32)
                self.delivering_agent_id_map[delivering_agent_cnt] = i
                delivering_agent_cnt += 1
                all_delivering_agent_ids.append(i)
            else:
                all_free_agent_ids.append(i)

        # truncated_size=1: 仅free agents可分配
        # truncated_size>1: free + delivering 都可作为可分配agent
        if self.task_truncated_size > 1:
            assignable_agent_ids = all_free_agent_ids + all_delivering_agent_ids
        else:
            assignable_agent_ids = all_free_agent_ids

        for local_idx, global_id in enumerate(assignable_agent_ids):
            ag = status.agents_all[global_id]
            location = self.loc(ag.start_location)
            start_timestep = ag.start_timestep
            free_agents[local_idx] = np.array(location + [start_timestep], dtype=np.float32)
            self.free_agent_id_map[local_idx] = global_id
            assignable_agent_is_delivering[local_idx, 0] = 1.0 if ag.is_delivering else 0.0
            free_agent_cnt += 1

        for k,v in self.agent_task_pair.items():
            if k in self.delivering_agent_id_map.values():
                delivering_task_agent_map[v[0]] = k

        # print("delivering_task_agent_map:", delivering_task_agent_map)

        delivering_tasks_id = [self.agent_task_pair[i][0] for i in self.delivering_agent_id_map.values()]
        free_task_cnt = 0
        delivering_task_cnt = 0
        reversed_delivering_agent_id_map = {v: k for k, v in self.delivering_agent_id_map.items()}
        
        dropped_free_tasks = 0
        for task in status.tasks:
            if free_task_cnt >= self.task_num:
                dropped_free_tasks += 1
                continue
            task_id = task.task_id
            pickup, delivery = task.goal_arr[:2]
            pickup = self.loc(pickup)
            delivery = self.loc(delivery)
            # if task_id not in delivering_tasks_id:
            self.free_task_id_map[free_task_cnt] = task_id
            free_tasks[free_task_cnt] = np.array((pickup+delivery), dtype=np.float32)
            free_task_p2d[free_task_cnt, 0] = float(self._get_distance(tuple(pickup), tuple(delivery)))
            free_task_cnt += 1
            # else:
        
        # print("free_task_id_map:", self.free_task_id_map)
        delivering_task_id_to_local = {}
        dropped_delivering_tasks = 0
        for task in status.delivering_tasks:
            if delivering_task_cnt >= self.task_num:
                dropped_delivering_tasks += 1
                continue
            task_id = task.task_id
            pickup = self.loc(task.goal_arr[0]) if len(task.goal_arr) > 0 else [-1, -1]
            delivery = self.loc(task.goal_arr[-1]) if len(task.goal_arr) > 0 else [-1, -1]
            self.delivering_task_id_map[delivering_task_cnt] = task_id
            delivering_task_id_to_local[task_id] = delivering_task_cnt
            agent_id = reversed_delivering_agent_id_map[delivering_task_agent_map[task_id]]
            delivering_tasks[delivering_task_cnt] = np.array(([agent_id]+pickup+delivery), dtype=np.float32)
            delivering_task_cnt += 1

        if self.debug_env and (dropped_free_tasks > 0 or dropped_delivering_tasks > 0):
            print(
                f"[env] task buffer capped by task_num={self.task_num}: "
                f"dropped_free={dropped_free_tasks}, dropped_delivering={dropped_delivering_tasks}"
            )

        # assignable行到其当前delivering task索引映射（非delivering为-1）
        for local_idx, global_id in self.free_agent_id_map.items():
            if global_id in self.agent_task_pair:
                d_task_id = self.agent_task_pair[global_id][0]
                if d_task_id in delivering_task_id_to_local:
                    assignable_agent_delivering_task_idx[local_idx, 0] = float(delivering_task_id_to_local[d_task_id])

        # print("delivering_task_id_map:", self.delivering_task_id_map)

        # assert len(status.delivering_tasks) == len(self.delivering_task_id_map)
        # build_state方法不应该计算奖励，奖励应该在step方法中计算
        # reward = status.estimated_service_time  # 移除错误的奖励计算

        self.reverse_task_id_map = {v: k for k, v in self.free_task_id_map.items()}

        expert_actions = []
        delivering_task_id_set = set(delivering_tasks_id)
        for agent_key, agent_id in sorted(self.free_agent_id_map.items()):
            picked = free_task_cnt
            if len(status.agent_task_sequences[agent_id]) > 0:
                for t in status.agent_task_sequences[agent_id]:
                    mapped = self.reverse_task_id_map.get(t, -1)
                    if mapped >= 0 and t not in delivering_task_id_set:
                        picked = mapped
                        break
            expert_actions.append(picked)

        expert_actions.append(free_task_cnt)
        
        expert_actions_padded = np.full((self.agent_num[1]+1), -100, dtype=np.int32)
        for i in range(len(expert_actions)):
            expert_actions_padded[i] = expert_actions[i]
        
        # if task_cnt == 0:
        #     breakpoint()
        self.expert_estimated_finish_time = status.expert_estimated_finish_time

        # Calculate distance maps to current pickup and delivery locations
        pickup_locations = []
        delivery_locations = []
        for j in range(free_task_cnt):
            pickup_loc = tuple(map(int, free_tasks[j][:2]))
            delivery_loc = tuple(map(int, free_tasks[j][2:]))
            pickup_locations.append(pickup_loc)
            delivery_locations.append(delivery_loc)
        
        if self.compute_cnn_distance_maps:
            pickup_distances, delivery_distances = self.cal_pickup_delivery_heuristics(pickup_locations, delivery_locations)
        else:
            pickup_distances = np.full(self.grid_size, -1.0, dtype=np.float32)
            delivery_distances = np.full(self.grid_size, -1.0, dtype=np.float32)

        # Construct CNN channel maps
        obstacle_map = np.zeros(self.grid_size, dtype=np.float32)
        free_agent_map = np.zeros(self.grid_size, dtype=np.float32)
        delivering_agent_map = np.zeros(self.grid_size, dtype=np.float32)
        delivering_task_id_map = np.zeros(self.grid_size, dtype=np.float32)
        pickup_location_map = np.zeros(self.grid_size, dtype=np.float32)
        delivery_location_map = np.zeros(self.grid_size, dtype=np.float32)
        
        # Obstacle channel: 1 for obstacles, 0 for free space
        obstacle_map[self.grid == -1] = 1.0
        
        # Free agent channel: 1 for free agent positions
        for i in range(free_agent_cnt):
            x, y = int(free_agents[i][0]), int(free_agents[i][1])
            free_agent_map[x, y] = 1.0
        
        # Delivering agent channel: 1 for delivering agent positions
        for i in range(delivering_agent_cnt):
            x, y = int(delivering_agents[i][0]), int(delivering_agents[i][1])
            delivering_agent_map[x, y] = 1.0
        
        # Delivering task id channel: j/M for delivering agents with task j
        total_tasks = max(1, free_task_cnt + delivering_task_cnt)  # Avoid division by zero
        for i in range(delivering_task_cnt):
            agent_id = int(delivering_tasks[i][0])  # Agent id is already mapped
            x, y = int(delivering_agents[agent_id][0]), int(delivering_agents[agent_id][1])
            task_id = self.delivering_task_id_map[i]
            # Add offset to avoid conflict with free task IDs
            delivering_task_id_map[x, y] = (i + free_task_cnt + 1) / total_tasks
        
        # Pickup location channel: j/M for pickup locations of task j
        for j in range(free_task_cnt):
            x, y = int(free_tasks[j][0]), int(free_tasks[j][1])
            pickup_location_map[x, y] = (j + 1) / total_tasks
        
        # Delivery location channel: j/M for delivery locations of task j  
        for j in range(free_task_cnt):
            x, y = int(free_tasks[j][2]), int(free_tasks[j][3])
            delivery_location_map[x, y] = (j + 1) / total_tasks
        
        # Add delivering tasks' delivery locations to delivery location channel
        for i in range(delivering_task_cnt):
            x, y = int(delivering_tasks[i][3]), int(delivering_tasks[i][4])  # delivery location from delivering_tasks
            # Add offset to avoid conflict with free task IDs
            delivery_location_map[x, y] = (i + free_task_cnt + 1) / total_tasks

        # 修改free_agents_nearest_tasks格式：[agent_id, task_rank, [task_id, agent_to_pickup_distance, pickup_to_delivery_distance]]
        # 统一候选口径：每个agent最多保留min(K, free_task_cnt)个候选
        free_agents_nearest_tasks = np.full((self.agent_num[1], self.obs_candidate_task_k, 3), -1, dtype=np.float32)
        # 与solver对齐：每步有效候选数使用 max(num_assignable_agents, base_k) 再截断到任务数
        effective_candidate_k = min(max(int(free_agent_cnt), int(self.candidate_task_k)), int(free_task_cnt))
        
        for i in range(free_agent_cnt):
            agent_loc = tuple(map(int, free_agents[i][:2]))
            agent_start_timestep = free_agents[i][2]
            task_info_list = []
            
            for j in range(free_task_cnt):
                task_pickup = tuple(map(int, free_tasks[j][:2]))
                task_delivery = tuple(map(int, free_tasks[j][2:]))
                
                # 计算两个距离（agent到pickup加上start_timestep）
                agent_to_pickup = agent_start_timestep + self._get_distance(agent_loc, task_pickup)
                pickup_to_delivery = self._get_distance(task_pickup, task_delivery)
                total_distance = agent_to_pickup + pickup_to_delivery  # 与LNS保持一致的总距离
                
                # 使用原始距离，不归一化
                task_info_list.append((j, total_distance, agent_to_pickup, pickup_to_delivery))
            
            # 按总距离排序（与LNS的cost计算方式保持一致）
            task_info_list.sort(key=lambda x: x[1])

            # 存储最近的任务信息（固定上限K）
            nearest_count = min(effective_candidate_k, self.obs_candidate_task_k, free_task_cnt)
            for k in range(nearest_count):
                task_id, total_dist, agent_to_pickup, pickup_to_delivery = task_info_list[k]
                free_agents_nearest_tasks[i, k, 0] = task_id  # 任务ID
                free_agents_nearest_tasks[i, k, 1] = agent_to_pickup  # 原始agent到pickup距离
                free_agents_nearest_tasks[i, k, 2] = pickup_to_delivery  # 原始pickup到delivery距离

        # assignable agents两两之间的图最短路距离（不包含start_timestep）
        for i in range(free_agent_cnt):
            loc_i = tuple(map(int, free_agents[i][:2]))
            assignable_agent_a2a[i, i] = 0.0
            for j in range(i + 1, free_agent_cnt):
                loc_j = tuple(map(int, free_agents[j][:2]))
                try:
                    d = float(self._get_distance(loc_i, loc_j))
                except Exception:
                    d = float(self._distance_missing)
                assignable_agent_a2a[i, j] = d
                assignable_agent_a2a[j, i] = d

        # 为delivering_task构建到free_task的nearest列表（按你的DT->FT语义）
        for dt_idx in range(delivering_task_cnt):
            d_end = tuple(map(int, delivering_tasks[dt_idx][3:5]))  # delivering任务结束位置
            task_info_list = []
            for ft_idx in range(free_task_cnt):
                task_pickup = tuple(map(int, free_tasks[ft_idx][:2]))
                task_delivery = tuple(map(int, free_tasks[ft_idx][2:]))
                d_end_to_pickup = self._get_distance(d_end, task_pickup)
                pickup_to_delivery = self._get_distance(task_pickup, task_delivery)
                total_distance = d_end_to_pickup + pickup_to_delivery
                task_info_list.append((ft_idx, total_distance, d_end_to_pickup, pickup_to_delivery))
            task_info_list.sort(key=lambda x: x[1])
            nearest_count = min(effective_candidate_k, self.obs_candidate_task_k, free_task_cnt)
            for k in range(nearest_count):
                task_id, total_dist, d_end_to_pickup, pickup_to_delivery = task_info_list[k]
                delivering_tasks_nearest_tasks[dt_idx, k, 0] = task_id
                delivering_tasks_nearest_tasks[dt_idx, k, 1] = d_end_to_pickup
                delivering_tasks_nearest_tasks[dt_idx, k, 2] = pickup_to_delivery

        # if not self.training:
        #     print("id:", self.seed, "free_agents", free_agents, "delivering_agents:", delivering_agents, "free_tasks:", free_tasks, "delivering_tasks:", delivering_tasks)
        #     print("expert_actions:", status.agent_task_sequences)
        #     print("expert_estimated_service_time:", self.expert_estimated_service_time)
        # 准备基础观察字典
        obs_dict = {
            "env_id": self.seed,
            "free_agents": free_agents,
            "delivering_agents": delivering_agents,
            "free_tasks": free_tasks,
            "delivering_tasks": delivering_tasks,
            "free_agents_num": free_agent_cnt,
            "delivering_agents_num": delivering_agent_cnt,
            "free_tasks_num": free_task_cnt,
            "delivering_tasks_num": delivering_task_cnt,
            "expert_actions": expert_actions_padded,
            "free_agents_nearest_tasks": free_agents_nearest_tasks,
            "assignable_agent_is_delivering": assignable_agent_is_delivering,
            "assignable_agent_delivering_task_idx": assignable_agent_delivering_task_idx,
            "delivering_tasks_nearest_tasks": delivering_tasks_nearest_tasks,
            "assignable_agent_a2a": assignable_agent_a2a,
            "free_task_p2d": free_task_p2d,
            "grid": self.grid,
            "pickup_distances": pickup_distances,
            "delivery_distances": delivery_distances,
            "obstacle_map": obstacle_map,
            "free_agent_map": free_agent_map,
            "delivering_agent_map": delivering_agent_map,
            "delivering_task_id_map": delivering_task_id_map,
            "pickup_location_map": pickup_location_map,
            "delivery_location_map": delivery_location_map,
        }
        
        # 添加SP-MPNN距离信息（展平到顶层）
        for dist in range(1, self.sp_mpnn_max_distance + 1):
            dist_key = f'dist_{dist}'
            obs_key = f'sp_mpnn_dist_{dist}'  # 新的观察空间键名
            max_edges = int(self.sp_mpnn_max_edges_by_dist.get(dist, 1))
            
            if dist_key in self.sp_mpnn_distance_edges and self.sp_mpnn_distance_edges[dist_key].numel() > 0:
                # 转换为numpy数组
                edges_tensor = self.sp_mpnn_distance_edges[dist_key]
                edges_np = edges_tensor.numpy().astype(np.int32)
                
                # 填充到观察空间的固定形状
                padded_edges = np.zeros((2, max_edges), dtype=np.int32)
                if edges_np.shape[1] > 0:
                    valid_count = min(edges_np.shape[1], max_edges)
                    padded_edges[:, :valid_count] = edges_np[:, :valid_count]
                
                obs_dict[obs_key] = padded_edges
            else:
                # 创建空的边索引，符合观察空间形状
                obs_dict[obs_key] = np.zeros((2, max_edges), dtype=np.int32)
        
        self.last_status_payload = self._serialize_status(status)
        # build_state方法返回状态信息，不返回奖励
        # 奖励应该在step方法中根据具体逻辑计算
        return obs_dict, status.estimated_finish_time, status.estimated_service_time, done, True

    def decode_action(self, action, mutate=True, verbose=None):
        try:
            action_arr = np.asarray(action)
        except Exception:
            action_arr = np.array(action)
        if action_arr.ndim > 1:
            action_arr = action_arr.reshape(-1)
        action_arr = action_arr.astype(np.int64, copy=False)[:self.agent_num_now]
        action_list = action_arr.tolist()
        self._dbg(f"decode_action: len={len(action_list)} agent_num_now={self.agent_num_now}")
        penalty = 0
        avail_task = 0
        agent_tasks = [[] for _ in range(max(0, int(self.agent_num_now)))]

        assigned_task = []

        last_action = self.last_action
        self.last_action = {}

        for k, v in self.free_agent_id_map.items():
            if k >= len(action_list):
                continue
            if v < 0 or v >= len(agent_tasks):
                continue
            if action_list[k] in self.free_task_id_map:
                task_id = self.free_task_id_map[action_list[k]]
                if task_id != -1:
                    if task_id not in assigned_task:
                        agent_tasks[v] = [task_id]
                        assigned_task.append(task_id)
                        avail_task += 1
                        self.last_action[v] = task_id
        
        switch_agents = 0
        if last_action != None:
            for k, v in self.last_action.items():
                if k in last_action.keys():
                    la = last_action[k]
                    if la != v:
                        switch_agents += 1

        # print("switch_agents:", switch_agents)

        for k, v in self.agent_task_pair.items():
            if k in self.delivering_agent_id_map.values():
                # truncated_size=1: delivering动作不生效，仅保留当前任务（由solver内部自动补）
                # truncated_size>1: 若该agent本轮被分配了next task，则保留其选择；否则空
                if self.task_truncated_size <= 1:
                    agent_tasks[k] = []

        # 控制是否写入环境状态
        if mutate:
            self.last_task_id.clear()
            for agent_task in agent_tasks:
                for t in agent_task:
                    self.last_task_id.append(t)

        if verbose is None:
            verbose = (not self.training)
        if verbose:
            print("id:", self.seed, "agent_tasks:",agent_tasks)
            sys.stdout.flush()
        return agent_tasks

    def _decode_with_maps(self, action, maps):
        action_list = action
        # print("action_list: ", action_list)
        free_agent_id_map = maps["free_agent_id_map"]
        delivering_agent_id_map = maps["delivering_agent_id_map"]
        agent_task_pair = maps["agent_task_pair"]
        free_task_id_map = maps["free_task_id_map"]

        total_agents = int(self.agent_num_now)
        if total_agents <= 0:
            # 回退：由映射中最大global id估计
            max_id = -1
            if free_agent_id_map:
                max_id = max(max_id, max(free_agent_id_map.values()))
            if delivering_agent_id_map:
                max_id = max(max_id, max(delivering_agent_id_map.values()))
            if agent_task_pair:
                max_id = max(max_id, max(agent_task_pair.keys()))
            total_agents = max_id + 1
        agent_tasks = [[] for _ in range(max(0, total_agents))]

        assigned_task = []

        for k, v in free_agent_id_map.items():
            if k >= len(action_list):
                continue
            if v < 0 or v >= len(agent_tasks):
                continue
            if action_list[k] in free_task_id_map:
                task_id = free_task_id_map[action_list[k]]
                if task_id != -1:
                    if task_id not in assigned_task:
                        agent_tasks[v] = [task_id]
                        assigned_task.append(task_id)

        for k, v in agent_task_pair.items():
            if k in delivering_agent_id_map.values():
                if int(maps.get("task_truncated_size", 1)) <= 1:
                    agent_tasks[k] = []
        return agent_tasks

    def _safe_total_distance(self, agent_loc, agent_start_timestep, task_pickup, task_delivery):
        """
        与free_agents_nearest_tasks一致的代理代价:
          total = (agent_start_timestep + d(agent,pickup)) + d(pickup,delivery)
        不可达时给大惩罚，避免负距离污染比较。
        """
        try:
            d1 = int(self._get_distance(agent_loc, task_pickup))
            d2 = int(self._get_distance(task_pickup, task_delivery))
            if d1 < 0 or d2 < 0:
                return float(2 * (self.grid_size[0] + self.grid_size[1]))
            return float(agent_start_timestep + d1 + d2)
        except Exception:
            return float(2 * (self.grid_size[0] + self.grid_size[1]))

    def _compute_action_hungarian_gap(self, agent_tasks):
        """
        在当前状态下比较：
          - model执行后的有效分配总代价（按最短路代理代价）
          - Hungarian最优总代价（同一代价定义）
        返回字典，供step日志打印。
        """
        try:
            if not isinstance(self.state, dict):
                return None
            M = int(self.state.get("free_agents_num", 0))
            N = int(self.state.get("free_tasks_num", 0))
            if M <= 0 or N <= 0:
                return {
                    "model_cost": 0.0,
                    "hungarian_cost": 0.0,
                }

            assignable_agents = self.state["free_agents"]
            free_tasks = self.state["free_tasks"]
            agent_is_delivering = self.state.get("assignable_agent_is_delivering", None)
            agent_delivering_task_idx = self.state.get("assignable_agent_delivering_task_idx", None)
            delivering_tasks = self.state.get("delivering_tasks", None)

            # 构建MxN代价矩阵
            cost_matrix = np.zeros((M, N), dtype=np.float32)
            for i in range(M):
                ax, ay, st = int(assignable_agents[i][0]), int(assignable_agents[i][1]), int(assignable_agents[i][2])
                agent_loc = (ax, ay)
                is_delivering = 0.0
                if agent_is_delivering is not None:
                    try:
                        is_delivering = float(agent_is_delivering[i][0])
                    except Exception:
                        is_delivering = 0.0
                for j in range(N):
                    px, py, dx, dy = map(int, free_tasks[j][:4])
                    task_pickup = (px, py)
                    task_delivery = (dx, dy)
                    if is_delivering >= 0.5 and self.task_truncated_size > 1 and delivering_tasks is not None and agent_delivering_task_idx is not None:
                        # delivering agent:
                        #   full cost = a->current-delivery(remain) + current-delivery-end->next-pickup + pickup->delivery
                        # 其中 a->current-delivery(remain) 用第三维st近似（由solver回传）
                        dt_idx = int(agent_delivering_task_idx[i][0])
                        if 0 <= dt_idx < len(delivering_tasks):
                            d_end_x, d_end_y = map(int, delivering_tasks[dt_idx][3:5])
                            d_end = (d_end_x, d_end_y)
                            d_end_to_pickup = int(self._get_distance(d_end, task_pickup))
                            pickup_to_delivery = int(self._get_distance(task_pickup, task_delivery))
                            if d_end_to_pickup < 0 or pickup_to_delivery < 0:
                                cost_matrix[i, j] = float(2 * (self.grid_size[0] + self.grid_size[1]))
                            else:
                                cost_matrix[i, j] = float(st + d_end_to_pickup + pickup_to_delivery)
                        else:
                            cost_matrix[i, j] = self._safe_total_distance(agent_loc, st, task_pickup, task_delivery)
                    else:
                        # free agent
                        cost_matrix[i, j] = self._safe_total_distance(agent_loc, st, task_pickup, task_delivery)

            # Hungarian最优（会自动处理M!=N，返回min(M,N)对）
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            hungarian_cost = float(cost_matrix[row_ind, col_ind].sum()) if len(row_ind) > 0 else 0.0

            # model有效分配代价（使用decode_action后的agent_tasks，已去重且可执行）
            reverse_task_id_map = {v: k for k, v in self.free_task_id_map.items()}
            model_cost = 0.0
            for local_agent_idx, global_agent_id in self.free_agent_id_map.items():
                if global_agent_id >= len(agent_tasks):
                    continue
                assigned = agent_tasks[global_agent_id]
                if not assigned:
                    continue
                task_id = assigned[0]
                local_task_idx = reverse_task_id_map.get(task_id, -1)
                if 0 <= local_task_idx < N:
                    model_cost += float(cost_matrix[local_agent_idx, local_task_idx])

            return {
                "model_cost": float(model_cost),
                "hungarian_cost": float(hungarian_cost),
            }
        except Exception:
            return None

    def _serialize_status(self, status):
        def serialize_path(path):
            out = []
            for st in path:
                try:
                    loc = self.loc(st.location)
                except Exception:
                    loc = [-1, -1]
                out.append({
                    "loc": loc,
                    "t": int(getattr(st, "timestep", -1)),
                    "o": int(getattr(st, "orientation", -1)),
                    "l": int(getattr(st, "l_val", -1)),
                })
            return out

        def serialize_task(task):
            goals = getattr(task, "goal_arr", [])
            goals_xy = []
            for g in goals:
                try:
                    goals_xy.append(self.loc(g))
                except Exception:
                    goals_xy.append([-1, -1])
            pickup = goals_xy[0] if len(goals_xy) > 0 else None
            delivery = goals_xy[1] if len(goals_xy) > 1 else None
            return {
                "task_id": int(getattr(task, "task_id", -1)),
                "release_time": int(getattr(task, "release_time", -1)),
                "pickup": pickup,
                "delivery": delivery,
                "goal_arr": goals_xy,
                "estimated_finish_time": int(getattr(task, "estimated_finish_time", 0)),
                "estimated_service_time": int(getattr(task, "estimated_service_time", 0)),
            }

        agents_free = []
        agents_delivering = []
        for i, ag in enumerate(getattr(status, "agents_all", [])):
            agent_info = {
                "idx": i,
                "agent_id": int(getattr(ag, "agent_id", -1)),
                "start_location": self.loc(getattr(ag, "start_location", -1)),
                "start_timestep": int(getattr(ag, "start_timestep", -1)),
                "is_delivering": bool(getattr(ag, "is_delivering", False)),
                "task_sequence": list(getattr(ag, "task_sequence", [])),
            }
            if agent_info["is_delivering"]:
                agents_delivering.append(agent_info)
            else:
                agents_free.append(agent_info)

        solution = getattr(status, "solution", [])
        paths = []
        for i, path in enumerate(solution):
            paths.append({
                "idx": i,
                "path": serialize_path(path),
            })

        return {
            "timestep": int(getattr(status, "timestep", -1)),
            "task_truncated_size": int(getattr(status, "task_truncated_size", self.task_truncated_size)),
            "agents_free": agents_free,
            "agents_delivering": agents_delivering,
            "agent_task_sequences": [list(seq) for seq in getattr(status, "agent_task_sequences", [])],
            "agent_task_pair": {int(k): (int(v[0]), int(v[1])) for k, v in getattr(status, "agent_task_pair", {}).items()},
            "paths": paths,
            "tasks": [serialize_task(t) for t in getattr(status, "tasks", [])],
            "delivering_tasks": [serialize_task(t) for t in getattr(status, "delivering_tasks", [])],
        }

    def _merge_status_for_showcase(self, base_status, updated_status):
        """
        Use base_status for agent/task positions (pre-step),
        and updated_status for assignments/paths (post-step).
        """
        if base_status is None:
            return updated_status
        merged = dict(base_status)
        if updated_status is None:
            return merged
        for key in ("agent_task_sequences", "agent_task_pair", "paths", "timestep"):
            if key in updated_status:
                merged[key] = updated_status[key]
        return merged

    def reset(self, seed=40):
        args = [
            "--map", self.grid_path,
            "--agentNum", str(self.num_r),
            "--seed", str(self.seed),
            "--solver", self.solver_name,
            "--task_truncated_size", str(self.task_truncated_size),
            "--candidate_task_k", str(self.candidate_task_k),
            "--infer_use_expert_fallback", "false" if self.model_only_eval else "true",
        ]
        if self.eval_pending_task_cap is not None:
            args += ["--pending_task_cap", str(self.eval_pending_task_cap)]
        self.solver = PBSSolver(args)

        self.last_total_finish_time = 0
        self.agent_num_now = 0
        self.last_finish_time = 0
        self.last_task_id = []
        self.task_id_map = {}
        self.free_agent_id_map = {}
        self.delivering_agent_id_map = {}
        self.agent_task_pair = {}
        self.free_task_id_map = {}
        self.delivering_task_id_map = {}
        self.last_free_agent_num = 0
        self.expert_estimated_finish_time = 0
        self.last_expert_action = None
        self.last_action = None

        if self.training: 
            agents, tasks, task_frequency, task_release_time = self.generate_agents_tasks()
        else:
            agents = []
            with open(self.eval_data_path, 'r') as f:
                task_num, task_frequency, task_release_time = f.readline().strip().split(" ")
                task_frequency = float(task_frequency)
                task_release_time = int(task_release_time)
                lines = [line.strip().split() for line in f]
                tasks = [[int(line[0]), int(line[1]), int(line[2])] for line in lines]
                if self.eval_max_tasks is not None:
                    tasks = tasks[:self.eval_max_tasks]

        print("before update task")
        if self.training:
            status = self.solver.update_task(tasks, agents, 5000, task_frequency, task_release_time, 1)
        else:
            status = self.solver.update_task(tasks, [], self.eval_simulation_time, task_frequency, task_release_time, 0)
            self.agent_num_now = len(status.agents_all)

        self.step_count = 0
        self.state, _, _, _, _ = self.build_state(status)
        if not self.storage_ready:
            self._capture_storage_snapshot()

        self.storage_ready = True
        try:
            self._dbg(
                f"reset: storage_ready={self.storage_ready}, fa={self.state['free_agents_num']}, ft={self.state['free_tasks_num']}, da={self.state['delivering_agents_num']}, dt={self.state['delivering_tasks_num']}",
                force=True,
            )
        except Exception:
            self._dbg("reset: storage_ready set", force=True)

        while (self.state['free_tasks_num'] == 0 or self.state['free_agents_num'] == 0):
            agent_tasks = self.decode_action(np.full((self.agent_num[1]+1), -100, dtype=np.int32))
            if self.training:
                status = self.solver.update(agent_tasks, 1, 0, 0)
            else:
                status = self.solver.update(agent_tasks, 1, 0, 1)
            self.state, _,_,_,_ = self.build_state(status)

        self.last_free_task_num = self.state['free_tasks_num']
        self.last_free_agent_num = self.state['free_agents_num']

        self.reset_this_step = True

        print("reset finish")
        if self.training:
            return self.state, {}
        else:
            return self.state
    
    def step(self, action):
        self.reset_this_step = False
        self._dbg("step: begin")
        self._capture_storage_snapshot()
        pre_status_payload = getattr(self, "last_status_payload", None)
        last_expert_action = self._decode_with_maps(self.state['expert_actions'], self.storage_snapshot)
        # print("last_expert_action:", self._decode_with_maps(self.state['expert_actions'], self.storage_snapshot))
        agent_tasks = self.decode_action(action, mutate=True)
        assign_gap = self._compute_action_hungarian_gap(agent_tasks)

        if self.training:
            status = self.solver.update(agent_tasks, 1, 1, 0)
            last_expert_finish_time = self.expert_estimated_finish_time
        else:
            status = self.solver.update(agent_tasks, 1, 1, 1)
            last_expert_finish_time = self.expert_estimated_finish_time
            # status = self.solver.update(agent_tasks, 0, 1)
            # last_expert_service_time = 0
        self.state, finish_time, service_time, done, valid = self.build_state(status)
        self.storage_ready = True
        
        while not done and (self.state['free_tasks_num'] == 0 or self.state['free_agents_num'] == 0):
            agent_tasks = self.decode_action(np.full((self.agent_num[1]+1), -100, dtype=np.int32))
            if self.training:
                status = self.solver.update(agent_tasks, 1, 0, 0)
            else:
                status = self.solver.update(agent_tasks, 1, 0, 1)
            self.state, _, _, done, valid = self.build_state(status)
        
        self.step_count += 1

        if self.training:
            s_time = finish_time - last_expert_finish_time # use expert service time as baseline
        else:
            s_time = finish_time - last_expert_finish_time
            # s_time = service_time - self.last_service_time
            
        self.last_finish_time = finish_time
        
        # breakpoint()
        if valid:
            if self.last_free_agent_num == 0:
                self.last_free_agent_num = self.agent_num_now
            reward = -(s_time)/((self.grid_size[0]+self.grid_size[1])*min(self.last_free_agent_num, self.last_free_task_num))
            # reward = -(s_time)/min(self.last_free_agent_num, self.last_free_task_num)
        else:
            reward = -1

        try:
            if os.environ.get("SHOWCASE", "False").upper() in ("TRUE", "1", "YES"):
                if valid and s_time <= -5:
                    expert_status = self.solver.update_storage(last_expert_action)
                    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
                    out_dir = os.path.join(base_dir, "showcase")
                    os.makedirs(out_dir, exist_ok=True)
                    ts = int(time.time() * 1000)
                    out_path = os.path.join(out_dir, f"case_{ts}.json")
                    model_status = self._merge_status_for_showcase(
                        pre_status_payload,
                        self._serialize_status(status),
                    )
                    expert_status = self._merge_status_for_showcase(
                        pre_status_payload,
                        self._serialize_status(expert_status),
                    )
                    payload = {
                        "seed": int(self.seed),
                        "episode": int(self.episode),
                        "step_count": int(self.step_count),
                        "s_time": float(s_time),
                        "reward": float(reward),
                        "model_action": [int(x) for x in np.asarray(action).reshape(-1).tolist()],
                        "expert_action": [int(x) for x in np.asarray(self.state["expert_actions"]).reshape(-1).tolist()],
                        "model_status": model_status,
                        "expert_status": expert_status,
                    }
                    with open(out_path, "w", encoding="utf-8") as f:
                        f.write(json.dumps(payload, ensure_ascii=True))
        except Exception:
            pass

        if reward > 500:
            reward = 0
        

        if assign_gap is not None:
            print(
                "id:", self.seed,
                "makespan", status.timestep,
                "reward:", reward,
                "finish_time:", self.last_finish_time,
                "service_time:", service_time,
                "s_time:", s_time,
                "agent_num:", self.agent_num_now,
                "done:", done,
                "last_free_agent_num:", self.last_free_agent_num,
                "last_free_task_num:", self.last_free_task_num,
                "model_assign_cost:", round(assign_gap["model_cost"], 3),
                "hungarian_assign_cost:", round(assign_gap["hungarian_cost"], 3),
            )
        else:
            print("id:", self.seed, "makespan", status.timestep,"reward:",reward, "finish_time:", self.last_finish_time, "service_time:", service_time, "s_time:", s_time, "agent_num:", self.agent_num_now, "done:", done, "last_free_agent_num:", self.last_free_agent_num, "last_free_task_num:", self.last_free_task_num,)
        print("______________________")

        if not done:
            self.last_free_agent_num = self.state['free_agents_num']
            self.last_free_task_num = self.state['free_tasks_num']
        else:
            self.last_free_agent_num = self.agent_num_now
            self.last_free_task_num = 1
        if done:
            self.episode += 1

        sys.stdout.flush()
        # if done:
        #     breakpoint()
        # if self.step_count >= self.max_steps:
        #     done = True
        info = {
            "solver_timestep": int(getattr(status, "timestep", 0)),
            "makespan": int(getattr(status, "timestep", 0)),
            "assignment_makespan": int(getattr(status, "makespan", 0)),
            "estimated_finish_time": int(finish_time),
            "estimated_service_time": int(service_time),
            "num_finished_tasks": int(getattr(status, "num_finished_tasks", 0)),
            # 口径统一：pending_tasks仅统计free tasks（current tasks），不计delivering tasks
            "pending_tasks": int(len(getattr(status, "tasks", []))),
            "valid": bool(valid),
            "time_limit_reached": bool(getattr(status, "time_limit_reached", False)),
        }
        if info["time_limit_reached"]:
            info["done_reason"] = "time_limit"
        elif done and valid:
            info["done_reason"] = "all_finished"
        elif done and not valid:
            info["done_reason"] = "invalid"
        else:
            info["done_reason"] = "ongoing"
        if self.training:
            return self.state, reward, done, False, info
        if done:
            return self.state, service_time, done, False, info
        return self.state, 0, done, False, info

    
    def evaluate_action_storage(self, action):
        """
        只读：评估单个动作的即时奖励与服务时间，不推进环境状态。
        返回: (reward, s_time, service_time, valid)
        """
        try:
            # fa = int(self.state.get('free_agents_num', 0)) if isinstance(self.state, dict) else 0
            # da = int(self.state.get('delivering_agents_num', 0)) if isinstance(self.state, dict) else 0
            # ft = int(self.state.get('free_tasks_num', 0)) if isinstance(self.state, dict) else 0
            # print("action_in: ", action)
            agent_tasks = self._decode_with_maps(action, self.storage_snapshot)
            # print("agent_tasks: ", agent_tasks)
            status = self.solver.update_storage(agent_tasks)
            finish_time = status.estimated_finish_time
            if self.training:
                last_expert_finish_time = self.storage_snapshot['expert_estimated_finish_time']
                s_time = finish_time - last_expert_finish_time
            else:
                s_time = finish_time - self.last_finish_time

            grid_scale = (self.grid_size[0] + self.grid_size[1])
            # grid_scale = 1
            min_count = min(self.storage_snapshot['last_free_agent_num'] or 0, self.storage_snapshot['last_free_task_num'] or 0)
            min_val = max(1, min_count)
            if status.valid:
                reward = -(s_time) / (grid_scale * min_val)
            else:
                reward = -1
            if reward > 500:
                reward = 0

            return reward, s_time, finish_time, bool(status.valid)
        except Exception:
            self._dbg("eval_storage: exception caught")
            return 0.0, 0.0, 0.0, False

    def evaluate_actions_storage(self, actions_dict):
        """
        只读：评估多个动作，逐个独立计算，不推进环境状态。
        :param actions_list: 可迭代的动作集合
        :return: 列表[(reward, s_time, service_time, valid), ...]
        """
        # py_list = []
        # print("evaluate_actions_storage: ", actions_list)
        results = []
        actions_list = actions_dict[0][self.seed]

        if actions_list == -1 or self.reset_this_step:
            return (self.seed, None)
        # print("evaluate_actions_storage: ", py_list)
        for action in actions_list:
            # print("action_out: ", action)
            result_action = self.evaluate_action_storage(action)
            # print("result_action: ", result_action)
            results.append(result_action)
        return (self.seed, results)


    def multi_source_bfs(self, queue, distance_map):
        """Multi-source BFS to calculate shortest distances from multiple starting points"""
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        while queue:
            x, y = queue.popleft()
            current_distance = distance_map[x, y]
            
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                
                if (0 <= nx < self.grid_size[0] and 
                    0 <= ny < self.grid_size[1] and 
                    self.grid[nx, ny] == 0 and 
                    distance_map[nx, ny] > current_distance + 1):
                    
                    distance_map[nx, ny] = current_distance + 1
                    queue.append((nx, ny))

    def cal_pickup_delivery_heuristics(self, pickup_locations, delivery_locations):
        """Calculate distance maps to nearest pickup and delivery locations using multi-source BFS"""
        # Initialize distance maps with infinity
        pickup_distances = np.full(self.grid_size, float('inf'), dtype=np.float32)
        delivery_distances = np.full(self.grid_size, float('inf'), dtype=np.float32)
        
        # Multi-source BFS from current pickup locations
        pickup_queue = deque()
        for x, y in pickup_locations:
            pickup_distances[x, y] = 0
            pickup_queue.append((x, y))
        
        if pickup_queue:
            self.multi_source_bfs(pickup_queue, pickup_distances)
        
        # Multi-source BFS from current delivery locations
        delivery_queue = deque()
        for x, y in delivery_locations:
            delivery_distances[x, y] = 0
            delivery_queue.append((x, y))
        
        if delivery_queue:
            self.multi_source_bfs(delivery_queue, delivery_distances)
        
        # 归一化处理：用网格长+宽作为最大距离进行归一化，不可达的节点设为-1
        max_distance = self.grid_size[0] + self.grid_size[1]
        
        # 处理pickup_distances
        pickup_distances_normalized = np.where(
            pickup_distances == float('inf'), 
            -1.0,  # 不可达的节点设为-1
            pickup_distances / max_distance  # 可达的节点归一化到[0,1]
        ).astype(np.float32)
        
        # 处理delivery_distances
        delivery_distances_normalized = np.where(
            delivery_distances == float('inf'), 
            -1.0,  # 不可达的节点设为-1
            delivery_distances / max_distance  # 可达的节点归一化到[0,1]
        ).astype(np.float32)
            
        return pickup_distances_normalized, delivery_distances_normalized

    def precompute_sp_mpnn_distances(self):
        """
        预计算SP-MPNN所需的k-hop距离信息
        这个计算在环境初始化时进行一次，避免在每个step重复计算
        """
        cache_path = None
        try:
            if self.grid_path is not None:
                cache_path = f"{self.grid_path}_spmpnn_k{int(self.sp_mpnn_max_distance)}.pt"
            if cache_path is not None and os.path.exists(cache_path):
                cached = torch.load(cache_path, map_location="cpu")
                if isinstance(cached, dict):
                    self.sp_mpnn_distance_edges = cached
                    print("reading sp_mpnn distance finished.")
                    return
        except Exception:
            pass

        height, width = self.grid_size
        num_nodes = height * width
        
        # 创建grid图的边索引（4连通）
        edges = []
        for i in range(height):
            for j in range(width):
                if self.grid[i, j] != -1:  # 非障碍物
                    current_idx = i * width + j
                    # 4个方向
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < height and 0 <= nj < width and self.grid[ni, nj] != -1:
                            neighbor_idx = ni * width + nj
                            edges.append([current_idx, neighbor_idx])
        
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        
        # 计算最短路径距离矩阵
        distance_edges = self._compute_shortest_paths(edge_index, num_nodes)
        
        # 存储预计算的结果
        self.sp_mpnn_distance_edges = distance_edges
        try:
            if cache_path is not None:
                torch.save(distance_edges, cache_path)
        except Exception:
            pass
        # print(f"预计算SP-MPNN距离信息完成，grid大小: {height}x{width}, 最大距离: {self.sp_mpnn_max_distance}")
        for dist in range(1, self.sp_mpnn_max_distance + 1):
            dist_key = f'dist_{dist}'
            if dist_key in distance_edges:
                num_edges = distance_edges[dist_key].shape[1] if distance_edges[dist_key].numel() > 0 else 0
                # print(f"  距离{dist}: {num_edges}条边")
    
    def _compute_shortest_paths(self, edge_index, num_nodes):
        """
        超高效的SP-MPNN k-hop邻居计算
        只计算k-hop邻居，时间复杂度从O(n²)优化到O(n×k×平均度数)
        
        相比原来的全图最短路径计算：
        - 旧方法: O(n²) 计算所有节点对距离
        - 新方法: O(n×k×度数) 只计算k-hop邻居
        - 对于网格图，平均度数为4，大幅提升性能
        """
        if edge_index.size(1) == 0:
            return {}
        
        import time
        from collections import deque
        start_time = time.time()
        
        # 创建双向邻接表
        adj_list = [[] for _ in range(num_nodes)]
        for i in range(edge_index.size(1)):
            u, v = edge_index[0, i].item(), edge_index[1, i].item()
            adj_list[u].append(v)
            adj_list[v].append(u)  # 确保无向图
        
        # 按距离存储边，避免重复
        distance_edges = {f'dist_{d}': [] for d in range(1, self.sp_mpnn_max_distance + 1)}
        
        # 统计有效节点
        valid_nodes = [i for i in range(num_nodes) if len(adj_list[i]) > 0]
        
        # 对每个节点执行限制深度的BFS，只到k-hop
        for start_node in valid_nodes:
            # BFS只到max_distance层
            visited = {start_node: 0}  # 节点: 距离
            queue = deque([start_node])
            
            while queue:
                current_node = queue.popleft()
                current_dist = visited[current_node]
                
                # 如果已达到最大距离，不再扩展
                if current_dist >= self.sp_mpnn_max_distance:
                    continue
                    
                # 遍历邻居
                for neighbor in adj_list[current_node]:
                    if neighbor not in visited:
                        new_dist = current_dist + 1
                        visited[neighbor] = new_dist
                        queue.append(neighbor)
                        
                        # 记录这条边（确保start_node < neighbor避免重复）
                        if start_node < neighbor and new_dist <= self.sp_mpnn_max_distance:
                            distance_edges[f'dist_{new_dist}'].append([start_node, neighbor])
        
        # 转换为张量
        result = {}
        total_edges = 0
        for dist in range(1, self.sp_mpnn_max_distance + 1):
            dist_key = f'dist_{dist}'
            if distance_edges[dist_key]:
                # 转换为张量并创建双向边
                edges = torch.tensor(distance_edges[dist_key], dtype=torch.long).t()
                bidirectional_edges = torch.cat([edges, edges.flip(0)], dim=1)
                result[dist_key] = bidirectional_edges
                total_edges += bidirectional_edges.size(1)
            else:
                result[dist_key] = torch.zeros((2, 0), dtype=torch.long)
        
        end_time = time.time()
    
        return result
    
    def _compute_shortest_paths_pyg(self, edge_index, num_nodes):
        """
        使用PyG k_hop_subgraph的SP-MPNN实现
        这是最简洁的实现方式，可作为备选方案
        """
        if edge_index.size(1) == 0:
            return {}
        
        import time
        start_time = time.time()
        # print(f"使用PyG k_hop_subgraph方法 (节点数: {num_nodes}, 最大距离: {self.sp_mpnn_max_distance})...")
        
        # 确保边索引是无向的
        edge_index_undirected = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        edge_index_undirected = torch.unique(edge_index_undirected, dim=1)
        
        distance_edges = {f'dist_{d}': [] for d in range(1, self.sp_mpnn_max_distance + 1)}
        
        # 统计有效节点
        valid_nodes = []
        adj_list = [[] for _ in range(num_nodes)]
        for i in range(edge_index.size(1)):
            u, v = edge_index[0, i].item(), edge_index[1, i].item()
            adj_list[u].append(v)
            adj_list[v].append(u)
        
        for i in range(num_nodes):
            if len(adj_list[i]) > 0:
                valid_nodes.append(i)
        
        # print(f"有效节点数: {len(valid_nodes)} / {num_nodes}")
        
        # 为每个节点获取k-hop子图
        for center_node in valid_nodes:
            try:
                subset, k_hop_edge_index, mapping, edge_mask = k_hop_subgraph(
                    node_idx=center_node,
                    num_hops=self.sp_mpnn_max_distance,
                    edge_index=edge_index_undirected,
                    relabel_nodes=True
                )
                
                if len(subset) <= 1:
                    continue
                    
                # 在子图上用BFS计算精确距离
                center_in_subgraph = mapping  # center_node在子图中的索引
                
                # 创建子图邻接表
                sub_adj = [[] for _ in range(len(subset))]
                for i in range(k_hop_edge_index.size(1)):
                    u, v = k_hop_edge_index[0, i].item(), k_hop_edge_index[1, i].item()
                    sub_adj[u].append(v)
                
                # 在子图上BFS
                visited = {center_in_subgraph: 0}
                queue = deque([center_in_subgraph])
                
                while queue:
                    current = queue.popleft()
                    current_dist = visited[current]
                    
                    if current_dist >= self.sp_mpnn_max_distance:
                        continue
                        
                    for neighbor in sub_adj[current]:
                        if neighbor not in visited:
                            new_dist = current_dist + 1
                            visited[neighbor] = new_dist
                            queue.append(neighbor)
                            
                            # 将子图节点索引转换回原图索引
                            orig_current = subset[current].item()
                            orig_neighbor = subset[neighbor].item()
                            
                            # 避免重复边（只记录较小索引到较大索引的边）
                            if orig_current < orig_neighbor and new_dist <= self.sp_mpnn_max_distance:
                                distance_edges[f'dist_{new_dist}'].append([orig_current, orig_neighbor])
            
            except Exception as e:
                # k_hop_subgraph在某些边界情况下可能出错，回退到BFS方法
                continue
        
        # 转换为张量并去重
        result = {}
        total_edges = 0
        for dist in range(1, self.sp_mpnn_max_distance + 1):
            dist_key = f'dist_{dist}'
            if distance_edges[dist_key]:
                # 去重
                edges_set = set(tuple(edge) for edge in distance_edges[dist_key])
                edges = torch.tensor(list(edges_set), dtype=torch.long).t()
                # 创建双向边
                bidirectional_edges = torch.cat([edges, edges.flip(0)], dim=1)
                result[dist_key] = bidirectional_edges
                total_edges += bidirectional_edges.size(1)
            else:
                result[dist_key] = torch.zeros((2, 0), dtype=torch.long)
        
        end_time = time.time()
        # print(f"PyG方法完成! 总耗时: {end_time - start_time:.3f}秒")
        # print(f"总边数: {total_edges}")
        
        return result
