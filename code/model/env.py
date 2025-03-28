import gymnasium as gym
import numpy as np
from mapf_solver.mapf_solver import PBSSolver, Agent, Task, AgentTaskStatus
import random
from scipy.optimize import linear_sum_assignment
import sys
from collections import deque

class MultiAgentPickupEnv(gym.Env):
    def __init__(self, training=True, grid_path=None, seed=40, 
            solver="PBS", agent_num_lower_bound=10, agent_num_higher_bound=50, eval_data_path=None, task_num=500, pos_reward=False):
        super().__init__()
        self.training = training
        self.solver_name = solver
        self.step_count = 0
        self.seed = seed
        self.agent_num = (agent_num_lower_bound, agent_num_higher_bound)
        self.task_num = task_num
        self.pos_reward = pos_reward

        self.read_grid(grid_path)

        self.observation_space = gym.spaces.Dict({
            "free_agents_grid": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(agent_num_higher_bound, 3, self.grid_size[0], self.grid_size[1]), dtype=np.float32
            ),
            "delivering_agents_grid": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(agent_num_higher_bound, 3, self.grid_size[0], self.grid_size[1]), dtype=np.float32
            ),
            "free_agents_path_length": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(agent_num_higher_bound, 1), dtype=np.float32
            ),
            "delivering_agents_path_length": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(agent_num_higher_bound, 1), dtype=np.float32
            ),
            "tasks_grid": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(task_num, 3, self.grid_size[0], self.grid_size[1]), dtype=np.float32
            ),
            "tasks_loc": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(task_num, 2), dtype=np.float32
            ),
            "free_agents_num": gym.spaces.Discrete(agent_num_higher_bound),
            "delivering_agents_num": gym.spaces.Discrete(agent_num_higher_bound),
            "tasks_num": gym.spaces.Discrete(task_num),
            "expert_actions": gym.spaces.Box(
                low=-1, high=np.inf, 
                shape=(agent_num_higher_bound+1, ),
                dtype=np.int32
            )
        })

        self.task_id_map = {}
        self.free_agent_id_map = {}
        self.delivering_agent_id_map = {}
        self.agent_task_pair = {}
        self.action_space = gym.spaces.MultiDiscrete([task_num] * agent_num_higher_bound)
        self.eval_data_path = eval_data_path
        self.last_task_id = []
        self.last_total_service_time = 0
        self.agent_num_now = 0
        self.last_service_time = 0
        self.episode = 0
        
    def cal_heuristics(self):
        self.heuristics = {}
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)] 

        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                if self.grid[x, y] == 0:
                    self.heuristics[(x, y)] = self.bfs((x, y), directions)

    def bfs(self, start, directions):
        distances = np.full(self.grid_size, -(self.grid_size[0]+self.grid_size[1]), dtype=np.int32)
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
                    distances[neighbor] == -(self.grid_size[0]+self.grid_size[1])):

                    distances[neighbor] = current_distance + 1
                    queue.append(neighbor)

        return distances
    
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
        ]
        self.solver = PBSSolver(args)

    def generate_agents_tasks(self):
        agent_num = random.randint(self.agent_num[0], self.agent_num[1]-1)
        self.agent_num_now = agent_num
        task_frequencies = [0.2, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 50, 100, 500]
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
                release_time = int(len(tasks)/task_release_time)
            tasks.append([release_time, pickup, delivery])

        return agents, tasks, task_frequency, task_release_time
            

    def loc(self, pos):
        return int(pos/self.grid_size[1]), pos % self.grid_size[1]

    def build_state(self, status):
        if status.allFinished == 1 and status.valid == True:
            done = True
            return None, status.finished_service_time, done, True
        else:
            done = False

        if status.valid == False:
            done = True
            penalty = self.agent_num_now*(self.grid_size[0]+self.grid_size[1])*2
            return None, penalty, done, False

        self.task_id_map.clear()
        self.free_agent_id_map.clear()
        self.delivering_agent_id_map.clear()
        self.agent_task_pair.clear()

        timestep = status.timestep
        agent_task_pair = status.agent_task_pair
        self.agent_task_pair = agent_task_pair
        paths = status.solution
        paths = [[path[i].location for i in range(len(path))][timestep:] for path in paths]
    
        delivering_agents_grid = np.zeros((self.agent_num[1], 3, self.grid_size[0], self.grid_size[1]), dtype=np.float32)
        free_agents_grid = np.zeros((self.agent_num[1], 3, self.grid_size[0], self.grid_size[1]), dtype=np.float32)
        # free_agents_grid = np.repeat(self.grid, self.agent_num[1], axis=0)
        tasks_grid = np.zeros((self.task_num, 3, self.grid_size[0], self.grid_size[1]), dtype=np.float32)
        free_agents_path_length = np.zeros((self.agent_num[1], 1), dtype=np.float32)
        delivering_agents_path_length = np.zeros((self.agent_num[1], 1), dtype=np.float32)
        tasks_loc = np.zeros((self.task_num, 2), dtype=np.float32)

        delivering_agents = []
        
        free_agent_cnt = 0
        delivering_agent_cnt = 0
        locations = []
        for i in range(len(status.agents_all)):
            location = self.loc(status.agents_all[i].start_location)
            locations.append(location)

        for i in range(len(status.agents_all)):
            is_delivering = status.agents_all[i].is_delivering
            location = self.loc(status.agents_all[i].start_location)

            if is_delivering:
                delivering_agents_grid[delivering_agent_cnt, 0] = self.grid.copy()
                delivering_agents_grid[delivering_agent_cnt, 1] = self.heuristics[location].astype(np.float32)/2*(self.grid_size[0]+self.grid_size[1]) + 0.5
                self.delivering_agent_id_map[delivering_agent_cnt] = i
                for j in range(len(paths[i])):
                    ploc = self.loc(paths[i][j])
                    delivering_agents_grid[delivering_agent_cnt, 2, ploc[0], ploc[1]] = j+1
                delivering_agents_path_length[delivering_agent_cnt] = len(paths[i])-1
                delivering_agents.append(i)
                for j in range(len(locations)):
                    if locations[j] != location:
                        delivering_agents_grid[delivering_agent_cnt, 0, locations[j][0], locations[j][1]] = 1
                delivering_agents_grid[delivering_agent_cnt, 0] = (delivering_agents_grid[delivering_agent_cnt, 0] + 1) / 2
                delivering_agents_grid[delivering_agent_cnt, 2] /= self.grid_size[0] + self.grid_size[1]
                delivering_agent_cnt += 1
            else:
                free_agents_grid[free_agent_cnt, 0] = self.grid.copy()
                free_agents_grid[free_agent_cnt, 1] = self.heuristics[location].astype(np.float32)/2*(self.grid_size[0]+self.grid_size[1]) + 0.5
                self.free_agent_id_map[free_agent_cnt] = i
                for j in range(len(locations)):
                    if locations[j] != location:
                        free_agents_grid[free_agent_cnt, 0, locations[j][0], locations[j][1]] = 1
                free_agents_grid[free_agent_cnt, 0] = (free_agents_grid[free_agent_cnt, 0] + 1) / 2
                free_agents_grid[free_agent_cnt, 2,location[0], location[1]] = 1
                free_agents_grid[free_agent_cnt, 2] /= self.grid_size[0] + self.grid_size[1]
                free_agents_path_length[free_agent_cnt] = len(paths[i])-1
                free_agent_cnt += 1

        delivering_tasks = [item[1][0] for item in agent_task_pair.items() if item[0] in delivering_agents] 

        delivering_reward = 0
        free_reward = 0

        task_cnt = 0
        for task in status.tasks:
            task_id = task.task_id
            pickup, delivery = task.goal_arr[:2]
            if task_id not in delivering_tasks:
                pickup = self.loc(pickup)
                pickup = pickup[0] * self.grid_size[1] + pickup[1]
                delivery = self.loc(delivery)
                delivery = delivery[0] * self.grid_size[1] + delivery[1]
                release_time = task.release_time
                self.task_id_map[task_cnt] = task_id
                tasks_loc[task_cnt] = np.array([pickup, delivery], dtype=np.float32)
                
                tasks_grid[task_cnt, 0] = self.heuristics[self.loc(pickup)].astype(np.float32)/2*(self.grid_size[0]+self.grid_size[1]) + 0.5
                tasks_grid[task_cnt, 1] = self.heuristics[self.loc(delivery)].astype(np.float32)/2*(self.grid_size[0]+self.grid_size[1]) + 0.5
                tasks_grid[task_cnt, 2] = self.grid.copy()
                for j in range(len(locations)):
                    tasks_grid[task_cnt, 0, locations[j][0], locations[j][1]] = 1
                tasks_grid[task_cnt, 0] = (tasks_grid[task_cnt, 0] + 1) / 2

                task_cnt += 1

                if task_id in self.last_task_id:
                    free_reward += task.estimated_service_time

        delivering_reward = status.delivering_service_time

        finished_service_time = status.finished_service_time
        reward = finished_service_time + delivering_reward + free_reward

        # print("delivering_agents", list(self.delivering_agent_id_map.keys()))
        sys.stdout.flush()

        self.reverse_task_id_map = {v: k for k, v in self.task_id_map.items()}

        # 获取专家动作并转换为当前任务ID映射
        expert_actions = []
        for agent_key, agent_id in sorted(self.free_agent_id_map.items()):
            assert agent_key == len(expert_actions)
            if len(status.agent_task_sequences[agent_id]) > 0:
                mapped_tasks = [self.reverse_task_id_map.get(t, -1) for t in status.agent_task_sequences[agent_id]]
                expert_actions.append(mapped_tasks[0])
            else:
                expert_actions.append(task_cnt)

        expert_actions.append(task_cnt)
        
        expert_actions_padded = np.full((self.agent_num[1]+1), -100, dtype=np.int32)
        for i in range(len(expert_actions)):
            expert_actions_padded[i] = expert_actions[i]
        
        # if task_cnt == 0:
        #     breakpoint()

        if not self.training:
            print("id:", self.seed, "free_agents", self.free_agent_id_map.values(), "delivering_agents:", self.delivering_agent_id_map.values(), "delivering_tasks:", delivering_tasks, "free_reward:", free_reward, "delivering_reward:",delivering_reward, "finished_service_time:", finished_service_time)
            print("expert_actions:", status.agent_task_sequences)
        return {
            "free_agents_grid": free_agents_grid,
            "delivering_agents_grid": delivering_agents_grid,
            "free_agents_path_length": free_agents_path_length,
            "delivering_agents_path_length": delivering_agents_path_length,
            "tasks_grid": tasks_grid,
            "tasks_loc": tasks_loc,
            "free_agents_num": free_agent_cnt,
            "delivering_agents_num": delivering_agent_cnt,
            "tasks_num": task_cnt,
            "expert_actions": expert_actions_padded
        }, reward, done, True

    def decode_action(self, action):
        action = action[:self.agent_num_now]
        penalty = 0
        avail_task = 0
        # if action has two or more agents share a same task, then add a penalty and only remain the first one's task
        action_list = action.tolist()
        free_agents_num = len(self.free_agent_id_map)
        delivering_num = len(self.delivering_agent_id_map)
        agent_tasks = [[] for i in range(free_agents_num + delivering_num)]

        assigned_task = []
        # print(action_list)
        # print(self.task_id_map)
        # print(self.free_agent_id_map)
        for k, v in self.free_agent_id_map.items():
            if action_list[k] in self.task_id_map:
                task_id = self.task_id_map[action_list[k]]
                if task_id != -1:
                    if task_id not in assigned_task:
                        agent_tasks[v] = [task_id]
                        assigned_task.append(task_id)
                        avail_task += 1
                    else:
                        agent_tasks[v] = []
                        penalty += 2*(self.grid_size[0]+self.grid_size[1])
        
        for k, v in self.agent_task_pair.items():
            if k in self.delivering_agent_id_map.values():
                agent_tasks[k] = [v[0]]

        self.last_task_id.clear()
        for agent_task in agent_tasks:
            for t in agent_task:
                self.last_task_id.append(t)

        # if avail_task == 0:
        #     agent_tasks[0] = [self.task_id_map[0]]
        # breakpoint()
        # for agent_task in agent_tasks:
        #     if -1 in agent_task:
        #         breakpoint()
        if not self.training:
            print("id:", self.seed, "agent_tasks:",agent_tasks)
            sys.stdout.flush()
        return agent_tasks, penalty

    def reset(self, seed=40):
        self.last_total_service_time = 0
        self.agent_num_now = 0
        self.last_service_time = 0
        self.last_task_id = []
        self.task_id_map = {}
        self.free_agent_id_map = {}
        self.delivering_agent_id_map = {}
        self.agent_task_pair = {}
        self.cal_heuristics()

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

        
        if self.training:
            status = self.solver.update_task(tasks, agents, 5000, task_frequency, task_release_time)
        else:
            status = self.solver.update_task(tasks, [], 5000, task_frequency, task_release_time)
            self.agent_num_now = len(status.agents_all)

        # print(self.agent_num_now)
        # construct state
        self.step_count = 0
        self.state, _, _, _ = self.build_state(status)

        while (self.state['tasks_num'] == 0 or self.state['free_agents_num'] == 0):
            agent_tasks, _ = self.decode_action(np.full((self.agent_num[1]+1), -100, dtype=np.int32))
            status = self.solver.update(agent_tasks)
            self.state, _,_,_ = self.build_state(status)

        if self.training:
            return self.state, {}
        else:
            return self.state

    def step(self, action):
        agent_tasks, penalty = self.decode_action(action)
        status = self.solver.update(agent_tasks)
        self.state, service_time, done, valid = self.build_state(status)
        
        # while there is no task or no free agent, update with empty action to keep runnning
        while not done and (self.state['tasks_num'] == 0 or self.state['free_agents_num'] == 0):
            # if self.state['free_agents_num'] == 0:
            #     breakpoint()
            agent_tasks, penalty = self.decode_action(np.full((self.agent_num[1]+1), -100, dtype=np.int32))
            status = self.solver.update(agent_tasks)
            self.state, service_time, done, valid = self.build_state(status)
        
        if not done:
            assert self.state['tasks_num'] > 0
        
        self.step_count += 1

        # normalize reward
        s_time = service_time - self.last_service_time
        self.last_service_time = service_time
        
        if valid:
            reward = -(s_time+ penalty)/((self.grid_size[0]+self.grid_size[1])*self.agent_num_now)
        else:
            reward = -1
        if self.pos_reward:
            reward += 1

        if done:
            reward += 20
            self.episode += 1

        print("id:", self.seed, "reward:",reward, "service_time:", self.last_service_time, "s_time:", s_time, "penalty:", penalty, "agent_num:", self.agent_num_now, "done:", done)
        print("______________________")
        sys.stdout.flush()
        # if done:
        #     breakpoint()
        # if self.step_count >= self.max_steps:
        #     done = True
        if self.training:
            return self.state, reward, done, False, {}
        else:
            if done:
                return self.state, service_time, done, False, {}
            else:
                return self.state, 0, done, False, {}
