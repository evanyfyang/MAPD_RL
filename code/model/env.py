import gymnasium as gym
import numpy as np
from mapf_solver.mapf_solver import PBSSolver, Agent, Task, AgentTaskStatus
import random
from scipy.optimize import linear_sum_assignment
import sys

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
                low=-np.inf, high=np.inf, shape=(agent_num_higher_bound, self.grid_size[0], self.grid_size[1]), dtype=np.float32
            ),
            "delivering_agents_grid": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(agent_num_higher_bound, self.grid_size[0], self.grid_size[1]), dtype=np.float32
            ),
            "free_agents_loc": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(agent_num_higher_bound, 2), dtype=np.float32
            ),
            "tasks_grid": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(task_num, self.grid_size[0], self.grid_size[1]), dtype=np.float32
            ),
            "tasks_loc": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(task_num, 5), dtype=np.float32
            ),
            "free_agents_num": gym.spaces.Discrete(agent_num_higher_bound),
            "delivering_agents_num": gym.spaces.Discrete(agent_num_higher_bound),
            "tasks_num": gym.spaces.Discrete(task_num)
        })

        self.task_id_map = {}
        self.free_agent_id_map = {}
        self.delivering_agent_id_map = {}
        self.agent_task_pair = {}

        # self.observation_space = gym.spaces.Box(
        #     low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        # )
        self.action_space = gym.spaces.MultiDiscrete([task_num] * agent_num_higher_bound)
        self.eval_data_path = eval_data_path
        self.last_task_id = []
        self.last_total_service_time = 0
        self.agent_num_now = 0
        self.last_service_time = 0

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
            
            self.grid = np.expand_dims(grid_np, axis=0)

        args = [
            "--map", grid_path,          
            "--agentNum", str(self.num_r),                      
            "--seed", str(self.seed),               
            "--solver",  self.solver_name,
        ]
        self.solver = PBSSolver(args)
        # print("read grid done")

    def generate_agents_tasks(self):
        agent_num = random.randint(self.agent_num[0], self.agent_num[1]-1)
        self.agent_num_now = agent_num
        task_frequencies = [0.2, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 50, 100, 500]
        task_frequency = random.choice(task_frequencies)
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

        while len(tasks) < self.task_num:
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
            return None, status.finished_service_time, done
        else:
            done = False

        self.task_id_map.clear()
        self.free_agent_id_map.clear()
        self.delivering_agent_id_map.clear()
        self.agent_task_pair.clear()

        # print(status.agents_all)
        # print(status.tasks)
        # print(status.valid)
        timestep = status.timestep
        agent_task_pair = status.agent_task_pair
        self.agent_task_pair = agent_task_pair
        paths = status.solution
        paths = [[path[i].location for i in range(len(path))][timestep:] for path in paths]
    
        delivering_agents_grid = np.repeat(self.grid, self.agent_num[1], axis=0)
        free_agents_grid = np.repeat(self.grid, self.agent_num[1], axis=0)
        tasks_grid = np.repeat(self.grid, self.task_num, axis=0)
        free_agents_loc = np.zeros((self.agent_num[1], 2), dtype=np.float32)
        tasks_loc = np.zeros((self.task_num, 5), dtype=np.float32)

        delivering_agents = []
        
        free_agent_cnt = 0
        delivering_agent_cnt = 0
        
        for i in range(len(status.agents_all)):
            is_delivering = status.agents_all[i].is_delivering
            location = self.loc(status.agents_all[i].start_location)
            if is_delivering:
                self.delivering_agent_id_map[delivering_agent_cnt] = i
                for j in range(len(paths[i])-1):
                    ploc = self.loc(paths[i][j])
                    delivering_agents_grid[delivering_agent_cnt, ploc[0], ploc[1]] = j+1
                if (len(paths[i]) < 1):
                    breakpoint()
                ploc = self.loc(paths[i][-1])
                delivering_agents_grid[delivering_agent_cnt, ploc[0], ploc[1]] = -len(paths[i])
                delivering_agents.append(i)
                delivering_agent_cnt += 1
            else:
                self.free_agent_id_map[free_agent_cnt] = i
                free_agents_grid[free_agent_cnt, location[0], location[1]] = 1
                free_agents_loc[free_agent_cnt] = np.array(location, dtype=np.float32)
                free_agent_cnt += 1

        delivering_tasks = [item[1][0] for item in agent_task_pair.items() if item[0] in delivering_agents] 

        # delivering task will not change, so we can compute its service time
        # some free tasks are assigned in last turn, compute their service time as part of the reward
        # the free tasks which are not assigned in last turn, do not compute their service time, they 
        # are not carried by any agent and will be reassigned in this turn
        delivering_reward = 0
        free_reward = 0

        task_cnt = 0
        for task in status.tasks:
            task_id = task.task_id
            pickup, delivery = task.goal_arr[:2]
            if task_id not in delivering_tasks:
                pickup = self.loc(pickup)
                delivery = self.loc(delivery)
                release_time = task.release_time
                self.task_id_map[task_cnt] = task_id
                tasks_grid[task_cnt, pickup[0], pickup[1]] = 1
                tasks_grid[task_cnt, delivery[0], delivery[1]] = 2
                tasks_loc[task_cnt] = np.array([release_time] + list(pickup) + list(delivery), dtype=np.float32)
                task_cnt += 1

                if task_id in self.last_task_id:
                    free_reward += task.estimated_service_time

        delivering_reward = status.delivering_service_time

        # if task_cnt == 0:
        #     breakpoint()
        while task_cnt < free_agent_cnt:
            self.task_id_map[task_cnt] = -1
            task_cnt += 1

        #  calculate reward
        finished_service_time = status.finished_service_time
        reward = finished_service_time + delivering_reward + free_reward

        print("id:", self.seed, "free_agents", self.free_agent_id_map.values(), "delivering_agents:", self.delivering_agent_id_map.values(), "delivering_tasks:", delivering_tasks, "free_reward:", free_reward, "delivering_reward:",delivering_reward, "finished_service_time:", finished_service_time)
        # print("delivering_agents", list(self.delivering_agent_id_map.keys()))
        sys.stdout.flush()
        return {
            "free_agents_grid": free_agents_grid,
            "delivering_agents_grid": delivering_agents_grid,
            "free_agents_loc": free_agents_loc,
            "tasks_grid": tasks_grid,
            "tasks_loc": tasks_loc,
            "free_agents_num": free_agent_cnt,
            "delivering_agents_num": delivering_agent_cnt,
            "tasks_num": task_cnt
        }, reward, done

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
            if k in self.delivering_agent_id_map.keys():
                agent_tasks[k] = [v[0]]

        self.last_task_id.clear()
        for agent_task in agent_tasks:
            for t in agent_task:
                self.last_task_id.append(t)

        # if avail_task == 0:
        #     agent_tasks[0] = [self.task_id_map[0]]

        for agent_task in agent_tasks:
            if -1 in agent_task:
                breakpoint()
        # print(agent_tasks)
        # sys.stdout.flush()
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

        if self.training: 
            agents, tasks, task_frequency, task_release_time = self.generate_agents_tasks()
        else:
            agents = []
            with open(self.eval_data_path, 'r') as f:
                task_num, task_frequency, task_release_time = f.readline().strip().split(" ")
                task_frequency = float(task_frequency)
                task_release_time = int(task_release_time)
                lines = [line.strip().split(" ") for line in f]
                tasks = [[int(line[0]), int(line[1]), int(line[2])] for line in lines]

        # print(self.agent_num_now)
        status = self.solver.update_task(tasks, agents, 5000, task_frequency, task_release_time)

        # construct state
        self.step_count = 0
        self.state, _, _ = self.build_state(status)
        # if True:
        #     breakpoint()
        return self.state, {}

    def step(self, action):
        agent_tasks, penalty = self.decode_action(action)
        # print(agent_tasks)
        status = self.solver.update(agent_tasks)
        self.state, service_time, done = self.build_state(status)
        self.step_count += 1

        # normalize reward
        s_time = service_time - self.last_service_time
        self.last_service_time = service_time
        
        reward = -(s_time+ penalty)/((self.grid_size[0]+self.grid_size[1])*self.agent_num_now)
        if self.pos_reward:
            reward += 1

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
