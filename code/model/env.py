import gymnasium as gym
import numpy as np
from ..mapf_solver.mapf_solver import PBSSolver, Agent, Task, AgentTaskStatus
import random

class MultiAgentPickupEnv(gym.Env):
    def __init__(self, training=True, obs_dim=4, action_dim=2, max_steps=50, grid_path=None, seed=40, 
            solver="PBS", agent_num_lower_bound=10, agent_num_higher_bound=50, eval_data_path=None, task_num=500):
        super().__init__()
        self.training = training
        self.solver_name = solver
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.max_steps = max_steps
        self.step_count = 0
        self.seed = seed
        self.agent_num = (agent_num_lower_bound, agent_num_higher_bound)
        self.task_num = task_num

        self.read_grid(grid_path)

        self.observation_space = gym.spaces.Dict({
            "free_agents_grid": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(agent_num_higher_bound, self.grid_size[0], self.grid_size[1]), dtype=np.float32
            ),
            "delivering_agents_grid": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(agent_num_higher_bound, self.grid_size[0], self.grid_size[1]), dtype=np.float32
            ),
            "free_agents_loc": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(agent_num_higher_bound), dtype=np.float32
            ),
            "tasks_grid": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(task_num, self.grid_size[0], self.grid_size[1]), dtype=np.float32
            ),
            "tasks_loc": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(task_num, 3), dtype=np.float32
            ),
            "free_agents_num": gym.spaces.Discrete(agent_num_higher_bound),
            "delivering_agents_num": gym.spaces.Discrete(agent_num_higher_bound),
            "tasks_num": gym.spaces.Discrete(task_num)
        })

        # self.observation_space = gym.spaces.Box(
        #     low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        # )
        # self.action_space = gym.spaces.Discrete(action_dim)
        self.eval_data_path = eval_data_path

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
        agent_num = random.randint(self.agent_num[0], self.agent_num[1])
        task_frequencies = [0.2, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 50, 100, 500]
        task_frequency = random.choice(task_frequencies)
        if task_frequency < 1:
            task_release_time = int(1/task_frequency)
        else:
            task_release_time = 1

        agents = list(random.choices(self.e_map.keys(), agent_num))
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
        return pos/self.grid_size[0], pos % self.grid_size[0]

    def build_state(self, status):
        if status.allFinished == 1 and status.valid == True:
            done = True
        else:
            done = False

        timestep = status.agents_all[0].start_timestep
        paths = status.solution
        paths = [[path[i].location for i in range(len(path))][timestep:] for path in paths]
        free_agents = [i for i in range(len(status.agents_all)) if status.agents_all[i].is_delivering == False]
        
        delivering_paths = [paths[i] for i in range(paths) if i not in free_agents]



        return None
        
    def reset(self):
        if self.training: 
            agents, tasks, task_frequency, task_release_time = self.generate_agent_tasks()
        else:
            agents = []
            with open(self.eval_data_path, 'r') as f:
                task_num, task_frequency, task_release_time = f.readline().strip().split(" ")
                task_frequency = float(task_frequency)
                task_release_time = int(task_release_time)
                lines = [line.strip().split(" ") for line in f]
                tasks = [[int(line[0]), int(line[1]), int(line[2])] for line in lines]

        status = self.solver.update_task(tasks, agents, 5000, task_frequency, task_release_time)

        # construct state
        self.step_count = 0
        self.state = np.random.randn(self.obs_dim).astype(np.float32)
        return self.state

    def step(self, action):
        self.step_count += 1
        reward = np.random.randn()  # 这里是随机示例，请替换为你的逻辑
        done = self.step_count >= self.max_steps
        next_state = np.random.randn(self.obs_dim).astype(np.float32) if not done else None
        return next_state, reward, done, {}
