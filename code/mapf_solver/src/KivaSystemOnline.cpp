#include "KivaSystemOnline.h"
#include "PBS.h"
#include <random>
#include <csignal>
#include <iostream>
#include <fstream>
#include <ctime>
#include <sstream>
#include <iomanip>
#include <algorithm>


KivaSystemOnline::KivaSystemOnline(KivaGrid& G, MAPFSolver& solver): BasicSystem(G, solver), G(G) {}

// 拷贝构造函数实现
KivaSystemOnline::KivaSystemOnline(const KivaSystemOnline& other) 
    : BasicSystem(other), G(other.G) 
{
    // 手动复制all_tasks，确保Task对象正确拷贝
    all_tasks.clear();
    for (const auto& task : other.all_tasks) {
        // 创建新的Task对象，只复制基本数据，不复制复杂的heap数据结构
        Task new_task(task.task_id, task.release_time, const_cast<vector<int>&>(task.goal_arr));
        new_task.delta_cost = task.delta_cost;
        new_task.is_delivered = task.is_delivered;
        new_task.relatedness = task.relatedness;
        new_task.pick_up_time = task.pick_up_time;
        new_task.delivery_time = task.delivery_time;
        new_task.estimated_service_time = task.estimated_service_time;
        new_task.estimated_finish_time = task.estimated_finish_time;
        // 注意：不复制ta和assignment_heap，因为它们可能包含无效的handle
        all_tasks.push_back(new_task);
    }
    
    // 复制all_tasks_list
    all_tasks_list = other.all_tasks_list;
    
    // 深拷贝current_tasks中的Task对象
    current_tasks.clear();
    for (const auto& pair : other.current_tasks) {
        const Task& task = pair.second;
        Task new_task(task.task_id, task.release_time, const_cast<vector<int>&>(task.goal_arr));
        new_task.delta_cost = task.delta_cost;
        new_task.is_delivered = task.is_delivered;
        new_task.relatedness = task.relatedness;
        new_task.pick_up_time = task.pick_up_time;
        new_task.delivery_time = task.delivery_time;
        new_task.estimated_service_time = task.estimated_service_time;
        new_task.estimated_finish_time = task.estimated_finish_time;
        current_tasks.insert(std::make_pair(pair.first, new_task));
    }
    
    // 深拷贝agents_task_sequences中的Task对象
    agents_task_sequences.clear();
    agents_task_sequences.resize(other.agents_task_sequences.size());
    for (size_t i = 0; i < other.agents_task_sequences.size(); ++i) {
        agents_task_sequences[i].clear();
        for (const auto& task : other.agents_task_sequences[i]) {
            Task new_task(task.task_id, task.release_time, const_cast<vector<int>&>(task.goal_arr));
            new_task.delta_cost = task.delta_cost;
            new_task.is_delivered = task.is_delivered;
            new_task.relatedness = task.relatedness;
            new_task.pick_up_time = task.pick_up_time;
            new_task.delivery_time = task.delivery_time;
            new_task.estimated_service_time = task.estimated_service_time;
            new_task.estimated_finish_time = task.estimated_finish_time;
            agents_task_sequences[i].push_back(new_task);
        }
    }
    
    // 复制其他简单类型成员变量
    task_num = other.task_num;
    finish_release = other.finish_release;
    delivering_agents = other.delivering_agents;
    agent_task_pair = other.agent_task_pair;
    free_agent_set = other.free_agent_set;
    all_agents_busy = other.all_agents_busy;
    finish_assign = other.finish_assign;
    new_agent_finish = other.new_agent_finish;
    current_assigned_endpoints = other.current_assigned_endpoints;
    apply_lns = other.apply_lns;
    deferred_task = other.deferred_task;
    node_expanded = other.node_expanded;
    mkspan = other.mkspan;
    fltime = other.fltime;
    fltime_tp = other.fltime_tp;
    finished_release_time = other.finished_release_time;
    last_plan_timestep = other.last_plan_timestep;
    task_plan_time = other.task_plan_time;
    
    // 复制BasicSystem中遗漏的重要成员变量
    total_num_of_tasks = other.total_num_of_tasks;
    task_frequency = other.task_frequency;
    task_release_period = other.task_release_period;
    simulation_time = other.simulation_time;
    num_of_drives = other.num_of_drives;
    timestep = other.timestep;
    seed = other.seed;
    time_limit = other.time_limit;
    planning_window = other.planning_window;
    simulation_window = other.simulation_window;
    neighborhood_size = other.neighborhood_size;
    task_truncated_size = other.task_truncated_size;
    candidate_task_k = other.candidate_task_k;
    use_LNS = other.use_LNS;
    REPLAN = other.REPLAN;
    look_ahead_horizon = other.look_ahead_horizon;
    consider_rotation = other.consider_rotation;
    k_robust = other.k_robust;
    hold_endpoints = other.hold_endpoints;
    useDummyPaths = other.useDummyPaths;
    travel_time_window = other.travel_time_window;
    screen = other.screen;
    log = other.log;
    
    // 复制其他向量成员
    path_len = other.path_len;
    newly_finished_agents_idx = other.newly_finished_agents_idx;
    total_release_time = other.total_release_time;
    num_finished_tasks = other.num_finished_tasks;
    flowtime_init_tp = other.flowtime_init_tp;
    agents_delivery_loc = other.agents_delivery_loc;
    agents_pickup_loc = other.agents_pickup_loc;
    agents_finish_sequence = other.agents_finish_sequence;
    task_sequences = other.task_sequences;
    
    // 复制BasicSystem中的向量成员
    free_agents = other.free_agents;
    path_planning_timestep = other.path_planning_timestep;
    task_planning_time_list = other.task_planning_time_list;
    path_planning_time_list = other.path_planning_time_list;
    throughput_per_timestep = other.throughput_per_timestep;
    throughput_accumulate = other.throughput_accumulate;
    agents_finish_task_goal_arr = other.agents_finish_task_goal_arr;
    new_agents = other.new_agents;
    assigned_agents = other.assigned_agents;
    starts = other.starts;
    goal_locations = other.goal_locations;
    paths = other.paths;
    finished_tasks = other.finished_tasks;
    remained_agents = other.remained_agents;
    goal_lens = other.goal_lens;
    num_of_tasks = other.num_of_tasks;
    outfile = other.outfile;
    saving_time = other.saving_time;
}

KivaSystemOnline::~KivaSystemOnline()
{
}

bool KivaSystemOnline::load_tasks(vector<vector<int>>& tasks, vector<int>& new_agents, int simulation_time, float task_frequency, int task_release_period) 
{
	this->task_frequency = task_frequency;
	this->task_release_period = task_release_period;
    all_tasks.clear();
    all_tasks_list.clear();
    total_release_time = 0;

	if (new_agents.size() > 0)
	{
		G.update_agents(new_agents);
		num_of_drives = new_agents.size();
	}
		
	// else:
	// 	num_of_drives = G.agent_home_locations.size();

    for (size_t i = 0; i < tasks.size(); ++i) {
        int release_time = tasks[i][0];
        vector<int> arr;
		arr.clear();

		for (size_t j = 1; j < tasks[i].size(); j++)
		{
			arr.push_back(G.endpoints[tasks[i][j]]);
		}
		// if (arr.size() < 2)
		// 		cout << "wrong" << endl;

        all_tasks.push_back(Task(i + 1, release_time, arr));

        arr.push_back(release_time);	
        all_tasks_list[i + 1] = arr;

        total_release_time += release_time;
    }

    total_num_of_tasks = static_cast<int>(tasks.size());
	initialize(simulation_time);

    return true;
}

void KivaSystemOnline::initialize(int simulation_time)
{
	this->simulation_time = simulation_time;
	initialize_solvers();
	starts.clear();
	starts.resize(num_of_drives);
	goal_locations.clear();
	goal_locations.resize(num_of_drives);
	paths.clear();
	paths.resize(num_of_drives);
	finished_tasks.clear();
	finished_tasks.resize(num_of_drives);
	task_sequences.clear();
	task_sequences.resize(num_of_drives);
	agents_task_sequences.clear();
	agents_task_sequences.resize(num_of_drives);
	agents_finish_sequence.clear();
	agents_finish_sequence.resize(num_of_drives);
	agents_finish_task_goal_arr.clear();
	agents_finish_task_goal_arr.resize(num_of_drives);
	mkspan = 0;
	fltime = 0;
	fltime_tp = 0;
	finished_release_time = 0;
	last_plan_timestep = 0;
	task_plan_time = 0;
	num_finished_tasks = 0;
	num_of_tasks = 0;
	deferred_task = false;
	
	path_len.clear();
	remained_agents.clear();
	for (int i = 0; i < num_of_drives; i++) {
		path_len.push_back(0);
		remained_agents.push_back(i);
	}

	timestep = 0;
	initialize_start_locations();
	// bool succ = load_records(); // continue simulating from the records
	// if (!succ)
	// {
	// 	timestep = 0;
	// 	succ = load_locations();
	// 	if (!succ)
	// 	{
	// 		initialize_start_locations();
	// 	}
	// }
}

void KivaSystemOnline::initialize_start_locations()
{
	// Choose random start locations
	// Any non-obstacle locations can be start locations
	// Start locations should be unique
	for (int k = 0; k < num_of_drives; k++)
	{
		int orientation = -1;
		if (consider_rotation)
		{
			orientation = rand() % 4;
		}
		starts[k] = State(G.agent_home_locations[k], 0, orientation);
		paths[k].emplace_back(starts[k]);
	}
}

void KivaSystemOnline::generate_tasks()
{
	int count = 0;
	auto it = all_tasks.begin();
	while (it != all_tasks.end() && count < task_release_period * task_frequency * look_ahead_horizon)
	{
		Task& task = *it;
		bool task_released = false;
		
		// 检查任务是否应该在look_ahead_horizon内释放
		for (int horizon_step = 0; horizon_step < look_ahead_horizon; horizon_step++)
		{
			if (task.release_time == timestep + horizon_step * task_release_period)
			{
				// pending_task_cap 仅约束 free pending tasks（不包含delivering tasks）。
				// 超上限时丢弃新释放任务，避免pending持续膨胀。
				bool should_drop = false;
				if (pending_task_cap > 0)
				{
					int free_pending = get_free_pending_task_count();
					if (free_pending >= pending_task_cap)
						should_drop = true;
				}
				if (!should_drop)
				{
					current_tasks.insert(make_pair(task.task_id, task));
				}
				it = all_tasks.erase(it);  // 删除并更新迭代器
				count++;
				task_released = true;
				break;
			}
		}
		
		// 如果任务没有被释放，移动到下一个任务
		if (!task_released)
		{
			++it;
		}
	}
	// if (count != 0)
	// 	std::cout << "Generate " << count << " new tasks " << endl;
}

std::set<int> KivaSystemOnline::get_delivering_task_ids() const
{
	std::set<int> delivering_ids;
	for (int i = 0; i < num_of_drives; i++)
	{
		if (i >= (int)task_sequences.size() || task_sequences[i].empty())
			continue;
		if (i >= (int)goal_locations.size() || goal_locations[i].empty())
			continue;

		int tid = task_sequences[i].front();
		auto cur_it = current_tasks.find(tid);
		if (cur_it == current_tasks.end())
			continue;

		const vector<int>& goal_arr = cur_it->second.goal_arr;
		if (goal_locations[i].size() <= 1 || goal_arr.size() == 1)
			continue;

		int curr_goal_loc = goal_locations[i].front().first;
		auto iter = std::find(goal_arr.begin(), goal_arr.end(), curr_goal_loc);
		if (iter != goal_arr.begin() && iter != goal_arr.end())
		{
			delivering_ids.insert(tid);
		}
	}
	return delivering_ids;
}

int KivaSystemOnline::get_free_pending_task_count() const
{
	if (current_tasks.empty())
		return 0;
	const std::set<int> delivering_ids = get_delivering_task_ids();
	const int free_cnt = (int)current_tasks.size() - (int)delivering_ids.size();
	return std::max(0, free_cnt);
}

int KivaSystemOnline::choose_good_endpoint(vector<int> current_assigned_endpoints, int last_task_endpoint)
{
	std::map<int, int> distance;
	distance.clear();
	for (int i = 0; i < G.endpoints.size(); i++)
	{
		if (find(current_assigned_endpoints.begin(), current_assigned_endpoints.end(), G.endpoints[i]) != current_assigned_endpoints.end())
			continue;
		// if (G.endpoints[i] == last_task_endpoint)
		// 	continue;
		// int dist = G.get_Manhattan_distance(last_task_endpoint, G.endpoints[i]);
		int dist = G.heuristics.at(G.endpoints[i])[last_task_endpoint];
		distance[dist] = G.endpoints[i];
	}
	if (!distance.empty())
	{
		return distance.begin()->second;
	}
	else
	{
		return -1;
		// for (int i = 0; i < G.agent_home_locations.size(); i++)
		// {
		// 	if (find(current_assigned_endpoints.begin(), current_assigned_endpoints.end(), G.agent_home_locations[i]) != current_assigned_endpoints.end())
		// 		continue;
		// 	if (G.agent_home_locations[i] == last_task_endpoint)
		// 		continue;
		// 	// int dist = G.get_Manhattan_distance(last_task_endpoint,  G.agent_home_locations[i]);
		// 	int dist = G.heuristics.at(last_task_endpoint)[G.agent_home_locations[i]];
		// 	distance[dist] = G.agent_home_locations[i];
		// }
		// return distance.begin()->second;
	}
}

int KivaSystemOnline::calculate_flowtime_tp(vector<vector<int>> finish_task_sequence)
{
	int res = 0;
	int flowtime = 0;
	int makespan = 0;
	int cnt = 0;
	std::set<int> total;
	for (int l = 0; l < finish_task_sequence.size(); l++)
	{
		vector<int> i = finish_task_sequence[l];
		for (int t = 0 ; t < i.size(); t++)
		{
			total.insert(i[t]);
			vector<int> arr = all_tasks_list[i[t]];
			if (t == 0)
				res = G.get_Manhattan_distance(G.agent_home_locations[l], arr[0]);
			res = max(res, arr[arr.size()-1]);
			// last element of arr is the release time of task
			for (int j = 0; j < arr.size()-2; j++)
			{
				res += G.get_Manhattan_distance(arr[j], arr[j+1]);
			}
			cnt++;
			flowtime += res;
			makespan = max(makespan, res);
			if (t != i.size()-1)
				res += G.get_Manhattan_distance(arr[arr.size()-2], all_tasks_list[i[t+1]][0]);
		}
	}
	return flowtime;
}

std::map<int, Task> KivaSystemOnline::get_initial_tasks()
{
	generate_tasks();
	agents_finish_task_goal_arr.resize(num_of_drives);
	mkspan = 0;
	fltime = 0;
	fltime_tp = 0;
	last_plan_timestep = 0;
	task_plan_time = 0;
	return current_tasks;
}

bool KivaSystemOnline::move_after_assignment()
{
	auto new_finished_tasks = move();
	int old = num_of_tasks;

	int prev_finish_tasks = num_finished_tasks;

	for (auto task : new_finished_tasks)
	{
		int id, loc, t;
		std::tie(id, loc, t) = task;
		if (find(G.agent_home_locations.begin(), G.agent_home_locations.end(), loc) != G.agent_home_locations.end())
			continue;
		finished_tasks[id].emplace_back(loc, t);
		num_of_tasks++;
		if (agents_task_sequences[id].empty())
			continue;
		vector<int> curr_task_goal = agents_task_sequences[id].front().goal_arr;
		int num_of_curr_task_goal = curr_task_goal.size();
		int left_ptr = num_of_curr_task_goal-1;
		int right_ptr = finished_tasks[id].size()-1;
		int start = 0;
		while (start < num_of_curr_task_goal)
		{
			list<Key>::iterator it = finished_tasks[id].begin();
			std::advance(it, right_ptr--);
			if (curr_task_goal[left_ptr--] == it->first)
				start++;
			else
				break;	
		}
		// this task could be a pickup loc, or some dummy loc
		if (start == num_of_curr_task_goal)
		{
			agents_finish_task_goal_arr[id].push_back(curr_task_goal);
			fltime += t;
			mkspan = std::max(t, mkspan);
			num_finished_tasks++;
			auto it = current_tasks.find(task_sequences[id].front());
			Task finished_task = it->second;
			finished_release_time += finished_task.release_time;
			current_tasks.erase(current_tasks.find(task_sequences[id].front()));
			agents_finish_sequence[id].push_back(task_sequences[id].front());
			task_sequences[id].erase(task_sequences[id].begin());
			agents_task_sequences[id].erase(agents_task_sequences[id].begin());
		}
	}

	// if (screen > 0)
	// {
	// 	std::cout << num_of_tasks - old << " goals just finished" << std::endl;
	// 	std::cout << num_of_tasks << " goals finished in total" << std::endl;
	// 	std::cout << num_finished_tasks << " tasks finished in total" << std::endl;
	// }
	throughput_accumulate.push_back(num_finished_tasks);
	if (num_finished_tasks == total_num_of_tasks) {
		return false;
	}
	int curr_finish_tasks = num_finished_tasks - prev_finish_tasks;
	throughput_per_timestep.push_back(curr_finish_tasks);
	return true;
}


void KivaSystemOnline::update_agent_tasks(const vector<vector<int>>& agent_tasks)
{	
	for (int i = 0; i < num_of_drives; i++)
	{
		// if (agent_tasks.size() <= i || task_sequences.size() <= i)
		// {
		// 	raise(SIGTRAP);
		// }
		if (i < static_cast<int>(agent_tasks.size())) {
			task_sequences[i] = agent_tasks[i];
		} else {
			task_sequences[i].clear();
		}
		int idx = remained_agents[i];
		int current_task_size = 0;
		if (delivering_agents.find(i) != delivering_agents.end())
		{
			int task_id = agent_task_pair[i].first;
			int task_idx = agent_task_pair[i].second;
			// if (REPLAN && !new_agent_finish)
			if ((!new_agent_finish) || (new_agent_finish && !current_tasks.empty()))
				task_sequences[i].insert(task_sequences[i].begin(), task_id);
			int release_time = current_tasks[task_id].release_time;
			for (int idx = task_idx; idx < current_tasks[task_id].goal_arr.size(); idx++)
			{
				int loc = current_tasks[task_id].goal_arr[idx];
				goal_locations[i].push_back(make_pair(loc, release_time));
			}
			agents_task_sequences[i].push_back(current_tasks[task_id]);
			current_task_size++;
		}
		for (int j = 0; j < task_sequences[i].size(); j++)
		{
			if (current_task_size >= task_truncated_size)
				break;
			int task_id = task_sequences[i][j];

			if (task_id == agent_task_pair[i].first) {
				continue;
			}
			int release_time = current_tasks[task_id].release_time;
			for (int idx = 0; idx < current_tasks[task_id].goal_arr.size(); idx++)
			{
				int loc = current_tasks[task_id].goal_arr[idx];
				goal_locations[i].push_back(make_pair(loc, release_time));
			}
			agents_task_sequences[i].push_back(current_tasks[task_id]);
			current_task_size++;
		}
		if (goal_locations[i].empty()) {
			free_agents.push_back(i);
		}		
	}

	// if (current_tasks.size() > num_of_drives - free_agents.size() && free_agents.size() > 0 && !deferred_task) {
	// 	printf("existing more free agents than tasks, assignment is wrong\n");
	// 	// raise(SIGTRAP);
	// 	printf("current tasks:\n");
	// 	for (auto it = current_tasks.begin(); it != current_tasks.end(); it++) {
	// 		printf("task %d: release time %d, goal arr: ", it->first, it->second.release_time);
	// 		for (int j = 0; j < it->second.goal_arr.size(); j++) {
	// 			printf("%d ", it->second.goal_arr[j]);
	// 		}
	// 		printf("\n");
	// 	}
	// 	printf("free agents:\n");
	// 	for (int i = 0; i < free_agents.size(); i++) {
	// 		printf("agent %d\n", free_agents[i]);
	// 	}
	// 	printf("num of drives: %d\n", num_of_drives);
	// 	printf("agents task sequences:\n");
	// 	for (int i = 0; i < num_of_drives; i++) {
	// 		printf("agent %d: task %d\n", i, (agents_task_sequences[i].size() > 0) ? agents_task_sequences[i][0].task_id : -1);
	// 	}
	// }

	
	
	// Collect all task endpoints
	for (auto itr = current_tasks.begin();itr != current_tasks.end(); itr++)
    {
        Task task = itr->second;
        int task_id = itr->first;
        int i = 0;
        for (; i < task.goal_arr.size(); i++)
        {
            if (find(current_assigned_endpoints.begin(), current_assigned_endpoints.end(), task.goal_arr[i]) != current_assigned_endpoints.end()) {
                continue;
            }
			current_assigned_endpoints.push_back(task.goal_arr[i]);
        }
    }

	// // Assign endpoints to non-free agents
	for (int i = 0; i < num_of_drives; i++)
	{
		if (goal_locations[i].size() == 0)
			continue;
		int loc = choose_good_endpoint(current_assigned_endpoints, goal_locations[i][goal_locations[i].size()-1].first);
		if (loc == -1)
			loc = G.agent_home_locations[i];
		goal_locations[i].push_back(make_pair(loc, 0));
		current_assigned_endpoints.push_back(loc);
	}

	// Assign endpoints to free agents
	for (int k : free_agents)
	{
		int loc = choose_good_endpoint(current_assigned_endpoints, starts[k].location);
		if (loc == -1)
			loc = G.agent_home_locations[k];
		current_assigned_endpoints.push_back(loc);
		goal_locations[k].push_back(make_pair(loc, 0));
	}

	// remember old endpoints
	current_assigned_endpoints.clear();
	for (int i = 0; i < num_of_drives; i++)
	{
		int	loc = goal_locations[i][goal_locations[i].size()-1].first;
		current_assigned_endpoints.push_back(loc);
	}

	// for (int i = 0; i < num_of_drives; i++){
	// 	if (task_sequences[i].size() > 0 && (task_sequences[i][0] != agents_task_sequences[i][0].task_id)){
	// 		raise(SIGTRAP);
	// 	}
	// }
	
	// for (int i = 0; i < num_of_drives; i++)
	// {
	// 	cout << i << ": ";
	// 	cout << starts[i].location << " ";
	// 	for (int j = 0; j < goal_locations[i].size(); j++)
	// 	{
	// 		cout << goal_locations[i][j].first << ' ';
	// 	}
	// 	cout << endl;
	// }		

}

void KivaSystemOnline::check_current_tasks()
{	
	for (auto it = current_tasks.begin(); it != current_tasks.end(); ++it) 
	{
        if (it->second.goal_arr.size() < 2)
		{
			cout << current_tasks.size() << endl;
			raise(SIGTRAP);
			cout << "task wrong" << endl;
		}
    }
}

AgentTaskStatus KivaSystemOnline::get_agent_tasks()
{
	vector<int> delivering_tasks;
	// std::map<int, vector<int>> delivering_agents;
	// std::map<int, pair<int,int>> agent_task_pair;
	delivering_agents.clear();
	agent_task_pair.clear();
	new_agents.clear();
	assigned_agents.clear();
	free_agents.clear();

	for (int i = 0; i < num_of_drives; i++)
	{
		// bool is_delivering = false;
		if (task_sequences[i].empty())
		{
			goal_locations[i].clear();
			agents_task_sequences[i].clear();
			// check_current_tasks();
			continue;
		}
		else if (current_tasks.count(task_sequences[i].front()))
		{

			vector<int>::iterator iter = find(current_tasks[task_sequences[i].front()].goal_arr.begin(), 
				current_tasks[task_sequences[i].front()].goal_arr.end(), goal_locations[i].front().first);
			if ( goal_locations[i].size() > 1 && current_tasks[task_sequences[i].front()].goal_arr.size() != 1 &&  iter != current_tasks[task_sequences[i].front()].goal_arr.begin()
				&& iter != current_tasks[task_sequences[i].front()].goal_arr.end())
			{
				// if (!current_tasks.count(task_sequences[i].front()))
				// {
				// 	raise(SIGTRAP);
				// }
				int idx = iter - current_tasks[task_sequences[i].front()].goal_arr.begin();
				delivering_tasks.push_back(task_sequences[i].front());
				vector<int> goal_subarr;
				// check_current_tasks();
				while (iter != current_tasks[task_sequences[i].front()].goal_arr.end())
				{
					goal_subarr.push_back(*iter);
					iter++;
				}

				delivering_agents.insert(make_pair(i, goal_subarr));
				agent_task_pair.insert(make_pair(i, make_pair(task_sequences[i].front(), idx)));
				
			}
			// check_current_tasks();
		}
		// check_current_tasks();
		if (!new_agent_finish || (new_agent_finish && !current_tasks.empty()))
			task_sequences[i].clear();
		// if (!is_delivering)
		// 	task_sequences[i].clear();
		goal_locations[i].clear();
		agents_task_sequences[i].clear();
		// check_current_tasks();
	}

	// check_current_tasks();
	
	if (timestep == 0)
	{
		for (int i =0; i < starts.size(); i++)
			current_assigned_endpoints.push_back(G.agent_home_locations[i]);
	}

	if (((!new_agent_finish) || (new_agent_finish && !current_tasks.empty())))
	{
		// check_current_tasks();
		// printf("current tasks:\n");
		// for (auto it = current_tasks.begin(); it != current_tasks.end(); it++) {
		// 	printf("task %d: release time %d, goal arr: ", it->first, it->second.release_time);
		// 	for (int j = 0; j < it->second.goal_arr.size(); j++) {
		// 		printf("%d ", it->second.goal_arr[j]);
		// 	}
		// 	printf("\n");
		// }
		// printf("delivering tasks:\n");
		// for (int i = 0; i < delivering_tasks.size(); i++) {
		// 	printf("task %d\n", delivering_tasks[i]);
		// }
		// printf("current assigned endpoints:\n");
		// for (int i = 0; i < current_assigned_endpoints.size(); i++) {
		// 	printf("%d ", current_assigned_endpoints[i]);
		// }
		// printf("\n");
		TasksLoader tl(current_tasks, delivering_tasks, current_assigned_endpoints, deferred_task);
		AgentsLoader al(G, starts, delivering_agents, task_sequences, solver.solution);;
		// if (task_truncated_size > 1){
		// 	al = AgentsLoader(G, starts, delivering_agents, task_sequences, solver.solution, true);
		// } 
		TasksLoader tl2(current_tasks, delivering_tasks, current_assigned_endpoints, deferred_task);
		AgentsLoader al2(G, starts, delivering_agents, task_sequences, solver.solution, true);
		LNS lns(G, tl, al2, 2, 1, 2, neighborhood_size);
		lns.run_Hungarian_greedy_without_delivering(task_truncated_size, candidate_task_k);

		// printf("tasks_all:\n");
		// for (int i = 0; i < tl.tasks_all.size(); i++) {
		// 	printf("task %d: release time %d, goal arr: ", tl.tasks_all[i].task_id, tl.tasks_all[i].release_time);
		// 	for (int j = 0; j < tl.tasks_all[i].goal_arr.size(); j++) {
		// 		printf("%d ", tl.tasks_all[i].goal_arr[j]);
		// 	}
		// 	printf("\n");
		// }
		// lns.run_Hungarian_greedy();
		
		// check_current_tasks();
		// cout<<starts<<endl;
		// cout<<num_of_drives<<endl;
		// return currernt_tasks, delivering_tasks, al.agents_all, solver.solution

		int delivering_service_time = 0;
		int delivering_finish_time = 0;
		for(int ti = 0; ti < delivering_tasks.size(); ti++)
		{
			int task_id = delivering_tasks[ti];
			delivering_service_time += current_tasks[task_id].estimated_service_time;
			delivering_finish_time += current_tasks[task_id].estimated_finish_time;
		}

		// if (tl.tasks_all.size() == 0){
		// 	raise(SIGTRAP);
		// }
			
		
		// 避免将TasksLoader内部含heap/handle的Task对象直接跨层拷贝到状态返回中，
		// 仅保留Python侧实际需要的轻量字段，降低悬挂handle导致的段错误风险。
		vector<Task> status_tasks;
		status_tasks.reserve(tl.tasks_all.size());
		for (const auto& t : tl.tasks_all) {
			vector<int> goals = t.goal_arr;
			Task light_t(t.task_id, t.release_time, goals);
			light_t.pick_up_time = t.pick_up_time;
			light_t.delivery_time = t.delivery_time;
			light_t.estimated_service_time = t.estimated_service_time;
			light_t.estimated_finish_time = t.estimated_finish_time;
			light_t.is_delivered = t.is_delivered;
			status_tasks.push_back(light_t);
		}

		vector<Task> status_delivering_tasks;
		status_delivering_tasks.reserve(tl.delivering_tasks_all.size());
		for (const auto& t : tl.delivering_tasks_all) {
			vector<int> goals = t.goal_arr;
			Task light_t(t.task_id, t.release_time, goals);
			light_t.pick_up_time = t.pick_up_time;
			light_t.delivery_time = t.delivery_time;
			light_t.estimated_service_time = t.estimated_service_time;
			light_t.estimated_finish_time = t.estimated_finish_time;
			light_t.is_delivered = t.is_delivered;
			status_delivering_tasks.push_back(light_t);
		}

		AgentTaskStatus status = AgentTaskStatus(
			status_tasks, status_delivering_tasks, al.agents_all, 
			paths, agent_task_pair, fltime,
			fltime-finished_release_time, delivering_service_time, 
			timestep, delivering_finish_time, 0,
			task_sequences  
		); 
		status.task_truncated_size = task_truncated_size;
		status.num_finished_tasks = num_finished_tasks;
		status.time_limit_reached = false;
		// for (int pp = 0; pp < paths.size();pp ++)
		// {
		// 	if (paths[pp].size() <= timestep)
		// 		cout << "wrong" << endl;
		// }

		// for (int tp = 0; tp < tl.tasks_all.size(); tp++)
		// {
		// 	if (tl.tasks_all[tp].goal_arr.size() < 2)
		// 	{
		// 		cout << current_tasks[tl.tasks_all[tp].task_id].goal_arr.size() << endl;
		// 		cout << "wrong" << endl;
		// 	}
				
		// }
		return status;
	}
	return AgentTaskStatus();
}

void KivaSystemOnline::estimate_service_time()
{
	for (auto it = current_tasks.begin(); it != current_tasks.end(); it++){
		it->second.estimated_service_time = 0;
		it->second.estimated_finish_time = 0;
	}
	for (int i = 0; i < num_of_drives; i++)
	{
		for (int j = 0; j < agents_task_sequences[i].size(); j++)
		{
			int task_id = agents_task_sequences[i][j].task_id;
			vector<int> goal_arr = current_tasks[task_id].goal_arr;
			int pickup = goal_arr[0];
			int delivery = goal_arr[1];

			bool ondelivery = false;
			for (int k = 0; k < solver.solution[i].size(); k++)
			{
				if (solver.solution[i][k].location == pickup)
				{
					ondelivery = true;
				}		
				if (solver.solution[i][k].location == delivery)
				{
					current_tasks[task_id].estimated_service_time = k + timestep - current_tasks[task_id].release_time;
					current_tasks[task_id].estimated_finish_time =  k + timestep;
					if (ondelivery){break;}
				}
			}
		}
	}
}

void write_log()
{
	
}


AgentTaskStatus KivaSystemOnline::simulate_until_next_assignment(const vector<vector<int>>& agent_tasks)
{
	if(timestep != 0 || agent_tasks.size()>0)
	{
		update_agent_tasks(agent_tasks);
		// check_current_tasks();
		if(!solve())
		{
			time_t now = time(nullptr);
			std::tm* localTime = std::localtime(&now);

			std::string filename = std::to_string(now) + ".txt";

			string filePath = "/local-scratchg/yifan/2024/MAPD/MAPD_RL/code/mapf_solver/log/" + filename;
			cout << filePath << endl;
			std::ofstream outFile(filePath);

			// if (outFile)
			// {
			if (!outFile)
			{
				cout << "outFile failed" << endl;
			}
			outFile << "Goal locations:" << endl;
			for (int i = 0; i < num_of_drives; i++)
			{
				outFile << i << ": ";
				outFile << starts[i].location << " ";
				for (int j = 0; j < goal_locations[i].size(); j++)
				{
					outFile << goal_locations[i][j].first << ' ';
				}
				outFile << endl;
			}	

			outFile << endl << endl << "Old Paths:" << endl;
			for (int i = 0; i < num_of_drives; i++)
			{   
				outFile << i << ": ";
				for (int j = 0; j<solver.old_paths[i].size();j++)
				{
					outFile<<solver.old_paths[i][j].location << " ";
				}
				outFile << endl;
			}
			outFile.close();
			return AgentTaskStatus();
		}

		estimate_service_time();

		if (!move_after_assignment())
		{
			AgentTaskStatus status(fltime-finished_release_time, 1);
			status.num_finished_tasks = num_finished_tasks;
			status.timestep = timestep;
			status.time_limit_reached = false;
			return status;
		}

		timestep++;
	}

	// check_current_tasks();
	

	for (; timestep < simulation_time; timestep ++)
	{
		// std::cout << "Timestep " << timestep << std::endl;

		new_agent_finish = false;

		if (all_tasks.size() != 0 && (timestep == 0 || (timestep % task_release_period == 0 && all_tasks.begin() != all_tasks.end()) ))
		{
			generate_tasks();
			// check_current_tasks();
			update_start_locations();
			// check_current_tasks();
			AgentTaskStatus status = get_agent_tasks();
			if (status.valid) 
				return status;
		}
		else
		{
			for (int k = 0; k < num_of_drives; k++)
			{
				// any non-free agent finishes their current goals
				if ((find(free_agents.begin(), free_agents.end(), k)==free_agents.end() && goal_locations[k].size() == 1))
				{
					new_agent_finish = true;
					break;
				}
			}	

			if (new_agent_finish || deferred_task)
			{	
				// check_current_tasks();
				update_start_locations();
				// check_current_tasks();
				AgentTaskStatus status = get_agent_tasks();
				deferred_task = false;
				if (status.valid) 
					return status;
			}
		}

		// check_current_tasks();

		if (!move_after_assignment())
		{
			AgentTaskStatus status(fltime-finished_release_time, 1);
			status.num_finished_tasks = num_finished_tasks;
			status.timestep = timestep;
			status.time_limit_reached = false;
			return status;
		}

		// check_current_tasks();
	}

	AgentTaskStatus status;
	status.valid = true;
	status.allFinished = 0;
	status.time_limit_reached = true;
	status.num_finished_tasks = num_finished_tasks;
	status.timestep = timestep;
	status.finished_flowtime = fltime;
	status.finished_service_time = fltime - finished_release_time;
	status.estimated_finish_time = fltime;
	status.estimated_service_time = fltime - finished_release_time;
	status.makespan = timestep;
	return status;
}