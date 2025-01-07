#pragma once
#include "BasicSystem.h"
#include "KivaGraph.h"
#include "LNS.h"
// #include "AgentsLoader.h"
// #include "States.h"
// #include "TasksLoader.h"
#include <chrono>
#include <vector>
#include <stdlib.h>
#include <ctime>
#include <cstdlib>
using namespace std::chrono;
typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::duration<float> fsec;

class AgentTaskStatus
{
	public:
		std::map<int, Task> tasks;
		vector<int> delivering_tasks;
		vector<Agent> agents_all;
		vector<Path> solution;
		int allFinished;
		std::map<int, pair<int,int>> agent_task_pair;
		bool valid;
		int finished_service_time;
		AgentTaskStatus(const std::map<int, Task>& tasks, const vector<int>& delivering_tasks, 
			const vector<Agent>& agents_all, const vector<Path>& solution,
			const std::map<int, pair<int,int>>& agent_task_pair, int finished_service_time, int allFinished): 
			tasks(tasks), delivering_tasks(delivering_tasks), agents_all(agents_all), 
			solution(solution), agent_task_pair(agent_task_pair), finished_service_time(finished_service_time), 
			allFinished(allFinished){this->valid=true;}

		AgentTaskStatus(int finished_service_time, int allFinished):finished_service_time(finished_service_time), allFinished(allFinished){this->valid=true;}

		AgentTaskStatus(){this->valid=false;}
};

class KivaSystemOnline :
	public BasicSystem
{
public:
	KivaSystemOnline(KivaGrid& G, MAPFSolver& solver);
	~KivaSystemOnline();

	void simulate(int simulation_time);
	void simulate(const vector<vector<int>>& goal_sequences);
	bool load_tasks(const vector<vector<int>>& tasks, vector<int>& new_agents, int simulation_time, float task_frequency, int task_release_period);
	int get_makespan();
	int get_flowtime() const;
	int calculate_flowtime_tp(vector<vector<int>> finish_task_sequence);
	vector<int> path_len;
	vector<int> newly_finished_agents_idx;
	int total_release_time = 0;
	int num_finished_tasks = 0;
	double flowtime_init_tp = 0;
	vector<vector<int>> agents_delivery_loc;
	vector<vector<int>> agents_pickup_loc;
	vector<vector<Task>> agents_task_sequences;
	vector<vector<int>> agents_finish_sequence;
	vector<Task> all_tasks;
	std::map<int, vector<int>> all_tasks_list;
	std::map<int, Task> current_tasks;
	int task_num;
	bool finish_release = false;
	std::map<int, vector<int>> delivering_agents;
	std::map<int, pair<int,int>> agent_task_pair;
	vector<int> free_agent_set;
	bool all_agents_busy=false;
	bool finish_assign = false;
	bool new_agent_finish = false;
	vector<int> current_assigned_endpoints;
	bool apply_lns = true;
	bool deferred_task = false;
	int node_expanded = 0;
	std::map<int, Task> get_initial_tasks();
	AgentTaskStatus simulate_until_next_assignment(const vector<vector<int>>& agent_task_sequences);
	int mkspan;
	int fltime;
	int fltime_tp;
	int finished_release_time;
	int last_plan_timestep;
	int task_plan_time;

private:
	KivaGrid& G;
	vector<vector<int> > task_sequences;

	void initialize(int simulation_time);
	void initialize_start_locations();
	void initialize_goal_locations();
	int choose_good_endpoint(vector<int> current_assigned_endpoints, int last_task_endpoint);
	void update_goal_locations();
	void generate_tasks();
	void remove_from_system();
	void update_goal_locations(std::map<int, int> delivering_agents, std::map<int, int> agent_task_pair);
	bool move_after_assignment();
	void update_agent_tasks(const vector<vector<int>>& agent_tasks);
	AgentTaskStatus get_agent_tasks();
	void estimate_service_time();

};	

