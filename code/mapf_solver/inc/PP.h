#pragma once
#include "PBSNode.h"
#include "MAPFSolver.h"
#include <ctime>

// TODO: add topological sorting

class PP:
	public MAPFSolver
{
public:
    bool lazyPriority;
    bool prioritize_start;

	 // runtime breakdown
    double runtime_rt;
    double runtime_plan_paths;
    double runtime_get_higher_priority_agents;
    double runtime_copy_priorities;
    double runtime_detect_conflicts;
    double runtime_copy_conflicts;
    double runtime_choose_conflict;
    double runtime_find_consistent_paths;
    double runtime_find_replan_agents;


	PBSNode* dummy_start;
	PBSNode* best_node;

	uint64_t HL_num_expanded;
	uint64_t HL_num_generated;
	uint64_t LL_num_expanded;
	uint64_t LL_num_generated;


	double min_f_val;


	// ReservationTable initial_rt;

    list< tuple<int, int, int> > initial_constraints; // <agent, location, timestep>
    // only this agent can stay in this location during before this timestep.

	// Runs the algorithm until the problem is solved or time is exhausted 
    bool run(const vector<State>& starts,
            const vector< vector<pair<int, int> > >& goal_locations, // an ordered list of pairs of <location, release time>
            int time_limit);


    PP(const BasicGraph& G, SingleAgentSolver& path_planner);
	~PP();

    void update_paths(PBSNode* curr);
	// Save results
	void save_results(const std::string &fileName, const std::string &instanceName) const;
	void save_search_tree(const std::string &fileName) const;
	void save_constraints_in_goal_node(const std::string &fileName) const;

	string get_name() const {return "PP"; }

	void clear();

	void setRT(bool use_cat, bool prioritize_start)
	{
		rt.use_cat = use_cat;
		rt.prioritize_start = prioritize_start;
	}

private:

    std::vector< Path* > paths;
    list<PBSNode*> allNodes_table;
    list<PBSNode*> dfs;

   //  vector<State> starts;
    // vector< vector<int> > goal_locations;

    std::clock_t start;

    int num_of_agents;

    double min_sum_of_costs;
    int max_makespan;
    // int node_expanded;

	int time_limit;
	// double focal_w = 1.0;
    unordered_map<int, double> travel_times;

    unordered_set<pair<int, int>> nogood;

    vector<vector<bool> > cat; // conflict avoidance table
    vector<unordered_set< pair<int, int> > > constraint_table;
    ReservationTable rt;
    vector<double> priority;
    vector<int> priority_indices;

    // SingleAgentICBS astar;


    bool generate_paths();
    void get_priority();

	// conflicts
    void find_conflicts(const list<Conflict>& old_conflicts, list<Conflict> & new_conflicts, int new_agent);
	void find_conflicts(list<Conflict> & conflicts, int a1, int a2);
    void find_conflicts(list<Conflict> & new_conflicts, int new_agent);
    void find_conflicts(list<Conflict> & new_conflicts);

	void choose_conflict(PBSNode &parent);
	void copy_conflicts(const list<Conflict>& conflicts, list<Conflict>& copy, int excluded_agent);
    void copy_conflicts(const list<Conflict>& conflicts,
                       list<Conflict>& copy, const vector<bool>& excluded_agents);

    double get_path_cost(const Path& path) const;
	
    // update information
    //void collect_constraints(const boost::unordered_set<int>& agents, int current_agent);
    void get_solution();

    void update_CAT(int ex_ag); // update conflict avoidance table
	void update_focal_list();
	inline void release_closed_list();

	// print and save
	void print_paths() const;
	void print_results() const;
	void print_conflicts(const PBSNode &curr) const;


	// validate
	bool validate_solution();
    bool validate_consistence(const list<Conflict>& conflicts, const PriorityGraph &G) const;

};

