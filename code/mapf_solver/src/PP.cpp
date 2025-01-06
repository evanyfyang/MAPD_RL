#include "PP.h"
#include <ctime>
#include <iostream>
#include "PathTable.h"


void PP::clear()
{
    runtime = 0;
    runtime_rt = 0;
	runtime_plan_paths = 0;
    runtime_get_higher_priority_agents = 0;
    runtime_copy_priorities = 0;
    runtime_detect_conflicts = 0;
    runtime_copy_conflicts = 0;
    runtime_choose_conflict = 0;
    runtime_find_consistent_paths = 0;
    runtime_find_replan_agents = 0;
    node_expanded = 0;

    HL_num_expanded = 0;
    HL_num_generated = 0;
    LL_num_expanded = 0;
    LL_num_generated = 0;
    solution_found = false;
    solution_cost = -2;
    min_f_val = -1;
    // focal_list_threshold = -1;
    avg_path_length = -1;
    paths.clear();
    nogood.clear();
    // focal_list.clear();
    dfs.clear();
    release_closed_list();
    starts.clear();
    goal_locations.clear();
    best_node = nullptr;
    priority.clear();
    priority_indices.clear();
}

// takes the paths_found_initially and UPDATE all (constrained) paths found for agents from curr to start
// also, do the same for ll_min_f_vals and paths_costs (since its already "on the way").
// void PP::update_paths(PBSNode* curr)
// {
//     vector<bool> updated(num_of_agents, false);  // initialized for false
// 	while (curr != nullptr)
// 	{
//         for (auto p = curr->paths.begin(); p != curr->paths.end(); ++p)
//         {
// 		    if (!updated[std::get<0>(*p)])
// 		    {
// 			    paths[std::get<0>(*p)] = &(std::get<1>(*p));
// 			    updated[std::get<0>(*p)] = true;
// 		    }
//         }
// 		curr = curr->parent;
// 	}
// }


// deep copy of all conflicts except ones that involve the particular agent
// used for copying conflicts from the parent node to the child nodes
void PP::copy_conflicts(const list<Conflict>& conflicts,
	list<Conflict>& copy, const vector<bool>& excluded_agents)
{
    clock_t t = clock();
	for (auto conflict : conflicts)
	{
		if (!excluded_agents[std::get<0>(conflict)] && !excluded_agents[std::get<1>(conflict)])
		{
			copy.push_back(conflict);
		}
	}
    runtime_copy_conflicts += (std::clock() - t) * 1.0 / CLOCKS_PER_SEC;
}
void PP::copy_conflicts(const list<Conflict>& conflicts, list<Conflict>& copy, int excluded_agent)
{
    clock_t t = clock();
    for (auto conflict : conflicts)
    {
        if (excluded_agent != std::get<0>(conflict) && excluded_agent != std::get<1>(conflict))
        {
            copy.push_back(conflict);
        }
    }
    runtime_copy_conflicts += (std::clock() - t) * 1.0 / CLOCKS_PER_SEC;
}


void PP::find_conflicts(list<Conflict>& conflicts, int a1, int a2)
{
    clock_t t = clock();
    if (paths[a1] == nullptr || paths[a2] == nullptr)
        return;
	if (hold_endpoints)
	{ 
		// TODO: add k-robust
		size_t min_path_length = paths[a1]->size() < paths[a2]->size() ? paths[a1]->size() : paths[a2]->size();
		for (size_t timestep = 0; timestep < min_path_length; timestep++)
		{
			int loc1 = paths[a1]->at(timestep).location;
			int loc2 = paths[a2]->at(timestep).location;
			if (loc1 == loc2 && G.types[loc1] != "Magic")
			{
				conflicts.emplace_back(a1, a2, loc1, -1, timestep);
				return;
			}
			else if (timestep < min_path_length - 1
				&& loc1 == paths[a2]->at(timestep + 1).location
				&& loc2 == paths[a1]->at(timestep + 1).location)
			{
				conflicts.emplace_back(a1, a2, loc1, loc2, timestep + 1); // edge conflict
				return;
			}
		}

		if (paths[a1]->size() != paths[a2]->size())
		{
			int a1_ = paths[a1]->size() < paths[a2]->size() ? a1 : a2;
			int a2_ = paths[a1]->size() < paths[a2]->size() ? a2 : a1;
			int loc1 = paths[a1_]->back().location;
			for (size_t timestep = min_path_length; timestep < paths[a2_]->size(); timestep++)
			{
				int loc2 = paths[a2_]->at(timestep).location;
				if (loc1 == loc2 && G.types[loc1] != "Magic")
				{
					conflicts.emplace_back(a1_, a2_, loc1, -1, timestep); // It's at least a semi conflict		
					return;
				}
			}
		}
	}
	else
	{
        int size1 = (int)paths[a1]->size();
		int size2 = (int)paths[a2]->size();
        int max_size = max(size1, size2);

    
        // REPLAN use the following
        for (int timestep = 0; timestep < max_size; timestep++)
        {
            int loc1 = 0, loc2 = 0;
            if (timestep <= size1 - 1)
                loc1 = paths[a1]->at(timestep).location;
            else
                loc1 = paths[a1]->at(size1 - 1).location;
			if (timestep <= size2 - 1)
                loc2 = paths[a2]->at(timestep).location;
            else
                loc2 = paths[a2]->at(size2 - 1).location;

            //  vertex collision
            if (loc1 == loc2)
		    {
				conflicts.emplace_back(a1, a2, loc1, -1, timestep); // k-robust vertex conflict
				runtime_detect_conflicts += (std::clock() - t) * 1.0 / CLOCKS_PER_SEC;
				return;
			}
            
            // edge collision
            if (timestep < size1 - 1 && timestep < size2 - 1)
            {
			    if (loc1 != loc2 && loc1 == paths[a2]->at(timestep + 1).location
						 && loc2 == paths[a1]->at(timestep + 1).location)
			    {
			    	conflicts.emplace_back(a1, a2, loc1, loc2, timestep + 1); // edge conflict
			    	runtime_detect_conflicts += (std::clock() - t) * 1.0 / CLOCKS_PER_SEC;
			    	return;
			    }
            }
        }
    }
	runtime_detect_conflicts += (std::clock() - t) * 1.0 / CLOCKS_PER_SEC;
}

void PP::find_conflicts(list<Conflict>& conflicts)
{
    for (int a1 = 0; a1 < num_of_agents; a1++)
    {
        for (int a2 = a1 + 1; a2 < num_of_agents; a2++)
        {
            find_conflicts(conflicts, a1, a2);
        }
    }
}

void PP::find_conflicts(list<Conflict>& new_conflicts, int new_agent)
{
    for (int a2 = 0; a2 < num_of_agents; a2++)
    {
        if(new_agent == a2)
            continue;
        find_conflicts(new_conflicts, new_agent, a2);
    }
}



void PP::find_conflicts(const list<Conflict>& old_conflicts, list<Conflict>& new_conflicts, int new_agent)
{
    // Copy from parent
    copy_conflicts(old_conflicts, new_conflicts, new_agent);

    // detect new conflicts
    find_conflicts(new_conflicts, new_agent);
}

// void PP::choose_conflict(PBSNode &node)
// {
//     clock_t t = clock();
// 	if (node.conflicts.empty())
// 	    return;

//     node.conflict = node.conflicts.front();

// 	/*vector<int> lower_nodes(num_of_agents, -1);
// 	 * double product = -1;
//     for (auto conflict : node.conflicts)
//     {
//         int a1 = std::get<0>(*conflict);
//         int a2 = std::get<1>(*conflict);
//         node.priorities.update_number_of_lower_nodes(lower_nodes, a1);
//         node.priorities.update_number_of_lower_nodes(lower_nodes, a2);
//         double new_product = (lower_nodes[a1] + 0.01) * (lower_nodes[a2] + 0.01);
//         if (new_product > product)
//         {
//             node.conflict = conflict;
//             product = new_product;
//         }
//         else if (new_product == product && std::get<4>(*conflict) < std::get<4>(*node.conflict)) // choose the earliest
//         {
//             node.conflict = conflict;
//         }
//     }

//     return;*/

// 	// choose the earliest
//     for (auto conflict : node.conflicts)
//     {
//         /*int a1 = std::get<0>(*conflict);
//         int a2 = std::get<1>(*conflict);
//         if (goal_locations[a1] == goal_locations[a2])
//         {
//             node.conflict = conflict;
//             return;
//         }*/
//         if (std::get<4>(conflict) < std::get<4>(node.conflict))
//             node.conflict = conflict;
//     }
//     node.earliest_collision = std::get<4>(node.conflict);

//     // choose the pair of agents with smaller indices
//     /*for (auto conflict : node.conflicts)
//     {
//         if (min(std::get<0>(*conflict), std::get<1>(*conflict)) <
//             min(std::get<0>(*node.conflict), std::get<1>(*node.conflict)) ||
//                 (min(std::get<0>(*conflict), std::get<1>(*conflict)) ==
//                  min(std::get<0>(*node.conflict), std::get<1>(*node.conflict)) &&
//                     max(std::get<0>(*conflict), std::get<1>(*conflict)) <
//                     max(std::get<0>(*node.conflict), std::get<1>(*node.conflict))))
//             node.conflict = conflict;
//     }*/

//     if (!nogood.empty())
//     {
//         for (auto conflict : node.conflicts)
//         {
//             int a1 = std::get<0>(conflict);
//             int a2 = std::get<1>(conflict);
//             for (auto p : nogood)
//             {
//                 if ((a1 == p.first && a2 == p.second) || (a1 == p.second && a2 == p.first))
//                 {
//                     node.conflict = conflict;
//                     runtime_choose_conflict += (std::clock() - t) * 1.0 / CLOCKS_PER_SEC;
//                     return;
//                 }
//             }
//         }
//     }

//     runtime_choose_conflict += (std::clock() - t) * 1.0 / CLOCKS_PER_SEC;
// }

double PP::get_path_cost(const Path& path) const
{
    double cost = 0;
    for (int i = 0; i < (int)path.size() - 1; i++)
    {
        double travel_time = 1;
        // if (i > window && travel_times.find(path[i].location) != travel_times.end())
        // if (travel_times.find(path[i].location) != travel_times.end())
        // {
        //     travel_time += travel_times.at(path[i].location);
        // }
        cost += G.get_weight(path[i].location, path[i + 1].location) * travel_time;
    }
    return cost;
}

bool PP::validate_consistence(const list<Conflict>& conflicts, const PriorityGraph &G) const
{
    for (auto conflict : conflicts)
    {
        int a1 = std::get<0>(conflict);
        int a2 = std::get<1>(conflict);
        if (G.connected(a1, a2))
            return false;
        else if (G.connected(a2, a1))
            return false;
    }
    return true;
}


bool PP::generate_paths()
{
    clock_t time = std::clock();

	// initialize paths_found_initially
	paths.resize(num_of_agents, nullptr);
    vis_goal_time.resize(num_of_agents);


    if (screen == 2)
        std::cout << "Generate root CT node ..." << std::endl;

    for (int i = 0; i < num_of_agents; i++) 
	{   
        int ag = priority_indices[i];
        initial_rt.clear();
        for (int j = 0; j < starts.size(); j++)
		{
            if (ag == j)
                continue;
			initial_rt.insertPath2CT(old_paths[j]);
		}

        boost::unordered_set<int> previous_elements;
        for (int j = 0; j < i; j++) 
        {
            previous_elements.insert(priority_indices[j]);
        }

        Path* path = new Path();
        double path_cost;
        int start_location  = starts[ag].location;
        clock_t t = std::clock();
		rt.copy(initial_rt);
        rt.build(paths, initial_constraints, previous_elements, ag, start_location);

        runtime_rt += (std::clock() - t) * 1.0 / CLOCKS_PER_SEC;
        vector< vector<double> > h_values(goal_locations[ag].size());

        t = std::clock();
        *path = path_planner.run(G, starts[ag], goal_locations[ag], rt);
		runtime_plan_paths += (std::clock() - t) * 1.0 / CLOCKS_PER_SEC;
        path_cost = path_planner.path_cost;
        node_expanded += path_planner.num_expanded;

        rt.clear();

        if (path->empty())
        {
            std::cout << "Agent num " << num_of_agents;
            std::cout << "Agent " << ag;
            std::cout << "NO SOLUTION EXISTS";
            delete path;
            return false;
        }

        if (paths[ag] != nullptr) {
            delete paths[ag];
        }

        paths[ag] = path;
        vis_goal_time[ag] = path_planner.vis_goal_time;

	}

    if (screen == 2)
    {
        double runtime = (std::clock() - time) * 1.0 / CLOCKS_PER_SEC;
        std::cout << "Done! (" << runtime << "s)" << std::endl;
    }
    return true;
}


// bool PP::generate_paths()
// {
//     clock_t time = std::clock();

// 	// initialize paths_found_initially
// 	paths.resize(num_of_agents, nullptr);
//     vis_goal_time.resize(num_of_agents);


//     if (screen == 2)
//         std::cout << "Generate root CT node ..." << std::endl;

//     for (int i = 0; i < num_of_agents; i++) 
// 	{   
//         int ag = priority_indices[i];

//         boost::unordered_set<int> previous_elements;
//         for (int j = 0; j < i; j++) 
//         {
//             previous_elements.insert(priority_indices[j]);
//         }

//         Path* path = new Path();
//         double path_cost;
//         int start_location  = starts[ag].location;
//         clock_t t = std::clock();
// 		// rt.copy(initial_rt);
//         rt.build(paths, initial_constraints, previous_elements, ag, start_location);
//         rt.print();

//         runtime_rt += (std::clock() - t) * 1.0 / CLOCKS_PER_SEC;
//         vector< vector<double> > h_values(goal_locations[ag].size());

//         t = std::clock();
//         *path = path_planner.run(G, starts[ag], goal_locations[ag], rt);
// 		runtime_plan_paths += (std::clock() - t) * 1.0 / CLOCKS_PER_SEC;
//         path_cost = path_planner.path_cost;
//         node_expanded += path_planner.num_expanded;

//         rt.clear();

//         if (path->empty())
//         {
//             std::cout << "Agent num " << num_of_agents;
//             std::cout << "Agent " << ag;
//             std::cout << "NO SOLUTION EXISTS";
//             delete path;
//             return false;
//         }

//         if (paths[ag] != nullptr) {
//             delete paths[ag];
//         }

//         paths[ag] = path;
//         vis_goal_time[ag] = path_planner.vis_goal_time;

// 	}

//     if (screen == 2)
//     {
//         double runtime = (std::clock() - time) * 1.0 / CLOCKS_PER_SEC;
//         std::cout << "Done! (" << runtime << "s)" << std::endl;
//     }
//     return true;
// }



void PP::get_priority()
{
    for (int ag = 0; ag < num_of_agents; ag++)
    {
        if(goal_locations[ag].size() > 0)
        {
            priority.push_back(G.heuristics.at(goal_locations[ag][0].first)[starts[ag].location]);
        }
        else
        {
            priority.push_back(0);
        }
        
    }

    for (int i = 0; i < (int)priority.size(); i++) {
        priority_indices.push_back(i);
    }

    sort(priority_indices.begin(), priority_indices.end(), [&](int a, int b){
        return priority[a] > priority[b];
    });
}

bool PP::run(const vector<State>& starts,
                    const vector< vector<pair<int, int> > >& goal_locations,
                    int time_limit)
{
    clear();

    // set timer
	start = std::clock();
    this->starts = starts;
    this->goal_locations = goal_locations;
    this->num_of_agents = starts.size();
    this->time_limit = time_limit;

    solution_cost = -2;
    solution_found = false;

    rt.num_of_agents = num_of_agents;
    rt.map_size = G.size();
    rt.k_robust = k_robust;
    rt.window = window;
	rt.hold_endpoints = hold_endpoints;
    path_planner.travel_times = travel_times;
	path_planner.hold_endpoints = hold_endpoints;
	path_planner.prioritize_start = prioritize_start;

    get_priority();

    for (int i = 0; i < priority_indices.size(); i++)
    {
        int ag = priority_indices[i];
    }

    if (!generate_paths())
    {
        return false;
    } 
    
    solution_found = true;

	runtime = (std::clock() - start) * 1.0 / CLOCKS_PER_SEC;
    get_solution();

	if (solution_found && !validate_solution())
	{
        std::cout << "Solution invalid!!!" << std::endl;
        // print_paths();
        exit(-1);
	}
    min_sum_of_costs = 0;
    for (int i = 0; i < num_of_agents; i++)
    {
        int start = starts[i].location;
        for (const auto& goal : goal_locations[i])
        {
            min_sum_of_costs += G.heuristics.at(goal.first)[start];
            start = goal.first;
        }
    }
	if (screen > 0) // 1 or 2
		print_results();
	return solution_found;
}



PP::PP(const BasicGraph& G, SingleAgentSolver& path_planner) : MAPFSolver(G, path_planner),
        lazyPriority(false), //useCAT(false), 
		// best_node(nullptr), initial_rt(G), rt(G) {}
        best_node(nullptr),  rt(G) {}


inline void PP::release_closed_list()
{
	for (auto it = allNodes_table.begin(); it != allNodes_table.end(); it++)
		delete *it;
	allNodes_table.clear();
}


PP::~PP()
{
	release_closed_list();
}


bool PP::validate_solution()
{
    list<Conflict> conflict;
	for (int a1 = 0; a1 < num_of_agents; a1++)
	{
		for (int a2 = a1 + 1; a2 < num_of_agents; a2++)
		{
            find_conflicts(conflict, a1, a2);
            if (!conflict.empty())
            {
                int a1, a2, loc1, loc2, t;
                std::tie(a1, a2, loc1, loc2, t) = conflict.front();
                if (loc2 < 0)
                    std::cout << "Agents "  << a1 << " and " << a2 << " collides at " << loc1 <<
                    " at timestep " << t << std::endl;
                else
                    std::cout << "Agents " << a1 << " and " << a2 << " collides at (" <<
                              loc1 << "-->" << loc2 << ") at timestep " << t << std::endl;
                return false;
            }
		}
	}
	return true;
}

void PP::print_paths() const
{
	for (int i = 0; i < num_of_agents; i++)
	{
		if (paths[i] == nullptr)
            continue;
        std::cout << "Agent " << i << " (" << paths[i]->size() - 1 << "): ";
		for (auto s : (*paths[i]))
			std::cout << s.location << "->";
		std::cout << std::endl;
	}
}


// adding new nodes to FOCAL (those with min-f-val*f_weight between the old and new LB)
void PP::update_focal_list()
{
	/*PBSNode* open_head = open_list.top();
	if (open_head->f_val > min_f_val)
	{
		if (screen == 2)
		{
			std::cout << "  Note -- FOCAL UPDATE!! from |FOCAL|=" << focal_list.size() << " with |OPEN|=" << open_list.size() << " to |FOCAL|=";
		}
		min_f_val = open_head->f_val;
		double new_focal_list_threshold = min_f_val * focal_w;
		for (PBSNode* n : open_list)
		{
			if (n->f_val > focal_list_threshold &&
				n->f_val <= new_focal_list_threshold)
				n->focal_handle = focal_list.push(n);
		}
		focal_list_threshold = new_focal_list_threshold;
		if (screen == 2)
		{
			std::cout << focal_list.size() << std::endl;
		}
	}*/
}

void PP::update_CAT(int ex_ag)
{
    size_t makespan = 0;
	for (int ag = 0; ag < num_of_agents; ag++) 
    {
        if (ag == ex_ag || paths[ag] == nullptr)
            continue;
        makespan = std::max(makespan, paths[ag]->size());
    }
	cat.clear();
    cat.resize(makespan);
    for (int t = 0; t < (int)makespan; t++)
        cat[t].resize(G.size(), false);

    for (int ag = 0; ag < num_of_agents; ag++) 
	{
        if (ag == ex_ag || paths[ag] == nullptr)
            continue; 
		for (int timestep = 0; timestep < (int)paths[ag]->size() - 1; timestep++)
		{
			int loc = paths[ag]->at(timestep).location;
			if (loc < 0)
			    continue;
			for (int t = max(0, timestep - k_robust); t <= timestep + k_robust; t++)
			    cat[t][loc] = true;
		}
	}
}

void PP::print_results() const
{
    std::cout << "PP:";
	if(solution_cost >= 0) // solved
		std::cout << "Succeed,";
	else if(solution_cost == -1) // time_out
		std::cout << "Timeout,";
	else if(solution_cost == -2) // no solution
		std::cout << "No solutions,";
	else if (solution_cost == -3) // nodes out
		std::cout << "Nodesout,";

	std::cout << runtime << "," << max_makespan << "," << 
		HL_num_expanded << "," << HL_num_generated << "," <<
		solution_cost << "," << min_sum_of_costs << "," <<
		avg_path_length << "," <<
		runtime_plan_paths << "," << runtime_rt << "," <<
		runtime_get_higher_priority_agents << "," <<
		runtime_copy_priorities << "," <<
		runtime_detect_conflicts << "," <<
		runtime_copy_conflicts << "," <<
		runtime_choose_conflict << "," <<
		runtime_find_consistent_paths << "," <<
		runtime_find_replan_agents <<
		std::endl;
}

void PP::save_results(const std::string &fileName, const std::string &instanceName) const
{
	std::ofstream stats;
	stats.open(fileName, std::ios::app);
	stats << runtime << "," <<
		HL_num_expanded << "," << HL_num_generated << "," <<
		LL_num_expanded << "," << LL_num_generated << "," <<
		solution_cost << "," << min_sum_of_costs << "," <<
		avg_path_length << "," <<
		instanceName << std::endl;
	stats.close();
}

void PP::print_conflicts(const PBSNode &curr) const
{
	for (auto c : curr.conflicts)
	{
		std::cout << c << std::endl;
	}
}


void PP::save_search_tree(const std::string &fname) const
{
    std::ofstream output;
    output.open(fname, std::ios::out);
    output << "digraph G {" << std::endl;
    output << "size = \"5,5\";" << std::endl;
    output << "center = true;" << std::endl;
    for (auto node : allNodes_table)
    {
        if (node == dummy_start)
            continue;
        else if (node->time_expanded == 0) // this node is in the openlist
            output << node->time_generated << " [color=blue]" << std::endl;
        output << node->parent->time_generated << " -> " << node->time_generated << std::endl;
    }
    output << "}" << std::endl;
    output.close();
}

void PP::save_constraints_in_goal_node(const std::string &fileName) const
{
	best_node->priorities.save_as_digraph(fileName );
}

void PP::get_solution()
{
    solution.resize(num_of_agents);
    for (int k = 0; k < num_of_agents; k++)
    {
        solution[k] = *paths[k];
    }

    solution_cost  = 0;
    avg_path_length = 0;

    for (int k = 0; k < num_of_agents; k++)
    {
        if (goal_locations[k].size() == 0) {
            continue;
        }
        avg_path_length += paths[k]->size();
    }
    avg_path_length /= num_of_agents;
}