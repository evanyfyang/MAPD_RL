#pragma once
#include "BasicGraph.h"


class KivaGrid :
	public BasicGraph
{
public:
	vector<int> endpoints;
	vector<int> agent_home_locations;
	// vector<int> initial_locations;

    bool load_map(string fname);
	bool load_Minghua_map(string fname);
    void preprocessing(bool consider_rotation); // compute heuristics
	void update_agents(vector<int>& new_agents);
};
