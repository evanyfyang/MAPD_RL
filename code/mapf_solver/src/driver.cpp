#include "KivaSystemOnline.h"
// #include "KivaSystem.h"
// #include "SortingSystem.h"
// #include "OnlineSystem.h"
// #include "BeeSystem.h"
#include "ID.h"
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <pybind11/stl.h>
#include <boost/any.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;
namespace po = boost::program_options;

MAPFSolver* set_solver(BasicGraph& G, const boost::program_options::variables_map& vm)
{
	string solver_name = vm["single_agent_solver"].as<string>();
	SingleAgentSolver* path_planner;
	MAPFSolver* mapf_solver;
	if (solver_name == "ASTAR")
	{
		path_planner = new StateTimeAStar();
	}
	else if (solver_name == "SIPP")
	{
		path_planner = new SIPP();
	}
	else
	{
		cout << "Single-agent solver " << solver_name << "does not exist!" << endl;
		exit(-1);
	}

	solver_name = vm["solver"].as<string>();
	if (solver_name == "PBS")
	{
		PBS* pbs = new PBS(G, *path_planner);
		pbs->lazyPriority = vm["lazyP"].as<bool>();
		pbs->prioritize_start = vm["prioritize_start"].as<bool>();
		pbs->setRT(vm["CAT"].as<bool>(), vm["prioritize_start"].as<bool>());
		mapf_solver = pbs;
	}
	else
	{
		cout << "Solver " << solver_name << "does not exist!" << endl;
		exit(-1);
	}

	if (vm["id"].as<bool>())
	{
		return new ID(G, *path_planner, *mapf_solver);
	}
	else
	{
		return mapf_solver;
	}
}

void set_parameters(BasicSystem& system, const boost::program_options::variables_map& vm)
{
	system.outfile = vm["output"].as<std::string>();
	system.screen = vm["screen"].as<int>();
	system.log = vm["log"].as<bool>();
	system.num_of_drives = vm["agentNum"].as<int>();
	system.time_limit = vm["cutoffTime"].as<int>();
	system.simulation_window = vm["simulation_window"].as<int>();
	system.planning_window = vm["planning_window"].as<int>();
	system.travel_time_window = vm["travel_time_window"].as<int>();
	system.consider_rotation = vm["rotation"].as<bool>();
	system.k_robust = vm["robust"].as<int>();
	system.hold_endpoints = vm["hold_endpoints"].as<bool>();
	system.useDummyPaths = vm["dummy_paths"].as<bool>();
	system.task_truncated_size = vm["task_truncated_size"].as<int>();
	system.REPLAN = vm["replan"].as<bool>();
	system.look_ahead_horizon = vm["look_ahead_horizon"].as<int>();
	system.neighborhood_size = vm["neighborhood_size"].as<int>();
	if (vm.count("seed"))
		system.seed = vm["seed"].as<int>();
	else
		system.seed = (int)time(0);
	srand(system.seed);
}

class PBSSolver
{	
	private:
		po::variables_map vm; 
		po::options_description desc; 
		KivaGrid G;
		MAPFSolver* solver;
		KivaSystemOnline* system;
	public:
		PBSSolver(const std::vector<std::string>& args) : desc("Allowed options") 
		{	
			po::options_description desc("Allowed options");
			desc.add_options()
				("help", "produce help message")
				("map,m", po::value<std::string>()->required(), "input map file")
				("task", po::value<std::string>()->default_value(""), "input task file")
				("output,o", po::value<std::string>()->default_value("../exp/test"), "output folder name")
				("agentNum,k", po::value<int>()->required(), "number of drives")
				("cutoffTime,t", po::value<int>()->default_value(500), "cutoff time (seconds)")
				("seed,d", po::value<int>(), "random seed")
				("screen,s", po::value<int>()->default_value(1), "screen option (0: none; 1: results; 2:all)")
				("solver", po::value<string>()->default_value("PBS"), "solver (PBS, WPBS)")
				("id", po::value<bool>()->default_value(false), "independence detection")
				("single_agent_solver", po::value<string>()->default_value("ASTAR"), "single-agent solver (ASTAR, SIPP)")
				("lazyP", po::value<bool>()->default_value(false), "use lazy priority")
				("simulation_time", po::value<int>()->default_value(5000), "run simulation")
				("simulation_window", po::value<int>()->default_value(5), "call the planner every simulation_window timesteps")
				("travel_time_window", po::value<int>()->default_value(0), "consider the traffic jams within the given window")
				("planning_window", po::value<int>()->default_value(INT_MAX / 2),
						"the planner outputs plans with first planning_window timesteps collision-free")
				("potential_function", po::value<string>()->default_value("NONE"), "potential function (NONE, SOC, IC)")
				("potential_threshold", po::value<double>()->default_value(0), "potential threshold")
				("rotation", po::value<bool>()->default_value(false), "consider rotation")
				("robust", po::value<int>()->default_value(0), "k-robust (for now, only work for PBS)")
				("CAT", po::value<bool>()->default_value(false), "use conflict-avoidance table")
				("hold_endpoints", po::value<bool>()->default_value(false),
						"Hold endpoints from Ma et al, AAMAS 2017")
				("dummy_paths", po::value<bool>()->default_value(false),
						"Find dummy paths from Liu et al, AAMAS 2019")
				("prioritize_start", po::value<bool>()->default_value(false), "Prioritize waiting at start locations")
				("suboptimal_bound", po::value<double>()->default_value(1), "Suboptimal bound for ECBS")
				("log", po::value<bool>()->default_value(false), "save the search trees (and the priority trees)")
				("task_truncated_size", po::value<int>()->default_value(1), "task-truncated size in online/offline MAPD")
				("replan", po::value<bool>()->default_value(true), "replan variant")
				("look_ahead_horizon", po::value<int>()->default_value(1), "1 means no look-ahead horizon applied")
				("neighborhood_size", po::value<int>()->default_value(2), "neighborhood_size")
				;
			clock_t start_time = clock();
			po::variables_map vm;

			try {
				po::store(po::command_line_parser(args).options(desc).run(), vm);
				if (vm.count("help")) {
					std::cout << desc << std::endl;
					throw std::runtime_error("Help requested.");
				}
				po::notify(vm); 
			} catch (const po::error& e) {
				throw std::runtime_error("Error parsing arguments: " + std::string(e.what()));
			}


			boost::filesystem::path dir(vm["output"].as<std::string>() +"/");
			boost::filesystem::create_directories(dir);
			if (vm["log"].as<bool>())
			{
				boost::filesystem::path dir1(vm["output"].as<std::string>() + "/goal_nodes/");
				boost::filesystem::path dir2(vm["output"].as<std::string>() + "/search_trees/");
				boost::filesystem::create_directories(dir1);
				boost::filesystem::create_directories(dir2);
			}

			if (!G.load_Minghua_map(vm["map"].as<std::string>()))
				throw std::runtime_error("Map Error.");
			solver = set_solver(G, vm);
			system = new KivaSystemOnline(G, *solver);
			set_parameters(*system, vm);
			G.preprocessing(system->consider_rotation);

		}

	~PBSSolver() {
        // delete solver;
        // delete system;
    }

	AgentTaskStatus update_task(vector<vector<int>>& task, vector<int>& new_agents, int simulation_time, float task_frequency, int task_release_period)
	{
		// cout<<1<<endl;
		// solver = set_solver(G, vm);
		system->load_tasks(task, new_agents, simulation_time, task_frequency, task_release_period);
		vector<vector<int>> agent_tasks = {};
		AgentTaskStatus status = system->simulate_until_next_assignment(agent_tasks);
		
		return status;
	}

    AgentTaskStatus update(const vector<vector<int>>& agent_tasks) {

        // 模拟并返回路径
        AgentTaskStatus status = system->simulate_until_next_assignment(agent_tasks);
        return status;
    }
};


PYBIND11_MODULE(mapf_solver, m) {
     py::class_<Task>(m, "Task")
        // 构造函数
        .def(py::init<>())  // Task()
        // Task(int id, int release_time, vector<int>& goal_arr)
        .def(py::init<int, int, std::vector<int>&>(),
             py::arg("id"), py::arg("release_time"), py::arg("goal_arr"))
        // 只暴露以下字段
        .def_readwrite("task_id", &Task::task_id)
        .def_readwrite("goal_arr", &Task::goal_arr)
        .def_readwrite("release_time", &Task::release_time)
		.def_readwrite("estimated_service_time", &Task::estimated_service_time)
        // 其余字段和 boost::heap 都不绑
        ;

    // 2) 绑定 Agent
    py::class_<Agent>(m, "Agent")
        .def(py::init<>())               // Agent()
        .def(py::init<int, int>(),       // Agent(int start_location, int agent_id)
             py::arg("start_location"), py::arg("agent_id"))
        // 暴露字段
        .def_readwrite("agent_id", &Agent::agent_id)
        .def_readwrite("start_location", &Agent::start_location)
        .def_readwrite("start_timestep", &Agent::start_timestep)
        .def_readwrite("is_delivering", &Agent::is_delivering)
        .def_readwrite("task_sequence", &Agent::task_sequence)
        ;

    // 3) 绑定 Path
    py::class_<State>(m, "State")
        // 构造
        .def(py::init<>())  // State()
        .def(py::init<int, int, int, int>(),
             py::arg("location")=-1, 
             py::arg("timestep")=-1, 
             py::arg("orientation")=-1, 
             py::arg("l_val")=1)
        // 只暴露下面四个值
        .def_readwrite("location", &State::location)
        .def_readwrite("timestep", &State::timestep)
        .def_readwrite("orientation", &State::orientation)
        .def_readwrite("l_val", &State::l_val)
        ;

	py::bind_vector<std::vector<State>>(m, "Path");

    // 4) 绑定 AgentTaskStatus
    py::class_<AgentTaskStatus>(m, "AgentTaskStatus")
        .def(py::init<>())  // 无参构造
        .def_readwrite("tasks", &AgentTaskStatus::tasks)
        .def_readwrite("delivering_tasks", &AgentTaskStatus::delivering_tasks)
        .def_readwrite("agents_all", &AgentTaskStatus::agents_all)
        .def_readwrite("solution", &AgentTaskStatus::solution)
        .def_readwrite("allFinished", &AgentTaskStatus::allFinished)
        .def_readwrite("agent_task_pair", &AgentTaskStatus::agent_task_pair)
		.def_readwrite("delivering_service_time", &AgentTaskStatus::delivering_service_time)
        .def_readwrite("valid", &AgentTaskStatus::valid)
		.def_readwrite("timestep",&AgentTaskStatus::timestep)
		.def_readwrite("finished_service_time", &AgentTaskStatus::finished_service_time)
        ;

	py::class_<PBSSolver>(m, "PBSSolver")
        // 绑定构造函数(支持用 list[str] 来初始化)
        .def(py::init<const std::vector<std::string>&>(),
             py::arg("args"))
        // 绑定成员函数 update_task
        .def("update_task", &PBSSolver::update_task,
             py::arg("task"), py::arg("new_agents"), py::arg("simulation_time"), py::arg("task_frequency"), py::arg("task_release_period"),
             R"pbdoc(
                Load new tasks into the system. Returns some 2D integer vector as result.
             )pbdoc")
        // 绑定成员函数 update
        .def("update", &PBSSolver::update,
             py::arg("agent_tasks"),
             R"pbdoc(
                Run the solver's simulate_until_next_assignment method, 
                and return an AgentTaskStatus.
             )pbdoc")
        ;
	
}
