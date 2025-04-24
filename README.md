# MAPD Reinforcement Learning

This repository contains code for Multi-Agent Path Decoupling (MAPD) with Reinforcement Learning.

## Installation

1. Install the required Python packages:
```bash
pip install python==3.11.9
pip install torch==2.4.0
pip install stable-baselines3==2.4.0
pip install tqdm==4.66.5
pip install pygmtools==0.5.3
pip install transformers==4.44.2
pip install pybind11==2.13.6
```

2. Compile the C++ solver code:
```bash
cd code/mapf_solver
cmake .
make
```

## Training

To train the model, use the `train.sh` script with the following parameters:

```bash
bash train.sh -g 0 -l 1e-4 -m 0.99 -t 0.01 -n 500 -p 16 -d bmm -h 512
```

Parameters:
- `-g`: GPU ID
- `-l`: Learning rate
- `-m`: A2C entropy coefficient
- `-t`: Sinkhorn temperature
- `-n`: Number of tasks to generate
- `-p`: Number of parallel processes
- `-d`: Probability matrix computation method (use `bmm`)
- `-h`: Hidden size for the model

## Testing

To test the trained model, use the `test.sh` script:

```bash
bash test.sh <task_num> <grid_path> <eval_data_path> <checkpoint_path> <gpu_id>
```

Parameters:
- `task_num`: Number of tasks
- `grid_path`: Path to the grid file
- `eval_data_path`: Path to evaluation data
- `checkpoint_path`: Path to the trained model checkpoint
- `gpu_id`: GPU ID to use for testing 