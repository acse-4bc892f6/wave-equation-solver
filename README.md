# Solve wave equation in computational grid using MPI

This is the submitted version of the individual coursework for the Parallel Programming module in the Applied Computational Science and Engineering course.

## Instructions to use the parallelized wave equation solver

Number of grids in domain, domain dimension, maximum time step, frequency of writing grid data to files, boundary conditions and wave propagation speed can be customised at the beginning of `Parallel_Wave_Equation.cpp`.

Execute the solver by compiling `main.cpp` and running its executable:

```mpicxx -o exe_file main.cpp```

```mpiexec -np num_process ./exe_file```

where `exe_file` and `num_process` are the name of the executable file and number of processes to execute the solver respectively.

Ensure all `to_file()` functions are uncommented in `main.cpp` if you wish to make an animation afterwards. All grid data would be within the `output` directory, which is created when `main.cpp` is running.

To make an animation, run the python script `collate_animate.py`:

```python collate_animate.py```

It would take a few minutes for the script to finish running. The output would be `animation.gif`.
