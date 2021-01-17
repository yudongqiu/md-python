# md-python
Instructional Python Implementation of Simple Molecular Simulation
In this package, you can find exmaple codes that simulate a cubic of atoms, interacting with each other by Lennard-Jones force.
You will also find 4 set of implementations:
1. Native python with for loops (Slowest)
2. Numba jit optimized version with for loops (Faster)
3. Numpy vectorized methods without loop (Fastest on CPU)
4. CUDA accelerated method running on GPU (Faster than CPU for larger systems)

## Setup
1. (optional) Create new conda env
   ```
   conda create -n md-python
   conda activate md-python
   ```

2. Install dependencies
   ```
   conda install numba numpy cudatoolkit
   ```
   cudatoolkit is only needed for GPU support

## Example

You can run the following command to perform a quick and simple md simulation with Lennard-Jones force
```
python md_python/simple_LJ.py
```

Optional arguments are provided in the menu
```
optional arguments:
  -h, --help            show this help message and exit
  -m {ref,jit,np,gpu}, --method {ref,jit,np,gpu}
                        Method to compute LJ force (default: ref)
  -c CUBE_SIZE, --cube_size CUBE_SIZE
                        Size of the cubic grid, 3 means 3x3x3 (default: 3)
  -n N_STEPS, --n_steps N_STEPS
                        Number of steps to perform the simulation (default: 1000)
  -s, --save_traj       Save trajectory every 100 step as traj.xyz (default: False)
```

You can also run the benchmark to compare CPU and GPU performances
```
python md_python/benchmark.py
```
Note: It depends on the matplotlib library, which can be installed by
```
conda install matplotlib
```
Resulting running on my PC (8700k + GTX2080) is provided as benchmark.pdf