#!/usr/bin/env python

import time
import numpy as np
from numba import vectorize, guvectorize, cuda, jit

# md parameters picked to produce better simulation traj
epsilon = -2.0
sigma = 0.9
s6 = sigma**6
step_length = 0.0000001

#####################################
# Methods to calculate the LJ force #
#####################################

# reference method to calculate LJ force
def ref_LJ(coords):
    """
    reference method calculate LJ force

    inputs
    ------
    coords: matrix of shape (N, 3), each row is the [x,y,z] coordinate of an atom

    returns
    -------
    force: matrix of shape (N, 3), each row is the [fx,fy,fz] force vector of an atom
    """
    noa = len(coords)
    forces = np.zeros((noa,3), dtype=np.float32)
    for i in range(noa):
        for j in range(i+1,noa):
            dc = coords[i] - coords[j]
            # distance between atom i and j
            r2 = dc[0]*dc[0] + dc[1]*dc[1] + dc[2]*dc[2]
            # LJ equation
            f = (-12 / r2**7 * s6 + 6 / r2**4) * 4 * epsilon * s6
            # accumulate the force for atom i
            forces[i] += f * dc
            # Pair (j,i) is skipped because we know the force on atom j is opposite of i
            forces[j] -= f * dc
    return forces

@jit(nopython=True)
def ref_LJ_jit(coords):
    """
    reference method optimized by numba.jit
    The content is exactly the same as the original reference method
    The performance gain is mainly from the for loops

    inputs
    ------
    coords: matrix of shape (N, 3), each row is the [x,y,z] coordinate of an atom

    returns
    -------
    force: matrix of shape (N, 3), each row is the [fx,fy,fz] force vector of an atom

    notes
    -----
    It can also be defined as:
    ref_LJ_jit = jit(nopython=True)(ref_LJ)
    """
    noa = len(coords)
    forces = np.zeros((noa,3), dtype=np.float32)
    for i in range(noa):
        for j in range(i+1,noa):
            dc = coords[i] - coords[j]
            # distance between atom i and j
            r2 = dc[0]*dc[0] + dc[1]*dc[1] + dc[2]*dc[2]
            # LJ equation
            f = (-12 / r2**7 * s6 + 6 / r2**4) * 4 * epsilon * s6
            # accumulate the force for atom i
            forces[i] += f * dc
            # Pair (j,i) is skipped because we know the force on atom j is opposite of i
            forces[j] -= f * dc
    return forces

def numpy_LJ(coords):
    """
    Numpy vectorized method to calculate LJ force
    This function is heavily optimized
    It utilize the einsum which is very efficient

    inputs
    ------
    coords: matrix of shape (N, 3), each row is the [x,y,z] coordinate of an atom

    returns
    -------
    force: matrix of shape (N, 3), each row is the [fx,fy,fz] force vector of an atom
    """
    # coordinate difference between all atoms
    c_diff = coords[:,np.newaxis,:] - coords[np.newaxis,:,:]
    # distance square matrix between all atoms
    r2_mat = np.sum(np.square(c_diff), axis=-1)
    # R^4 matrix to be used later
    r4_mat = np.square(r2_mat)
    # diagonal is 0, fill it with 1 to prevent dividing by 0 in the next step
    np.fill_diagonal(r4_mat, 1.0)
    # R^-8 matrix
    rn8_mat = 1.0 / np.square(r4_mat)
    # R^-14 matrix
    rn14_mat = np.square(rn8_mat) * r2_mat
    # LJ equation to compute f norm
    f_lj_mat = (-12.0*rn14_mat * s6 + 6.0*rn8_mat) * 4 * epsilon * s6
    # contract C diff with f_norm to get the final force matrix
    return np.einsum('ijk,ij->ik', c_diff, f_lj_mat)

###############################################################
# Methods to calculate the LJ force and perform a step on GPU #
###############################################################

@guvectorize(['void(float32[:], float32[:,:], float32[:])'], '(n),(m,n)->(n)', target='cuda')
def cuda_LJ_step(this_coord, coords, prev_coord):
    """
    Vectorized ufunc for running on GPU
    Each run computes force for one atom then update its coord

    Inputs
    ------
    this_coord: [x, y, z] for this atom
    coords: matrix of shape (N, 3), for coordinates of all atoms
    prev_coord: [x, y, z] for this atom, passed in then overwritten by the next coords

    Returns
    -------
    This function has no return, but writes into the provided prev_coord matrix
    """
    noa = len(coords)
    # build force
    fx = 0.0
    fy = 0.0
    fz = 0.0
    for i in range(noa):
        dx = this_coord[0] - coords[i][0]
        dy = this_coord[1] - coords[i][1]
        dz = this_coord[2] - coords[i][2]
        r2 = dx**2 + dy**2 + dz**2
        # prevent self interaction
        if r2 != 0:
            f = (-12 / r2**7 * s6 + 6 / r2**4) * 4 * epsilon * s6
            fx += f * dx
            fy += f * dy
            fz += f * dz
    # compute dr
    dx = fx * step_length
    dy = fy * step_length
    dz = fz * step_length
    # apply verlet update
    # output is written to prev_coord
    prev_coord[0] = this_coord[0]*2 + dx - prev_coord[0]
    prev_coord[1] = this_coord[1]*2 + dy - prev_coord[1]
    prev_coord[2] = this_coord[2]*2 + dz - prev_coord[2]

def cuda_LJ_kernal_step(coords, _, prev_coords):
    """
    Function to launch cuda kernal to perform one step
    Mainly for keeping the interface consistent with cuda_LJ_step
    
    Inputs
    ------
    coords: matrix of shape (N, 3), for coordinates of all atoms
    _: not used, only for keeping the interface consistent with cuda_LJ_step
    prev_coords: matrix of shape (N, 3), passed in as coords from last step, then overwritten by the next coords

    Returns
    -------
    This function has no return, but writes into the provided prev_coords matrix
    """
    # parallelize the gpu kernal runs in 1D
    threadsperblock = 32 # optimized
    blockspergrid = (len(coords) + (threadsperblock - 1)) // threadsperblock
    cuda_LJ_kernal[blockspergrid, threadsperblock](coords, prev_coords)

@cuda.jit
def cuda_LJ_kernal(coords, prev_coords):
    """
    Kernal function for computing one verlet step on GPU
    Each run computes force for one atom then update its coord

    Inputs
    ------
    coords: matrix of shape (N, 3), for coordinates of all atoms
    prev_coords: matrix of shape (N, 3), passed in as coords from last step, then overwritten by the next coords

    Returns
    -------
    This function has no return, but writes into the provided prev_coords matrix
    """
    noa = len(coords)
    pos = cuda.grid(1)
    # check boundaries
    if pos >= noa:
        return
    this_coord = coords[pos]
    # build force
    fx = 0.0
    fy = 0.0
    fz = 0.0
    for i in range(noa):
        dx = this_coord[0] - coords[i][0]
        dy = this_coord[1] - coords[i][1]
        dz = this_coord[2] - coords[i][2]
        r2 = dx**2 + dy**2 + dz**2
        # prevent self interaction
        if r2 != 0:
            f = (-12 / r2**7 * s6 + 6 / r2**4) * 4 * epsilon * s6
            fx += f * dx
            fy += f * dy
            fz += f * dz
    # compute dr
    dx = fx * step_length
    dy = fy * step_length
    dz = fz * step_length
    # apply verlet update
    # output is written to prev_coord
    prev_coords[pos, 0] = this_coord[0]*2 + dx - prev_coords[pos, 0]
    prev_coords[pos, 1] = this_coord[1]*2 + dy - prev_coords[pos, 1]
    prev_coords[pos, 2] = this_coord[2]*2 + dz - prev_coords[pos, 2]

#######################################
# Simulation with integration methods #
#######################################

def sim_verlet(force_func, coords, n_steps, save_traj=False, verbose=True):
    """
    Verlet integration
    R_next = R * 2 + dR - R_prev
    """
    if save_traj:
        outfile = open('traj.xyz', 'w')
    # store prev coords to use later
    prev_coords = coords.copy()
    # start loop
    for t in range(1, n_steps+1):
        # compute a (Nx3) matrix of force, each row for an atom
        m_force = force_func(coords)
        # calculate step movement
        dr = m_force * step_length
        # verlet integration R_next = R * 2 + dR - R_prev
        # this part is not optimized because it takes no time on CPU
        next_coords = coords*2 + dr - prev_coords
        prev_coords, coords = coords, next_coords
        # print progress and optionally save the traj
        if t % 100 == 0:
            if verbose:
                print(f'{t:5d} steps complete', end='\r')
            if save_traj:
                save_xyz(coords, outfile)
    if save_traj:
        outfile.close()

def sim_verlet_gpu(step_func, coords, n_steps, save_traj=False, verbose=True):
    """
    CUDA version of the verlet integration simulation
    Optimized to improve performance by avoid copying between host and device memory
    """
    if save_traj:
        outfile = open('traj.xyz', 'w')
    # allocate GPU memory for a few matrices
    coords_gpu = cuda.to_device(coords)
    prev_coords_gpu = cuda.to_device(coords)
    # start loop
    for t in range(1, n_steps+1):
        # force calculation and apply update are combine here to improve performance
        # note: after update, the new coords is written into prev_coords_gpu
        step_func(coords_gpu, coords_gpu, prev_coords_gpu)
        # switch reference
        coords_gpu, prev_coords_gpu = prev_coords_gpu, coords_gpu
        # print progress and optionally save the traj
        if t % 100 == 0:
            if verbose:
                # blocking call to finish GPU execution
                cuda.synchronize()
                print(f'{t:8d} steps complete', end='\r')
            if save_traj:
                # copy coords from gpu to host memory, blocking
                coords_gpu.copy_to_host(coords)
                save_xyz(coords, outfile)
    # ensure everything on GPU is done
    cuda.synchronize()
    if save_traj:
        outfile.close()

def save_xyz(coords, outputfile):
    """
    Utilite function to save the coordinate in file
    """
    outputfile.write('%d\n\n'%len(coords))
    # scale the coords to fit unit of Angstrom
    for i in coords*5:
        outputfile.write('Cu %10.7f %10.7f %10.7f\n'%(i[0],i[1],i[2]))

# all supported methods
SUPPORTED_METHODS = {
    'Cref': (ref_LJ, sim_verlet),
    'Cjit': (ref_LJ_jit, sim_verlet),
    'Cnpy': (numpy_LJ, sim_verlet),
    'Gvec': (cuda_LJ_step, sim_verlet_gpu),
    'Gknl': (cuda_LJ_kernal_step, sim_verlet_gpu),
}

def run_md(cube_size, n_steps, method, save_traj=False, verbose=True):
    """
    master function to perform MD simulation, using verlet integration
    """
    # build a cube as the (Nx3) matrix of cartesian coordinates
    if verbose:
        print(f'Building a {cube_size}x{cube_size}x{cube_size} cube with {cube_size**3} atoms')
    coords = np.array([[x,y,z] for x in range(cube_size) for y in range(cube_size) for z in range(cube_size)], dtype=np.float32)
    # define integration method and force functions
    force_func, integration_func = SUPPORTED_METHODS[method]
    # perform md simulation with integration method
    t0 = time.time()
    if verbose:
        print(f'Running simulation of {n_steps} steps with {method} method')
    # run the simulation
    integration_func(force_func, coords, n_steps, save_traj=save_traj, verbose=verbose)
    t1 = time.time()
    if verbose:
        print(f'\nFinished in {t1-t0:.3f} s')

# handle args from command line
def main():
    import argparse
    parser = argparse.ArgumentParser("Run a simple MD simulation with Cu atoms in a cubic grid with Lennard-Johns force", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--method', default='np', choices=SUPPORTED_METHODS.keys(), help='Method to compute LJ force')
    parser.add_argument('-c', '--cube_size', default=3, type=int, help='Size of the cubic grid, 3 means 3x3x3')
    parser.add_argument('-n', '--n_steps', default=1000, type=int, help='Number of steps to perform the simulation')
    parser.add_argument('-s', '--save_traj', default=False, action='store_true', help='Save trajectory every 100 step as traj.xyz')
    args = parser.parse_args()

    run_md(args.cube_size, args.n_steps, args.method, save_traj=args.save_traj)

if __name__ == "__main__":
    main()