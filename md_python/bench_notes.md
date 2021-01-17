# Notes for the benchmark

The benchmark was done with i7-8700K CPU (6-core) and RTX 2080 GPU (CUDA 11.0)

A few interesting observations from the benchmark:
1. [Cref] CPU ref method shows python for loops are really slow
2. [Cjit] At very small sizes (2^3=8 atoms), CPU jit is fastest when ignoring compilation cost (which can be cached)
3. [Cnpy] Numpy optimized method is fastest at (3^3=27 atoms), and stays as the fastest CPU method. This is because the vectorized method overcomes the overhead and einsum runs very efficiently on CPU.
3. [Gvec] Vectorized ufunc running on GPU has the highest overhead (~0.3ms), I guess it's mainly because the vectorization is figured out on CPU every step. It casts the operation on the whole coords matrix row by row. Also, memory copy is heavily optimized here with tricks like storing the next coord in prev_coords, otherwise the performance will be much worse. Regardless of the overhead, this GPU method become faster than all CPU methods at (5^3=125 atoms). Interestingly, after 1000 atoms, the scaling become better, I guess because it starts to utilize the power of thousands of GPU cores more efficiently.
4. [Gknl] This method has so far the best performance overall. The efficient kernal running on GPU figures out which data block to compute (one per atom), compute the force and directly updates the coordinates. The performance was surprisingly much better than Gvec although both avoids memory copies as much as possible.

With a good GPU, the Gknl method can run simulation on a big cube pretty fast, without any cutoff method:
```
$ python simple_LJ.py -c 8 -m Gknl -n 30000 -s
Building a 8x8x8 cube with 512 atoms
Running simulation of 30000 steps with Gknl method
   30000 steps complete
Finished in 11.207 s
```

And the resulting trajectory can be visualized by vmd:
```
$ vmd traj.xyz
```
