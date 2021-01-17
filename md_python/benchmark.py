#!/usr/bin/env python

"""
Simple script to perform a benchmark
It shows performance of difference methods on various system sizes
"""

import time
import simple_LJ

methods = simple_LJ.SUPPORTED_METHODS.keys()
cube_sizes = range(2, 11)
n_steps = 1000

result = {}
print(f"Running benchmark for {n_steps} steps each")
for method in methods:
    result[method] = {}
    for csize in cube_sizes:
        t0 = time.time()
        simple_LJ.run_md(csize, n_steps, method, verbose=False)
        cost = time.time() - t0
        print(f"{method:4s} method | {csize:3d}^3 = {csize**3:<4d} atoms | cost: {cost:7.3f} s")
        result[method][csize] = cost
        if cost > 10:
            print("Skipping the rest of larger sizes as it takes too long")
            break

import matplotlib.pyplot as plt

for m in result:
    x = [n**3 for n in result[m].keys()]
    y = result[m].values()
    plt.plot(x, y, 'o-', label=m)
plt.legend()
plt.xticks([n**3 for n in cube_sizes], [rf'{n}$^3$' for n in cube_sizes])
plt.xlabel(r'N atoms (size$^3$)')
plt.ylabel('Time cost (seconds)')
plt.title(f'Benchmark with {n_steps} steps of LJ simulation')
plt.savefig('benchmark.pdf')

print("Result ploted as benchmark.pdf")