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

# below are ploting part
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

for m in result:
    x = [n**3 for n in result[m].keys()]
    y = result[m].values()
    ax.plot(x, y, 'o-', label=m)

axins = ax.inset_axes([0.3, 0.55, 0.47, 0.43])
for m in result:
    x = [n**3 for n in result[m].keys()]
    y = result[m].values()
    axins.plot(x, y, 'o-')

axins.set_xticks([n**3 for n in cube_sizes])
axins.set_xticklabels([rf'{n}$^3$' for n in cube_sizes])
axins.set_xlim(0, 150)
axins.set_ylim(0, 0.6)

plt.xlim(0, 1050)
plt.ylim(0, 30)
plt.legend()
ax.indicate_inset_zoom(axins)
plt.xticks([n**3 for n in cube_sizes], [rf'{n}$^3$' for n in cube_sizes])
plt.xlabel(r'N atoms (size$^3$)')
plt.ylabel('Time cost (seconds)')
plt.title(f'Benchmark with {n_steps} steps of LJ simulation')

plt.savefig('benchmark.pdf')

print("Result ploted as benchmark.pdf")