"""Ch.1  The fifth modality: a graph. Build the neighbour graph of crystalline Si.

Book references
    tab:modalities (row "graph"), exercise ex:matching note on item (d)

Atoms are nodes; an edge joins two atoms closer than a cutoff. Periodic
boundary conditions are applied with the minimum-image convention.
Lattice constant: experimental a = 5.431 Angstrom (room temperature).
Physics check: in the diamond structure every Si atom has 4 nearest
neighbours at a*sqrt(3)/4.
"""
import itertools

import numpy as np

A = 5.431  # Angstrom, experimental lattice constant of Si

# 8 atoms of the conventional cubic cell of the diamond structure (fractional)
fcc = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])
basis = np.vstack([fcc, fcc + 0.25])
N_REP = 2                                   # 2 x 2 x 2 supercell -> 64 atoms
cells = np.array(list(itertools.product(range(N_REP), repeat=3)))
frac = (basis[None, :, :] + cells[:, None, :]).reshape(-1, 3) / N_REP
box = A * N_REP
pos = frac * box
n = len(pos)

d = pos[:, None, :] - pos[None, :, :]
d -= box * np.round(d / box)                 # minimum image
dist = np.linalg.norm(d, axis=-1)

d_nn = A * np.sqrt(3) / 4
cutoff = 0.5 * (d_nn + A / np.sqrt(2))       # halfway to the 2nd shell (a/sqrt(2))
adj = (dist > 0) & (dist < cutoff)
edges = np.argwhere(np.triu(adj))
degree = adj.sum(1)

print(f"atoms (nodes)            : {n}")
print(f"edges                    : {len(edges)}")
print(f"degree of every node     : {sorted(set(degree.tolist()))}  (diamond -> 4)")
print(f"nearest-neighbour length : {dist[adj].min():.3f} A  "
      f"(a*sqrt(3)/4 = {d_nn:.3f} A)")
print(f"edges per atom           : {len(edges) / n:.1f}  (= 4/2, each bond shared)")
print("\nfirst 5 edges (i, j, length in A):")
for i, j in edges[:5]:
    print(f"  {i:2d} {j:2d} {dist[i, j]:.3f}")
