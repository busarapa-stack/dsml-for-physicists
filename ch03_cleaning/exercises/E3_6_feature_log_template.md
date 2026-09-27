# Feature log (E3.6) — template for the README of the AFM feature work

**Final features used**
| feature | unit | why it is physical | assumptions |
|---|---|---|---|
| contact slope dF/dz | nN/nm (= N/m) | stiffness of tip + sample | linear contact region; includes cantilever compliance |
| maximum adhesion | nN | tip–surface adhesion | no smoothing wider than the snap-off |
| work of adhesion ∫F dz | nN·nm (= aJ) | energy to separate tip and surface | baseline removed correctly |

**Features tried and NOT used (with reason)**
| feature | result | reason for not using |
|---|---|---|
| (e.g. raw maximum force) | | depends on the set-point, not on the sample |

**Sensitivity analysis**: how the conclusions change with the smoothing window and the
contact threshold (numbers, not words).

**Statement**: all features that were tried are listed above. Number tried: __ .
