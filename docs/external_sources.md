# External-source cross energy

`GTOElectrostaticCrossEnergy` evaluates the electrostatic interaction between
two disjoint sets of GTO multipoles.  It is intended for dynamic environments
such as electrostatic ML/MM embedding, while remaining independent of MACE and
OpenMM.

The result is

```text
E_cross(A, B) = E(A + B) - E(A) - E(B)
```

and therefore contains neither `A-A` nor `B-B` interactions.  The optimized
implementation evaluates the damped bipartite interaction directly in real
space and uses separately assembled Fourier densities in reciprocal space.
Both paths use the same `multipoles` normalization as
`GTOElectrostaticEnergy`.

```python
from graph_longrange.external_source_energy import GTOElectrostaticCrossEnergy

cross = GTOElectrostaticCrossEnergy(
    density_max_l=1,
    density_smearing_width=1.0,
    kspace_cutoff=8.0,
    pbc_handling="auto",
)

energy_ab = cross(
    k_vectors=k_vectors,
    k_norm2=k_norm2,
    k_vector_batch=k_vector_batch,
    k0_mask=k0_mask,
    source_feats=source_multipoles,
    source_positions=source_positions,
    source_batch=source_batch,
    target_feats=environment_multipoles,
    target_positions=environment_positions,
    target_batch=environment_batch,
    volume=volume,
    pbc=pbc,
)
```

The two position tensors remain in the autograd graph, so differentiating this
energy produces equal-and-opposite reaction forces on the two subsystems.
