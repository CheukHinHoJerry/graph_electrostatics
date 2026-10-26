"""PolarMACE in a periodic MM environment, with the MM field evaluated by PME.

A QM water molecule is modelled with MACE-POLAR-1-S and embedded in a box of
MM point charges. The MM field enters PolarMACE's electrostatic descriptor and
polarises the QM charge density, and autograd returns forces on both the QM
and the MM atoms. This is the same wiring as the OpenMM-ML PolarMACE
embedding; `method="PME"` keeps the MM side memory-bounded for large
environments.

Requires mace-torch >= 0.3.17 (ACEsuit/mace main; older releases target an
older graph_longrange API) and the optional `torch-pme` package for PME.

usage: python examples/mace_polar_pme.py
"""

import torch
from ase.build import molecule
from mace.calculators.foundations_models import mace_polar

from graph_longrange.external_source_energy import GTOElectrostaticExternalSourceEnergy
from graph_longrange.external_source_features import (
    GTOElectrostaticExternalSourceFeatures,
)

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- PolarMACE with external-source wrappers ---------------------------------
calc = mace_polar("polar-1-s", device=device, default_dtype="float64")
model = calc.models[0]
descriptor = model.electric_potential_descriptor

# PolarMACE has two spin channels; each receives half of the external potential
pme = GTOElectrostaticExternalSourceFeatures.from_features(
    descriptor, external_scale=0.5, method="PME"
)
direct = GTOElectrostaticExternalSourceFeatures.from_features(
    descriptor, external_scale=0.5
)
model.coulomb_energy = GTOElectrostaticExternalSourceEnergy.from_energy(
    model.coulomb_energy
)

# --- QM region: one water molecule in a periodic box ---------------------------
box = 30.0
qm = molecule("H2O", cell=[box, box, box], pbc=True)
qm.center()
qm.info.update(charge=0, spin=1)
data = calc._atoms_to_batch(qm).to_dict()
cell = data["cell"].view(-1, 3, 3)
dtype = data["positions"].dtype

# --- MM environment: neutral point charges, at least 4 A from the QM water -----
n_mm = 3000
mm_positions = torch.rand(4 * n_mm, 3, dtype=dtype, device=device) * box
centre = data["positions"].mean(dim=0)
mm_positions = mm_positions[(mm_positions - centre).norm(dim=-1) > 4.0][:n_mm]
mm_charges = 0.4 * torch.randn(n_mm, dtype=dtype, device=device)
mm_charges -= mm_charges.mean()

# point charges as multipoles [q, p_y, p_z, p_x, ...] with zero higher moments
mm_feats = torch.zeros(
    n_mm, (model.atomic_multipoles_max_l + 1) ** 2, dtype=dtype, device=device
)
mm_feats[:, 0] = mm_charges
mm_batch = torch.zeros(n_mm, dtype=torch.long, device=device)


def evaluate(features_block):
    """Energy, QM forces and MM forces with the given MM-field method."""
    model.electric_potential_descriptor = features_block
    positions = mm_positions.clone().requires_grad_()
    features_block.set_external_sources(mm_feats, positions, mm_batch, cell=cell)
    model.coulomb_energy.set_external_sources(mm_feats, positions, mm_batch)
    try:
        energy = model(dict(data), compute_force=False)["energy"]
        qm_grad, mm_grad = torch.autograd.grad(
            energy.sum(), [data["positions"], positions]
        )
    finally:
        features_block.clear_external_sources()
        model.coulomb_energy.clear_external_sources()
    return energy.detach(), -qm_grad, -mm_grad


energy_vacuum = model(dict(data), compute_force=False)["energy"].detach()
energy, qm_forces, mm_forces = evaluate(pme)
print(f"energy without MM: {energy_vacuum.item():.6f} eV")
print(f"energy with {n_mm} MM charges (PME): {energy.item():.6f} eV")
print(f"max |MM force|: {mm_forces.abs().max():.3e} eV/A")

# --- the same evaluation with the direct sum, for comparison -------------------
energy_d, qm_forces_d, mm_forces_d = evaluate(direct)
print(f"energy difference PME - direct: {(energy - energy_d).abs().item():.1e} eV")
print(f"max QM force difference: {(qm_forces - qm_forces_d).abs().max():.1e} eV/A")
print(f"max MM force difference: {(mm_forces - mm_forces_d).abs().max():.1e} eV/A")
