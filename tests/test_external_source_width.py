"""Source width of external sources: Gaussian (density width) vs point charges."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import torch
from scipy.constants import pi
from scipy.special import erf

torch.serialization.add_safe_globals([slice])

from graph_longrange.energy import GTOElectrostaticEnergy
from graph_longrange.external_source_energy import (
    GTOElectrostaticCrossEnergy,
    GTOElectrostaticExternalSourceEnergy,
)
from graph_longrange.external_source_features import (
    GTOElectrostaticExternalSourceFeatures,
)
from graph_longrange.external_source_realspace import bipartite_pairs_within_cutoff
from graph_longrange.features import GTOElectrostaticFeatures
from graph_longrange.kspace import compute_k_vectors_flat
from graph_longrange.utils import FIELD_CONSTANT

needs_torchpme = pytest.mark.skipif(
    importlib.util.find_spec("torchpme") is None, reason="torch-pme is not installed"
)

SIGMA_D = 1.0
FEATURE_WIDTHS = [1.0, 1.6]
KSPACE_CUTOFF = 2.5


@pytest.fixture(autouse=True)
def _float64():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


def _features_block(density_max_l, feature_max_l, pbc_handling, density_width=SIGMA_D):
    return GTOElectrostaticFeatures(
        density_max_l=density_max_l,
        density_smearing_width=density_width,
        feature_max_l=feature_max_l,
        feature_smearing_widths=FEATURE_WIDTHS,
        include_self_interaction=False,
        kspace_cutoff=KSPACE_CUTOFF,
        pbc_handling=pbc_handling,
    )


def _energy_block(density_max_l, pbc_handling):
    return GTOElectrostaticEnergy(
        density_max_l=density_max_l,
        density_smearing_width=SIGMA_D,
        kspace_cutoff=KSPACE_CUTOFF,
        include_self_interaction=False,
        pbc_handling=pbc_handling,
    )


def _system(n_src, n_tgt, m_src, m_tgt, cell, seed=0):
    generator = torch.Generator().manual_seed(seed)
    src = torch.rand(n_src, 3, generator=generator) @ cell
    tgt = torch.rand(n_tgt, 3, generator=generator) @ cell
    return {
        "src_positions": src,
        "src_batch": torch.zeros(n_src, dtype=torch.long),
        "tgt_positions": tgt,
        "tgt_batch": torch.zeros(n_tgt, dtype=torch.long),
        "source_feats": torch.randn(n_src, m_src, generator=generator),
        "target_feats": torch.randn(n_tgt, m_tgt, generator=generator),
    }


def _geometry(cell, periodic):
    cells = cell.unsqueeze(0)
    if periodic:
        rcells = 2 * pi * torch.linalg.inv(cells).transpose(-1, -2)
        k_vectors, k_norm2, k_vector_batch, k0_mask = compute_k_vectors_flat(
            KSPACE_CUTOFF, cells, rcells
        )
    else:
        k_vectors = torch.empty((0, 3))
        k_norm2 = torch.empty((0,))
        k_vector_batch = torch.empty((0,), dtype=torch.long)
        k0_mask = torch.empty((0,))
    return {
        "k_vectors": k_vectors,
        "k_norm2": k_norm2,
        "k_vector_batch": k_vector_batch,
        "k0_mask": k0_mask,
        "volume": torch.det(cells).abs(),
        "pbc": torch.tensor([[periodic] * 3]),
    }


def _field(block, geometry, system, cell=None):
    return block.forward_source_target(
        **geometry,
        source_feats=system["source_feats"],
        src_positions=system["src_positions"],
        src_batch=system["src_batch"],
        tgt_positions=system["tgt_positions"],
        tgt_batch=system["tgt_batch"],
        cell=cell,
    )


def _cross(block, geometry, system, cell=None):
    return block(
        **geometry,
        source_feats=system["source_feats"],
        source_positions=system["src_positions"],
        source_batch=system["src_batch"],
        target_feats=system["target_feats"],
        target_positions=system["tgt_positions"],
        target_batch=system["tgt_batch"],
        cell=cell,
    )


# --------------------------------------------------------------------------- pairs


def test_pairs_within_cutoff_include_periodic_images():
    cell = torch.diag(torch.tensor([6.0, 6.0, 6.0]))
    src = torch.tensor([[5.8, 3.0, 3.0]])
    tgt = torch.tensor([[0.4, 3.0, 3.0]])
    zero = torch.zeros(1, dtype=torch.long)
    edges, shifts = bipartite_pairs_within_cutoff(
        src, zero, tgt, zero, cutoff=2.0, cell=cell, pbc=torch.tensor([[True] * 3])
    )
    # only the image shifted by -L is within 2 A (distance 0.6)
    assert edges.shape == (2, 1)
    torch.testing.assert_close(shifts, torch.tensor([[-6.0, 0.0, 0.0]]))
    edges, shifts = bipartite_pairs_within_cutoff(
        src, zero, tgt, zero, cutoff=2.0, cell=None, pbc=torch.tensor([[False] * 3])
    )
    assert edges.shape == (2, 0)
    # a cutoff larger than the box collects several images
    edges, shifts = bipartite_pairs_within_cutoff(
        src, zero, tgt, zero, cutoff=13.0, cell=cell, pbc=torch.tensor([[True] * 3])
    )
    distances = torch.linalg.norm(src[edges[0]] + shifts - tgt[edges[1]], dim=-1)
    assert bool((distances <= 13.0).all()) and edges.shape[1] > 20


# ----------------------------------------------------------------- feature block


@pytest.mark.parametrize("density_max_l,feature_max_l", [(0, 0), (1, 1), (1, 0), (0, 1)])
@pytest.mark.parametrize("periodic", [False, True])
def test_density_width_sources_reproduce_default(density_max_l, feature_max_l, periodic):
    cell = torch.diag(torch.tensor([9.0, 10.0, 11.0]))
    handling = "pbc" if periodic else "realspace"
    base = _features_block(density_max_l, feature_max_l, handling)
    system = _system(7, 4, (density_max_l + 1) ** 2, 1, cell)
    geometry = _geometry(cell, periodic)
    default = GTOElectrostaticExternalSourceFeatures.from_features(base)
    same = GTOElectrostaticExternalSourceFeatures.from_features(base, source_width=SIGMA_D)
    torch.testing.assert_close(
        _field(same, geometry, system, cell), _field(default, geometry, system, cell), rtol=0, atol=0
    )


def test_point_sources_match_brute_force_realspace():
    cell = torch.diag(torch.tensor([9.0, 9.0, 9.0]))
    base = _features_block(0, 0, "realspace")
    system = _system(6, 3, 1, 1, cell)
    geometry = _geometry(cell, periodic=False)
    block = GTOElectrostaticExternalSourceFeatures.from_features(base, source_width=0.0)
    got = _field(block, geometry, system).numpy()
    src = system["src_positions"].numpy()
    tgt = system["tgt_positions"].numpy()
    q = system["source_feats"][:, 0].numpy()
    l0 = base.realspace_features.l0_factors.numpy()
    expected = np.zeros_like(got)
    for t in range(tgt.shape[0]):
        d = np.linalg.norm(src - tgt[t], axis=1)
        for r, sigma_f in enumerate(FEATURE_WIDTHS):
            # point charge seen through a Gaussian probe of width sigma_f
            expected[t, r] = l0[r] * FIELD_CONSTANT / (4 * pi) * np.sum(
                q * erf(d / np.sqrt(2.0 * sigma_f**2)) / (d + 1e-6)  # the block's regulariser
            )
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("density_max_l,feature_max_l", [(0, 0), (1, 1)])
def test_point_sources_equal_vanishing_density_width_realspace(density_max_l, feature_max_l):
    cell = torch.diag(torch.tensor([9.0, 9.0, 9.0]))
    system = _system(6, 3, (density_max_l + 1) ** 2, 1, cell)
    geometry = _geometry(cell, periodic=False)
    narrow = GTOElectrostaticExternalSourceFeatures.from_features(
        _features_block(density_max_l, feature_max_l, "realspace"), source_width=0.0
    )
    reference = GTOElectrostaticExternalSourceFeatures.from_features(
        _features_block(density_max_l, feature_max_l, "realspace", density_width=1e-9)
    )
    torch.testing.assert_close(
        _field(narrow, geometry, system), _field(reference, geometry, system), rtol=1e-9, atol=1e-10
    )


@pytest.mark.parametrize("density_max_l,feature_max_l", [(0, 0), (1, 1)])
def test_periodic_width_correction_matches_realspace_in_large_box(density_max_l, feature_max_l):
    # In a box much larger than every smearing length, the periodic result
    # for point sources must agree with the real-space point-source sum.
    cell = torch.diag(torch.tensor([60.0, 60.0, 60.0]))
    system = _system(5, 3, (density_max_l + 1) ** 2, 1, cell)
    # keep everything inside a 10 A ball so no image is within range
    centre = torch.tensor([30.0, 30.0, 30.0])
    system["src_positions"] = centre + 5.0 * (system["src_positions"] / 60.0 - 0.5)
    system["tgt_positions"] = centre + 5.0 * (system["tgt_positions"] / 60.0 - 0.5)
    periodic = GTOElectrostaticExternalSourceFeatures.from_features(
        _features_block(density_max_l, feature_max_l, "pbc"), source_width=0.0
    )
    realspace = GTOElectrostaticExternalSourceFeatures.from_features(
        _features_block(density_max_l, feature_max_l, "realspace"), source_width=0.0
    )
    got = _field(periodic, _geometry(cell, True), system, cell)
    expected = _field(realspace, _geometry(cell, False), system)
    # the k-space sum itself is only converged to ~1e-4 relative at this cutoff;
    # the width correction must not change that agreement
    default_p = GTOElectrostaticExternalSourceFeatures.from_features(
        _features_block(density_max_l, feature_max_l, "pbc")
    )
    default_r = GTOElectrostaticExternalSourceFeatures.from_features(
        _features_block(density_max_l, feature_max_l, "realspace")
    )
    kspace_error = (
        _field(default_p, _geometry(cell, True), system, cell)
        - _field(default_r, _geometry(cell, False), system)
    ).abs().max()
    assert (got - expected).abs().max() <= 2.0 * float(kspace_error) + 1e-8
    # and the point-charge field is visibly different from the smeared one
    assert (expected - _field(default_r, _geometry(cell, False), system)).abs().max() > 1e-3


def test_periodic_width_correction_is_invariant_to_source_image():
    cell = torch.diag(torch.tensor([8.0, 8.0, 8.0]))
    block = GTOElectrostaticExternalSourceFeatures.from_features(
        _features_block(1, 1, "pbc"), source_width=0.0
    )
    system = _system(4, 3, 4, 1, cell)
    geometry = _geometry(cell, True)
    reference = _field(block, geometry, system, cell)
    shifted = dict(system)
    shifted["src_positions"] = system["src_positions"] + torch.tensor([8.0, -16.0, 0.0])
    torch.testing.assert_close(_field(block, geometry, shifted, cell), reference, rtol=1e-10, atol=1e-10)


def test_width_correction_vanishes_beyond_cutoff():
    cell = torch.diag(torch.tensor([40.0, 40.0, 40.0]))
    base = _features_block(1, 1, "pbc")
    system = _system(3, 2, 4, 1, cell)
    system["src_positions"] = torch.tensor([[1.0, 1.0, 1.0], [1.5, 0.5, 1.0], [0.8, 1.2, 1.4]])
    system["tgt_positions"] = torch.tensor([[20.0, 20.0, 20.0], [20.5, 19.5, 20.0]])
    geometry = _geometry(cell, True)
    default = _field(GTOElectrostaticExternalSourceFeatures.from_features(base), geometry, system, cell)
    narrow = _field(
        GTOElectrostaticExternalSourceFeatures.from_features(base, source_width=0.0, correction_cutoff=10.0),
        geometry,
        system,
        cell,
    )
    torch.testing.assert_close(narrow, default, rtol=0, atol=0)


def test_width_correction_gradients_match_finite_differences():
    cell = torch.diag(torch.tensor([8.0, 8.0, 8.0]))
    block = GTOElectrostaticExternalSourceFeatures.from_features(
        _features_block(1, 1, "pbc"), source_width=0.0
    )
    system = _system(4, 3, 4, 1, cell)
    geometry = _geometry(cell, True)
    weights = torch.randn(3, 4 * len(FEATURE_WIDTHS), generator=torch.Generator().manual_seed(3))

    def scalar(src_positions):
        trial = dict(system)
        trial["src_positions"] = src_positions
        return (weights * _field(block, geometry, trial, cell)).sum()

    positions = system["src_positions"].clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(scalar(positions), positions)
    h = 1e-5
    for i, a in [(0, 0), (2, 1), (3, 2)]:
        plus = system["src_positions"].clone()
        plus[i, a] += h
        minus = system["src_positions"].clone()
        minus[i, a] -= h
        numeric = (scalar(plus) - scalar(minus)) / (2 * h)
        torch.testing.assert_close(grad[i, a], numeric, rtol=1e-6, atol=1e-7)


@needs_torchpme
def test_pme_and_direct_agree_with_point_sources():
    cell = torch.diag(torch.tensor([9.0, 10.0, 11.0]))
    base = _features_block(1, 1, "pbc")
    system = _system(12, 4, 4, 1, cell)
    geometry = _geometry(cell, True)
    direct = GTOElectrostaticExternalSourceFeatures.from_features(base, source_width=0.0)
    pme = GTOElectrostaticExternalSourceFeatures.from_features(base, method="PME", source_width=0.0)
    direct_default = GTOElectrostaticExternalSourceFeatures.from_features(base)
    pme_default = GTOElectrostaticExternalSourceFeatures.from_features(base, method="PME")
    # The width correction is the same real-space term on both paths, so the
    # PME-vs-direct discrepancy must be exactly the mesh's own aliasing error.
    point_gap = _field(pme, geometry, system, cell) - _field(direct, geometry, system, cell)
    default_gap = _field(pme_default, geometry, system, cell) - _field(direct_default, geometry, system, cell)
    torch.testing.assert_close(point_gap, default_gap, rtol=0, atol=1e-10)
    assert float(point_gap.abs().max()) < 1e-3 * float(_field(direct, geometry, system, cell).abs().max())


def test_periodic_point_sources_need_the_cell():
    cell = torch.diag(torch.tensor([9.0, 9.0, 9.0]))
    block = GTOElectrostaticExternalSourceFeatures.from_features(_features_block(0, 0, "pbc"), source_width=0.0)
    system = _system(3, 2, 1, 1, cell)
    with pytest.raises(ValueError, match="cell"):
        _field(block, _geometry(cell, True), system, cell=None)


def test_negative_source_width_is_rejected():
    with pytest.raises(ValueError):
        GTOElectrostaticExternalSourceFeatures.from_features(_features_block(0, 0, "pbc"), source_width=-0.1)


# ------------------------------------------------------------------ energy block


@pytest.mark.parametrize("density_max_l", [0, 1])
@pytest.mark.parametrize("periodic", [False, True])
def test_density_width_targets_reproduce_default_energy(density_max_l, periodic):
    cell = torch.diag(torch.tensor([9.0, 10.0, 11.0]))
    handling = "pbc" if periodic else "realspace"
    energy = _energy_block(density_max_l, handling)
    system = _system(6, 4, (density_max_l + 1) ** 2, (density_max_l + 1) ** 2, cell)
    geometry = _geometry(cell, periodic)
    default = GTOElectrostaticCrossEnergy.from_energy(energy)
    same = GTOElectrostaticCrossEnergy.from_energy(energy, target_width=SIGMA_D)
    torch.testing.assert_close(
        _cross(same, geometry, system, cell), _cross(default, geometry, system, cell), rtol=0, atol=0
    )


def test_point_targets_match_brute_force_energy_realspace():
    cell = torch.diag(torch.tensor([9.0, 9.0, 9.0]))
    energy = _energy_block(0, "realspace")
    system = _system(5, 4, 1, 1, cell)
    block = GTOElectrostaticCrossEnergy.from_energy(energy, target_width=0.0)
    got = float(_cross(block, _geometry(cell, False), system))
    src = system["src_positions"].numpy()
    tgt = system["tgt_positions"].numpy()
    qs = system["source_feats"][:, 0].numpy()
    qt = system["target_feats"][:, 0].numpy()
    expected = 0.0
    for s in range(src.shape[0]):
        d = np.linalg.norm(tgt - src[s], axis=1)
        # Gaussian (sigma_d) source against point targets
        expected += np.sum(qs[s] * qt * erf(d / np.sqrt(2.0 * SIGMA_D**2)) / (d + 1e-6))  # block regulariser
    expected *= FIELD_CONSTANT / (4 * pi)
    assert got == pytest.approx(expected, rel=1e-10, abs=1e-12)


@pytest.mark.parametrize("density_max_l", [0, 1])
def test_periodic_point_targets_match_realspace_in_large_box(density_max_l):
    cell = torch.diag(torch.tensor([60.0, 60.0, 60.0]))
    m = (density_max_l + 1) ** 2
    system = _system(5, 4, m, m, cell)
    centre = torch.tensor([30.0, 30.0, 30.0])
    system["src_positions"] = centre + 5.0 * (system["src_positions"] / 60.0 - 0.5)
    system["tgt_positions"] = centre + 5.0 * (system["tgt_positions"] / 60.0 - 0.5)
    periodic = GTOElectrostaticCrossEnergy.from_energy(_energy_block(density_max_l, "pbc"), target_width=0.0)
    realspace = GTOElectrostaticCrossEnergy.from_energy(_energy_block(density_max_l, "realspace"), target_width=0.0)
    default_p = GTOElectrostaticCrossEnergy.from_energy(_energy_block(density_max_l, "pbc"))
    default_r = GTOElectrostaticCrossEnergy.from_energy(_energy_block(density_max_l, "realspace"))
    got = _cross(periodic, _geometry(cell, True), system, cell)
    expected = _cross(realspace, _geometry(cell, False), system)
    kspace_error = (
        _cross(default_p, _geometry(cell, True), system, cell) - _cross(default_r, _geometry(cell, False), system)
    ).abs().max()
    assert float((got - expected).abs().max()) <= 2.0 * float(kspace_error) + 1e-8
    assert float((expected - _cross(default_r, _geometry(cell, False), system)).abs()) > 1e-3


def test_energy_width_correction_gradients_match_finite_differences():
    cell = torch.diag(torch.tensor([8.0, 8.0, 8.0]))
    block = GTOElectrostaticCrossEnergy.from_energy(_energy_block(1, "pbc"), target_width=0.0)
    system = _system(4, 3, 4, 4, cell)
    geometry = _geometry(cell, True)

    def scalar(tgt_positions):
        trial = dict(system)
        trial["tgt_positions"] = tgt_positions
        return _cross(block, geometry, trial, cell).sum()

    positions = system["tgt_positions"].clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(scalar(positions), positions)
    h = 1e-5
    for i, a in [(0, 0), (1, 2), (2, 1)]:
        plus = system["tgt_positions"].clone()
        plus[i, a] += h
        minus = system["tgt_positions"].clone()
        minus[i, a] -= h
        numeric = (scalar(plus) - scalar(minus)) / (2 * h)
        torch.testing.assert_close(grad[i, a], numeric, rtol=1e-6, atol=1e-7)


def test_periodic_point_targets_need_the_cell():
    cell = torch.diag(torch.tensor([9.0, 9.0, 9.0]))
    block = GTOElectrostaticCrossEnergy.from_energy(_energy_block(0, "pbc"), target_width=0.0)
    system = _system(3, 2, 1, 1, cell)
    with pytest.raises(ValueError, match="cell"):
        _cross(block, _geometry(cell, True), system, cell=None)


def test_external_source_energy_wrapper_passes_cell_and_width():
    cell = torch.diag(torch.tensor([9.0, 10.0, 11.0]))
    energy = _energy_block(1, "pbc")
    system = _system(5, 3, 4, 4, cell)  # the cross block wants equal multipole widths
    geometry = _geometry(cell, True)
    wrapped = GTOElectrostaticExternalSourceEnergy.from_energy(energy, external_width=0.0)
    wrapped.set_external_sources(
        external_feats=system["target_feats"],
        external_positions=system["tgt_positions"],
        external_batch=system["tgt_batch"],
        cell=cell,
    )
    total = wrapped(
        **geometry,
        source_feats=system["source_feats"],
        node_positions=system["src_positions"],
        batch=system["src_batch"],
    )
    native = energy(
        **geometry,
        source_feats=system["source_feats"],
        node_positions=system["src_positions"],
        batch=system["src_batch"],
    )
    cross = _cross(GTOElectrostaticCrossEnergy.from_energy(energy, target_width=0.0), geometry, system, cell)
    torch.testing.assert_close(total, native + cross, rtol=1e-12, atol=1e-12)


# ------------------------------------------------- review follow-ups (round 1)


def _image_grid(cell, n):
    offsets = torch.cartesian_prod(*[torch.arange(-n, n + 1)] * 3).to(cell.dtype)
    return offsets @ cell


def test_pairs_triclinic_cell_match_brute_force_images():
    cell = torch.tensor([[7.0, 0.0, 0.0], [2.5, 6.0, 0.0], [-1.0, 1.5, 6.5]])
    system = _system(6, 4, 1, 1, cell, seed=5)
    system["src_positions"] = system["src_positions"] + torch.tensor([[23.0, -11.0, 4.0]])  # far outside
    pbc = torch.tensor([[True] * 3])
    cutoff = 9.0
    edges, shifts = bipartite_pairs_within_cutoff(
        system["src_positions"], system["src_batch"], system["tgt_positions"], system["tgt_batch"], cutoff, cell, pbc
    )
    got = {(int(s), int(t), tuple(torch.round(sh, decimals=6).tolist())) for s, t, sh in zip(edges[0], edges[1], shifts)}
    expected = set()
    for s in range(6):
        for t in range(4):
            for image in _image_grid(cell, 6):
                d = torch.linalg.norm(system["src_positions"][s] + image - system["tgt_positions"][t])
                if d <= cutoff:
                    expected.add((s, t, tuple(torch.round(image, decimals=6).tolist())))
    assert got == expected
    # chunking must not change the pair set
    edges_1, shifts_1 = bipartite_pairs_within_cutoff(
        system["src_positions"], system["src_batch"], system["tgt_positions"], system["tgt_batch"], cutoff, cell, pbc, pair_budget=1
    )
    got_1 = {(int(s), int(t), tuple(torch.round(sh, decimals=6).tolist())) for s, t, sh in zip(edges_1[0], edges_1[1], shifts_1)}
    assert got_1 == expected


def test_pairs_partial_periodicity_has_no_images_along_open_axis():
    cell = torch.diag(torch.tensor([6.0, 6.0, 6.0]))
    src = torch.tensor([[3.0, 3.0, 5.8]])
    tgt = torch.tensor([[3.0, 3.0, 0.4]])
    zero = torch.zeros(1, dtype=torch.long)
    edges, _ = bipartite_pairs_within_cutoff(src, zero, tgt, zero, 2.0, cell, torch.tensor([[True, True, False]]))
    assert edges.shape[1] == 0


@pytest.mark.parametrize("width", [0.0, 0.5, 2.0])
def test_periodic_width_change_matches_explicit_image_sum_features(width):
    # (width-w result) - (default result) must equal the explicit image sum of the
    # kernel difference, independently of the reciprocal-space evaluation.
    cell = torch.diag(torch.tensor([7.0, 8.0, 9.0]))
    base = _features_block(0, 0, "pbc")
    system = _system(5, 3, 1, 1, cell, seed=2)
    geometry = _geometry(cell, True)
    default = _field(GTOElectrostaticExternalSourceFeatures.from_features(base), geometry, system, cell)
    changed = _field(
        GTOElectrostaticExternalSourceFeatures.from_features(base, source_width=width, correction_cutoff=30.0),
        geometry,
        system,
        cell,
    )
    src = system["src_positions"].numpy()
    tgt = system["tgt_positions"].numpy()
    q = system["source_feats"][:, 0].numpy()
    l0 = base.realspace_features.l0_factors.numpy()
    images = _image_grid(cell, 5).numpy()
    expected = np.zeros_like(default.numpy())
    for t in range(tgt.shape[0]):
        for s in range(src.shape[0]):
            d = np.linalg.norm(src[s] + images - tgt[t], axis=1)
            d = d[d <= 30.0]
            for r, sigma_f in enumerate(FEATURE_WIDTHS):
                k_new = erf(d / np.sqrt(2.0 * (width**2 + sigma_f**2))) / (d + 1e-6)
                k_ref = erf(d / np.sqrt(2.0 * (SIGMA_D**2 + sigma_f**2))) / (d + 1e-6)
                expected[t, r] += l0[r] * FIELD_CONSTANT / (4 * pi) * q[s] * np.sum(k_new - k_ref)
    np.testing.assert_allclose((changed - default).numpy(), expected, rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("width", [0.0, 2.0])
def test_periodic_width_change_matches_explicit_image_sum_energy(width):
    cell = torch.diag(torch.tensor([7.0, 8.0, 9.0]))
    energy = _energy_block(0, "pbc")
    system = _system(5, 4, 1, 1, cell, seed=4)
    geometry = _geometry(cell, True)
    default = _cross(GTOElectrostaticCrossEnergy.from_energy(energy), geometry, system, cell)
    changed = _cross(
        GTOElectrostaticCrossEnergy.from_energy(energy, target_width=width, correction_cutoff=30.0), geometry, system, cell
    )
    src = system["src_positions"].numpy()
    tgt = system["tgt_positions"].numpy()
    qs = system["source_feats"][:, 0].numpy()
    qt = system["target_feats"][:, 0].numpy()
    images = _image_grid(cell, 5).numpy()
    expected = 0.0
    for s in range(src.shape[0]):
        for t in range(tgt.shape[0]):
            d = np.linalg.norm(src[s] + images - tgt[t], axis=1)
            d = d[d <= 30.0]
            k_new = erf(d / np.sqrt(2.0 * (SIGMA_D**2 + width**2))) / (d + 1e-6)
            k_ref = erf(d / np.sqrt(2.0 * (2.0 * SIGMA_D**2))) / (d + 1e-6)
            expected += qs[s] * qt[t] * np.sum(k_new - k_ref)
    expected *= FIELD_CONSTANT / (4 * pi)
    assert float(changed - default) == pytest.approx(expected, rel=1e-9, abs=1e-11)


@pytest.mark.parametrize("width", [0.5, 2.0])
def test_broader_and_intermediate_widths_periodic_vs_realspace(width):
    cell = torch.diag(torch.tensor([60.0, 60.0, 60.0]))
    centre = torch.tensor([30.0, 30.0, 30.0])
    system = _system(5, 3, 4, 4, cell, seed=7)
    system["src_positions"] = centre + 5.0 * (system["src_positions"] / 60.0 - 0.5)
    system["tgt_positions"] = centre + 5.0 * (system["tgt_positions"] / 60.0 - 0.5)
    # features
    got = _field(GTOElectrostaticExternalSourceFeatures.from_features(_features_block(1, 1, "pbc"), source_width=width), _geometry(cell, True), system, cell)
    expected = _field(GTOElectrostaticExternalSourceFeatures.from_features(_features_block(1, 1, "realspace"), source_width=width), _geometry(cell, False), system)
    kspace_error = (
        _field(GTOElectrostaticExternalSourceFeatures.from_features(_features_block(1, 1, "pbc")), _geometry(cell, True), system, cell)
        - _field(GTOElectrostaticExternalSourceFeatures.from_features(_features_block(1, 1, "realspace")), _geometry(cell, False), system)
    ).abs().max()
    assert float((got - expected).abs().max()) <= 2.0 * float(kspace_error) + 1e-8
    # energy
    got_e = _cross(GTOElectrostaticCrossEnergy.from_energy(_energy_block(1, "pbc"), target_width=width), _geometry(cell, True), system, cell)
    expected_e = _cross(GTOElectrostaticCrossEnergy.from_energy(_energy_block(1, "realspace"), target_width=width), _geometry(cell, False), system)
    kspace_error_e = (
        _cross(GTOElectrostaticCrossEnergy.from_energy(_energy_block(1, "pbc")), _geometry(cell, True), system, cell)
        - _cross(GTOElectrostaticCrossEnergy.from_energy(_energy_block(1, "realspace")), _geometry(cell, False), system)
    ).abs().max()
    assert float((got_e - expected_e).abs().max()) <= 2.0 * float(kspace_error_e) + 1e-8


def test_default_cutoff_grows_with_broad_widths():
    narrow = GTOElectrostaticExternalSourceFeatures.from_features(_features_block(1, 1, "pbc"), source_width=0.0)
    broad = GTOElectrostaticExternalSourceFeatures.from_features(_features_block(1, 1, "pbc"), source_width=10.0)
    assert broad.effective_correction_cutoff > narrow.effective_correction_cutoff
    assert broad.effective_correction_cutoff == pytest.approx(4.0 * np.sqrt(10.0**2 + max(FEATURE_WIDTHS) ** 2))
    e_broad = GTOElectrostaticCrossEnergy.from_energy(_energy_block(1, "pbc"), target_width=10.0)
    assert e_broad.effective_correction_cutoff == pytest.approx(4.0 * np.sqrt(SIGMA_D**2 + 10.0**2))


def test_feature_width_correction_gradients_wrt_targets():
    cell = torch.diag(torch.tensor([8.0, 8.0, 8.0]))
    block = GTOElectrostaticExternalSourceFeatures.from_features(_features_block(1, 1, "pbc"), source_width=0.0)
    system = _system(4, 3, 4, 1, cell, seed=9)
    geometry = _geometry(cell, True)
    weights = torch.randn(3, 4 * len(FEATURE_WIDTHS), generator=torch.Generator().manual_seed(11))

    def scalar(tgt_positions):
        trial = dict(system)
        trial["tgt_positions"] = tgt_positions
        return (weights * _field(block, geometry, trial, cell)).sum()

    positions = system["tgt_positions"].clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(scalar(positions), positions)
    h = 1e-5
    for i, a in [(0, 1), (1, 0), (2, 2)]:
        plus = system["tgt_positions"].clone(); plus[i, a] += h
        minus = system["tgt_positions"].clone(); minus[i, a] -= h
        torch.testing.assert_close(grad[i, a], (scalar(plus) - scalar(minus)) / (2 * h), rtol=1e-6, atol=1e-7)


def test_energy_width_correction_gradients_wrt_sources():
    cell = torch.diag(torch.tensor([8.0, 8.0, 8.0]))
    block = GTOElectrostaticCrossEnergy.from_energy(_energy_block(1, "pbc"), target_width=0.0)
    system = _system(4, 3, 4, 4, cell, seed=13)
    geometry = _geometry(cell, True)

    def scalar(src_positions):
        trial = dict(system)
        trial["src_positions"] = src_positions
        return _cross(block, geometry, trial, cell).sum()

    positions = system["src_positions"].clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(scalar(positions), positions)
    h = 1e-5
    for i, a in [(0, 2), (1, 1), (3, 0)]:
        plus = system["src_positions"].clone(); plus[i, a] += h
        minus = system["src_positions"].clone(); minus[i, a] -= h
        torch.testing.assert_close(grad[i, a], (scalar(plus) - scalar(minus)) / (2 * h), rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0])
def test_non_finite_or_negative_widths_rejected(bad):
    with pytest.raises(ValueError):
        GTOElectrostaticExternalSourceFeatures.from_features(_features_block(0, 0, "pbc"), source_width=bad)
    with pytest.raises(ValueError):
        GTOElectrostaticCrossEnergy.from_energy(_energy_block(0, "pbc"), target_width=bad)


@pytest.mark.parametrize("bad", [0.0, -3.0, float("nan"), float("inf")])
def test_bad_cutoffs_rejected(bad):
    with pytest.raises(ValueError):
        GTOElectrostaticExternalSourceFeatures.from_features(_features_block(0, 0, "pbc"), source_width=0.0, correction_cutoff=bad)
    with pytest.raises(ValueError):
        GTOElectrostaticCrossEnergy.from_energy(_energy_block(0, "pbc"), target_width=0.0, correction_cutoff=bad)


def test_width_correction_rejects_higher_angular_orders():
    # The library's real-space block only exists for l <= 1 (constructing a base
    # with feature_max_l=2 already raises NotImplementedError), so emulate a
    # higher-order block to check the wrapper's own guard.
    base = _features_block(1, 1, "pbc")
    base.realspace_features.projection_max_l = 2
    GTOElectrostaticExternalSourceFeatures.from_features(base)  # default width still fine
    with pytest.raises(ValueError, match="max_l <= 1"):
        GTOElectrostaticExternalSourceFeatures.from_features(base, source_width=0.0)


# ------------------------------------------------- review follow-ups (round 2)


@pytest.mark.parametrize("density_max_l,feature_max_l", [(0, 0), (1, 1), (1, 0), (0, 1)])
def test_feature_correction_is_independent_of_chunking(density_max_l, feature_max_l):
    cell = torch.diag(torch.tensor([8.0, 9.0, 10.0]))
    base = _features_block(density_max_l, feature_max_l, "pbc")
    system = _system(9, 5, (density_max_l + 1) ** 2, 1, cell, seed=21)
    geometry = _geometry(cell, True)
    big = GTOElectrostaticExternalSourceFeatures.from_features(base, source_width=0.3)
    tiny = GTOElectrostaticExternalSourceFeatures.from_features(base, source_width=0.3, correction_budget_bytes=1)
    torch.testing.assert_close(_field(tiny, geometry, system, cell), _field(big, geometry, system, cell), rtol=1e-9, atol=1e-10)  # summation order


@pytest.mark.parametrize("density_max_l", [0, 1])
def test_energy_correction_is_independent_of_chunking(density_max_l):
    cell = torch.diag(torch.tensor([8.0, 9.0, 10.0]))
    m = (density_max_l + 1) ** 2
    system = _system(9, 5, m, m, cell, seed=22)
    geometry = _geometry(cell, True)
    big = GTOElectrostaticCrossEnergy.from_energy(_energy_block(density_max_l, "pbc"), target_width=0.3)
    tiny = GTOElectrostaticCrossEnergy.from_energy(
        _energy_block(density_max_l, "pbc"), target_width=0.3, realspace_chunk_size=1, realspace_chunk_budget_bytes=1
    )
    torch.testing.assert_close(_cross(tiny, geometry, system, cell), _cross(big, geometry, system, cell), rtol=1e-9, atol=1e-10)


def test_width_correction_rejects_higher_density_order_too():
    base = _features_block(1, 1, "pbc")
    base.realspace_features.density_max_l = 2
    GTOElectrostaticExternalSourceFeatures.from_features(base, source_width=SIGMA_D)  # equal width: no correction
    with pytest.raises(ValueError, match="max_l <= 1"):
        GTOElectrostaticExternalSourceFeatures.from_features(base, source_width=0.0)
