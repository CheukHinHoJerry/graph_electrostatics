"""PME source side of the external-source features against the direct sum."""

from __future__ import annotations

import pytest
import torch
from scipy.constants import pi

pytest.importorskip("torchpme")

from graph_longrange.external_source_features import (
    GTOElectrostaticExternalSourceFeatures,
)
from graph_longrange.features import GTOElectrostaticFeatures
from graph_longrange.kspace import compute_k_vectors_flat


@pytest.fixture(autouse=True)
def _float64():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


PBC_FLAGS = {
    "pbc": [True, True, True],
    "slab": [True, True, False],
    "molecule_in_box": [False, False, False],
    "mixed_periodic": [True, True, False],
}


def _base(density_max_l, feature_max_l, pbc_handling):
    return GTOElectrostaticFeatures(
        density_max_l=density_max_l,
        density_smearing_width=1.0,
        feature_max_l=feature_max_l,
        feature_smearing_widths=[1.0, 1.6],
        include_self_interaction=False,
        kspace_cutoff=2.5,
        pbc_handling=pbc_handling,
    )


def _cells():
    return torch.stack(
        [
            torch.tensor([[11.0, 0.0, 0.0], [1.5, 10.0, 0.0], [0.5, -0.8, 12.5]]),
            torch.diag(torch.tensor([9.0, 13.0, 10.0])),
        ]
    )


def _system(cells, n_src_per_graph, n_tgt_per_graph, m_dim, seed=0):
    generator = torch.Generator().manual_seed(seed)
    src_pos, src_batch, tgt_pos, tgt_batch = [], [], [], []
    for g, (n_src, n_tgt) in enumerate(zip(n_src_per_graph, n_tgt_per_graph)):
        frac_src = torch.rand(n_src, 3, generator=generator)
        frac_tgt = torch.rand(n_tgt, 3, generator=generator)
        src_pos.append(frac_src @ cells[g])
        tgt_pos.append(frac_tgt @ cells[g])
        src_batch.append(torch.full((n_src,), g, dtype=torch.long))
        tgt_batch.append(torch.full((n_tgt,), g, dtype=torch.long))
    n_src = sum(n_src_per_graph)
    return {
        "src_positions": torch.cat(src_pos),
        "src_batch": torch.cat(src_batch),
        "tgt_positions": torch.cat(tgt_pos),
        "tgt_batch": torch.cat(tgt_batch),
        "source_feats": torch.randn(n_src, m_dim, generator=generator),
    }


def _geometry(cells, pbc_handling, kspace_cutoff=2.5):
    rcells = 2 * pi * torch.linalg.inv(cells).transpose(-1, -2)
    k_vectors, k_norm2, k_vector_batch, k0_mask = compute_k_vectors_flat(
        kspace_cutoff, cells, rcells
    )
    pbc = torch.tensor([PBC_FLAGS[pbc_handling]] * cells.shape[0])
    return {
        "k_vectors": k_vectors,
        "k_norm2": k_norm2,
        "k_vector_batch": k_vector_batch,
        "k0_mask": k0_mask,
        "volume": torch.det(cells).abs(),
        "pbc": pbc,
    }


def _features(block, geometry, system, cell=None):
    return block.forward_source_target(
        **geometry,
        source_feats=system["source_feats"],
        src_positions=system["src_positions"],
        src_batch=system["src_batch"],
        tgt_positions=system["tgt_positions"],
        tgt_batch=system["tgt_batch"],
        cell=cell,
    )


def _pair(density_max_l, feature_max_l, pbc_handling, **pme_kwargs):
    base = _base(density_max_l, feature_max_l, pbc_handling)
    direct = GTOElectrostaticExternalSourceFeatures.from_features(base)
    pme = GTOElectrostaticExternalSourceFeatures.from_features(
        base, kspace_method="pme", **pme_kwargs
    )
    return direct, pme


def _assert_close_scaled(actual, expected, rel):
    scale = expected.abs().max()
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=rel * float(scale))


@pytest.mark.parametrize("pbc_handling", list(PBC_FLAGS))
@pytest.mark.parametrize("density_max_l,feature_max_l", [(0, 0), (1, 1), (0, 1), (1, 0)])
def test_pme_matches_direct(pbc_handling, density_max_l, feature_max_l):
    cells = _cells()[:1]
    direct, pme = _pair(density_max_l, feature_max_l, pbc_handling)
    geometry = _geometry(cells, pbc_handling)
    system = _system(cells, [60], [5], (density_max_l + 1) ** 2)

    expected = _features(direct, geometry, system)
    actual = _features(pme, geometry, system, cell=cells)
    _assert_close_scaled(actual, expected, rel=5e-5)


def test_pme_batched_cells_and_graph_without_sources():
    cells = torch.cat([_cells(), _cells()[:1] * 1.1])
    direct, pme = _pair(1, 1, "mixed_periodic")
    geometry = _geometry(cells, "pbc")
    system = _system(cells, [50, 0, 40], [4, 3, 6], 4, seed=3)

    expected = _features(direct, geometry, system)
    actual = _features(pme, geometry, system, cell=cells)
    _assert_close_scaled(actual, expected, rel=5e-5)
    graph_1 = system["tgt_batch"] == 1
    torch.testing.assert_close(actual[graph_1], expected[graph_1])


def test_pme_gradients_match_direct():
    cells = _cells()[:1]
    direct, pme = _pair(1, 1, "pbc")
    geometry = _geometry(cells, "pbc")
    system = _system(cells, [40], [4], 4, seed=5)

    grads = []
    for block, cell in ((direct, None), (pme, cells)):
        inputs = {
            key: value.clone().requires_grad_(value.is_floating_point())
            for key, value in system.items()
        }
        features = _features(block, geometry, inputs, cell=cell)
        (features ** 2).sum().backward()
        grads.append(
            [inputs[key].grad for key in ("src_positions", "source_feats", "tgt_positions")]
        )
    for actual, expected in zip(grads[1], grads[0]):
        _assert_close_scaled(actual, expected, rel=5e-4)


def test_pme_error_decreases_with_mesh_factor():
    cells = _cells()[:1]
    geometry = _geometry(cells, "pbc")
    system = _system(cells, [60], [5], 4, seed=7)
    direct, coarse = _pair(1, 1, "pbc", pme_mesh_factor=1.0)
    _, fine = _pair(1, 1, "pbc", pme_mesh_factor=3.0)

    expected = _features(direct, geometry, system)
    coarse_error = (_features(coarse, geometry, system, cell=cells) - expected).abs().max()
    fine_error = (_features(fine, geometry, system, cell=cells) - expected).abs().max()
    assert fine_error < 0.1 * coarse_error


def test_pme_wrapper_uses_external_cell():
    cells = _cells()[:1]
    pbc_geometry = _geometry(cells, "pbc")
    base = _base(1, 1, "pbc")
    system = _system(cells, [30], [4], 4, seed=11)
    internal_feats = torch.randn(4, 4)
    geometry = {
        **pbc_geometry,
        "node_positions": system["tgt_positions"],
        "batch": system["tgt_batch"],
    }
    external = (
        system["source_feats"], system["src_positions"], system["src_batch"]
    )

    direct = GTOElectrostaticExternalSourceFeatures.from_features(base, external_scale=0.5)
    direct.set_external_sources(*external)
    pme = GTOElectrostaticExternalSourceFeatures.from_features(
        base, external_scale=0.5, kspace_method="pme"
    )
    pme.set_external_sources(*external, cell=cells)

    expected = direct(source_feats=internal_feats, **geometry)
    actual = pme(source_feats=internal_feats, **geometry)
    _assert_close_scaled(actual, expected, rel=5e-5)

    pme.clear_external_sources()
    assert pme._external_cell is None


def test_pme_validates_inputs():
    with pytest.raises(ValueError, match="kspace_method"):
        GTOElectrostaticExternalSourceFeatures.from_features(
            _base(0, 0, "pbc"), kspace_method="ewald"
        )

    cells = _cells()[:1]
    geometry = _geometry(cells, "pbc")
    system = _system(cells, [10], [2], 1)
    _, pme = _pair(0, 0, "pbc")
    with pytest.raises(ValueError, match="needs the cell"):
        _features(pme, geometry, system)
    with pytest.raises(ValueError, match="integer combinations"):
        _features(pme, geometry, system, cell=cells * 1.37)
