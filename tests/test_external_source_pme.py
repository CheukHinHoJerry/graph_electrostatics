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


def test_pme_float32_matches_direct():
    torch.set_default_dtype(torch.float32)
    cells = _cells()[:1] * 4.0  # larger |n| makes single-precision k rounding visible
    direct, pme = _pair(1, 1, "pbc")
    geometry = _geometry(cells, "pbc")
    system = _system(cells, [200], [5], 4, seed=13)

    expected = _features(direct, geometry, system)
    actual = _features(pme, geometry, system, cell=cells)
    assert actual.dtype == torch.float32
    _assert_close_scaled(actual, expected, rel=1e-3)


def test_pme_split_api_reuses_geometry_for_new_source_feats():
    cells = _cells()[:1]
    direct, pme = _pair(1, 1, "pbc")
    geometry = _geometry(cells, "pbc")
    system = _system(cells, [50], [4], 4, seed=17)
    positions = {
        key: system[key]
        for key in ("src_positions", "src_batch", "tgt_positions", "tgt_batch")
    }
    direct_cache = direct.precompute_geometry_source_target(**geometry, **positions)
    pme_cache = pme.precompute_geometry_source_target(**geometry, **positions, cell=cells)
    assert "cosines_src" not in pme_cache and "sines_src" not in pme_cache

    for seed in (0, 1):
        feats = torch.randn(50, 4, generator=torch.Generator().manual_seed(seed))
        expected = direct.forward_dynamic_source_target(direct_cache, feats)
        actual = pme.forward_dynamic_source_target(pme_cache, feats.unsqueeze(-2))
        _assert_close_scaled(actual, expected, rel=5e-5)


def test_pme_cell_gradient_matches_direct():
    system = _system(_cells()[:1], [40], [4], 4, seed=19)
    frac = {
        key: system[key] @ torch.linalg.inv(_cells()[0])
        for key in ("src_positions", "tgt_positions")
    }
    grads = []
    for method in ("direct", "pme"):
        cells = _cells()[:1].clone().requires_grad_()
        rcells = 2 * pi * torch.linalg.inv(cells).transpose(-1, -2)
        k_vectors, k_norm2, k_vector_batch, k0_mask = compute_k_vectors_flat(
            2.5, cells, rcells
        )
        block = GTOElectrostaticExternalSourceFeatures.from_features(
            _base(1, 1, "pbc"), kspace_method=method
        )
        features = block.forward_source_target(
            k_vectors=k_vectors,
            k_norm2=k_norm2,
            k_vector_batch=k_vector_batch,
            k0_mask=k0_mask,
            source_feats=system["source_feats"],
            src_positions=frac["src_positions"] @ cells[0],
            src_batch=system["src_batch"],
            tgt_positions=frac["tgt_positions"] @ cells[0],
            tgt_batch=system["tgt_batch"],
            volume=torch.det(cells),
            pbc=torch.tensor([[True, True, True]]),
            cell=cells if method == "pme" else None,
        )
        (features ** 2).sum().backward()
        grads.append(cells.grad)
    _assert_close_scaled(grads[1], grads[0], rel=5e-4)


def test_pme_is_periodic_in_source_images():
    cells = _cells()[:1]
    _, pme = _pair(1, 1, "pbc")
    geometry = _geometry(cells, "pbc")
    system = _system(cells, [30], [4], 4, seed=23)
    shifts = torch.randint(-3, 4, (30, 3), generator=torch.Generator().manual_seed(0))
    unwrapped = dict(system, src_positions=system["src_positions"] + shifts.double() @ cells[0])

    expected = _features(pme, geometry, system, cell=cells)
    actual = _features(pme, geometry, unwrapped, cell=cells)
    _assert_close_scaled(actual, expected, rel=1e-10)


def test_pme_is_translation_invariant():
    cells = _cells()[:1]
    direct, pme = _pair(1, 1, "pbc")
    geometry = _geometry(cells, "pbc")
    system = _system(cells, [40], [4], 4, seed=29)
    offset = torch.tensor([0.37, -1.91, 2.53])
    shifted = dict(
        system,
        src_positions=system["src_positions"] + offset,
        tgt_positions=system["tgt_positions"] + offset,
    )

    expected = _features(direct, geometry, system)
    actual = _features(pme, geometry, shifted, cell=cells)
    _assert_close_scaled(actual, expected, rel=5e-5)


@pytest.mark.parametrize("nodes,rel", [(3, 2e-3), (4, 5e-4), (5, 5e-5)])
def test_pme_interpolation_orders(nodes, rel):
    cells = _cells()[:1]
    direct, pme = _pair(1, 1, "pbc", pme_interpolation_nodes=nodes)
    geometry = _geometry(cells, "pbc")
    system = _system(cells, [60], [5], 4, seed=31)

    expected = _features(direct, geometry, system)
    actual = _features(pme, geometry, system, cell=cells)
    _assert_close_scaled(actual, expected, rel=rel)


def test_pme_broadcasts_single_cell_and_rejects_wrong_count():
    cells = torch.cat([_cells()[:1], _cells()[:1]])
    direct, pme = _pair(0, 0, "pbc")
    geometry = _geometry(cells, "pbc")
    system = _system(cells, [20, 25], [3, 2], 1, seed=37)

    expected = _features(direct, geometry, system)
    _assert_close_scaled(_features(pme, geometry, system, cell=cells[0]), expected, rel=5e-5)
    with pytest.raises(ValueError, match="graphs"):
        _features(pme, geometry, system, cell=torch.cat([cells, cells[:1]]))


def test_pme_empty_source_and_target_sets():
    cells = _cells()[:1]
    direct, pme = _pair(1, 1, "pbc")
    geometry = _geometry(cells, "pbc")

    no_sources = _system(cells, [0], [3], 4)
    torch.testing.assert_close(
        _features(pme, geometry, no_sources, cell=cells),
        _features(direct, geometry, no_sources),
    )
    no_targets = _system(cells, [10], [0], 4)
    actual = _features(pme, geometry, no_targets, cell=cells)
    assert actual.shape == _features(direct, geometry, no_targets).shape == (0, 8)


def test_pme_mixed_periodic_batch_with_slab_and_bulk_graphs():
    cells = _cells()
    rcells = 2 * pi * torch.linalg.inv(cells).transpose(-1, -2)
    k_vectors, k_norm2, k_vector_batch, k0_mask = compute_k_vectors_flat(2.5, cells, rcells)
    geometry = {
        "k_vectors": k_vectors,
        "k_norm2": k_norm2,
        "k_vector_batch": k_vector_batch,
        "k0_mask": k0_mask,
        "volume": torch.det(cells).abs(),
        "pbc": torch.tensor([[True, True, False], [True, True, True]]),
    }
    system = _system(cells, [40, 35], [4, 3], 4, seed=41)
    for handling in ("mixed_periodic", "auto"):
        direct, pme = _pair(1, 1, handling)
        expected = _features(direct, geometry, system)
        actual = _features(pme, geometry, system, cell=cells)
        _assert_close_scaled(actual, expected, rel=5e-5)


def test_pme_auto_without_periodicity_needs_no_cell():
    cells = _cells()[:1]
    direct, pme = _pair(1, 1, "auto")
    geometry = dict(_geometry(cells, "pbc"), pbc=torch.tensor([[False, False, False]]))
    system = _system(cells, [20], [3], 4, seed=43)

    torch.testing.assert_close(
        _features(pme, geometry, system), _features(direct, geometry, system)
    )


def test_pme_missing_torchpme_raises_import_error(monkeypatch):
    import graph_longrange.external_source_pme as pme_module

    monkeypatch.setattr(pme_module, "MeshInterpolator", None)
    cells = _cells()[:1]
    _, pme = _pair(0, 0, "pbc")
    system = _system(cells, [5], [2], 1)
    with pytest.raises(ImportError, match="torch-pme"):
        _features(pme, _geometry(cells, "pbc"), system, cell=cells)
