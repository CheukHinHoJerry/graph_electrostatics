"""Equivalence tests for the source-target descriptor API.

The new `forward_source_target` path on `GTOElectrostaticFeatures` (and the
underlying `RealSpaceFiniteDifferenceElectrostaticFeatures`,
`NonPeriodicFeatureCorrections`, `CorrectivePotentialBlock`,
`slab_dipole_correction_node_fields_source_target`) must produce the same
field features at target nodes as the legacy concat-and-slice approach when
the QM (target-only) source coefficients are zero — that is, the case
exercised by `_compute_mm_field_features` in `mace/mace/modules/extensions.py`.

Adding zero rows to a batched matmul / scatter is exact in IEEE float, so
equivalence is expected at float64 round-off (allclose @ 1e-12).
"""
from __future__ import annotations

# PyTorch 2.6 changed torch.load default to weights_only=True; e3nn's
# constants.pt uses `slice` which is not in the default allowlist. Patch.
import torch
torch.serialization.add_safe_globals([slice])

import pytest

from graph_longrange.features import GTOElectrostaticFeatures
from graph_longrange.kspace import compute_k_vectors_flat


@pytest.fixture(autouse=True)
def _restore_default_dtype():
    default_dtype = torch.get_default_dtype()
    yield
    torch.set_default_dtype(default_dtype)


# ---------------------------------------------------------------------------
# Reference impl: legacy concat-and-slice path (mirrors _compute_mm_field_features)
# ---------------------------------------------------------------------------

def _legacy_concat_and_slice(
    descriptor: GTOElectrostaticFeatures,
    k_vectors,
    k_norm2,
    k_vector_batch,
    k0_mask,
    src_positions,
    src_batch,
    src_feats,                          # [N_src, m_dim]
    tgt_positions,
    tgt_batch,
    volume,
    pbc,
    force_pbc_evaluator: bool = False,
):
    n_src = src_positions.shape[0]
    m_dim = src_feats.shape[-1]
    tgt_zeros = torch.zeros(
        tgt_positions.shape[0], m_dim, dtype=src_feats.dtype, device=src_feats.device
    )
    all_positions = torch.cat([src_positions, tgt_positions], dim=0)
    all_batch = torch.cat([src_batch, tgt_batch], dim=0)
    all_feats = torch.cat([src_feats, tgt_zeros], dim=0)

    cache = descriptor.precompute_geometry(
        k_vectors=k_vectors,
        k_norm2=k_norm2,
        k_vector_batch=k_vector_batch,
        k0_mask=k0_mask,
        node_positions=all_positions,
        batch=all_batch,
        volume=volume,
        pbc=pbc,
        force_pbc_evaluator=force_pbc_evaluator,
    )
    feats_full = descriptor.forward_dynamic(
        cache=cache, source_feats=all_feats.unsqueeze(-2), pbc=pbc
    )
    return feats_full[n_src:]


def _new_source_target(
    descriptor: GTOElectrostaticFeatures,
    k_vectors,
    k_norm2,
    k_vector_batch,
    k0_mask,
    src_positions,
    src_batch,
    src_feats,
    tgt_positions,
    tgt_batch,
    volume,
    pbc,
    force_pbc_evaluator: bool = False,
):
    cache = descriptor.precompute_geometry_source_target(
        k_vectors=k_vectors,
        k_norm2=k_norm2,
        k_vector_batch=k_vector_batch,
        k0_mask=k0_mask,
        src_positions=src_positions,
        src_batch=src_batch,
        tgt_positions=tgt_positions,
        tgt_batch=tgt_batch,
        volume=volume,
        pbc=pbc,
        force_pbc_evaluator=force_pbc_evaluator,
    )
    return descriptor.forward_source_target(
        cache=cache, source_feats=src_feats.unsqueeze(-2), pbc=pbc
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _set_dtype():
    torch.set_default_dtype(torch.float64)


def _make_pbc_geometry(box: float = 8.0):
    cell = torch.eye(3).unsqueeze(0) * box
    rcell = torch.inverse(cell)
    volume = torch.det(cell)
    pbc = torch.tensor([[True, True, True]], dtype=torch.bool)
    return cell, rcell, volume, pbc


def _make_nonpbc_geometry(box: float = 30.0):
    # Use a large pseudo-cell for the molecule-correction code path.
    cell = torch.eye(3).unsqueeze(0) * box
    rcell = torch.inverse(cell)
    volume = torch.det(cell)
    pbc = torch.tensor([[False, False, False]], dtype=torch.bool)
    return cell, rcell, volume, pbc


def _build_descriptor(density_max_l, feature_max_l, feature_widths, kspace_cutoff):
    return GTOElectrostaticFeatures(
        density_max_l=density_max_l,
        density_smearing_width=1.0,
        feature_max_l=feature_max_l,
        feature_smearing_widths=feature_widths,
        kspace_cutoff=kspace_cutoff,
        include_self_interaction=False,
    )


# ---------------------------------------------------------------------------
# PBC equivalence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("density_max_l,feature_max_l", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_pbc_equivalence_single_graph(density_max_l, feature_max_l):
    _set_dtype()
    torch.manual_seed(0)

    kspace_cutoff = 4.0
    descriptor = _build_descriptor(
        density_max_l=density_max_l,
        feature_max_l=feature_max_l,
        feature_widths=[0.8, 1.6],
        kspace_cutoff=kspace_cutoff,
    )

    n_src, n_tgt = 6, 4
    src_positions = torch.randn(n_src, 3) * 2.0
    tgt_positions = torch.randn(n_tgt, 3) * 2.0 + 0.5
    src_batch = torch.zeros(n_src, dtype=torch.long)
    tgt_batch = torch.zeros(n_tgt, dtype=torch.long)
    m_dim = (density_max_l + 1) ** 2
    src_feats = torch.randn(n_src, m_dim)

    cell, rcell, volume, pbc = _make_pbc_geometry()
    k_vectors, k_norm2, k_vector_batch, k0_mask = compute_k_vectors_flat(
        kspace_cutoff, cell, rcell
    )

    legacy = _legacy_concat_and_slice(
        descriptor, k_vectors, k_norm2, k_vector_batch, k0_mask,
        src_positions, src_batch, src_feats, tgt_positions, tgt_batch, volume, pbc,
    )
    new = _new_source_target(
        descriptor, k_vectors, k_norm2, k_vector_batch, k0_mask,
        src_positions, src_batch, src_feats, tgt_positions, tgt_batch, volume, pbc,
    )
    assert legacy.shape == new.shape
    torch.testing.assert_close(new, legacy, rtol=1e-10, atol=1e-12)


def test_pbc_equivalence_two_graphs():
    _set_dtype()
    torch.manual_seed(1)

    kspace_cutoff = 4.0
    descriptor = _build_descriptor(
        density_max_l=1, feature_max_l=1,
        feature_widths=[1.0], kspace_cutoff=kspace_cutoff,
    )

    src_positions = torch.randn(8, 3) * 2.0
    tgt_positions = torch.randn(5, 3) * 2.0 + 0.5
    # 4 sources in graph 0, 4 in graph 1; 3 targets in graph 0, 2 in graph 1.
    src_batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    tgt_batch = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
    src_feats = torch.randn(8, 4)

    box = 8.0
    cells = torch.eye(3).unsqueeze(0).expand(2, 3, 3).contiguous() * box
    rcells = torch.inverse(cells)
    volume = torch.det(cells)
    pbc = torch.tensor([[True, True, True], [True, True, True]], dtype=torch.bool)

    k_vectors, k_norm2, k_vector_batch, k0_mask = compute_k_vectors_flat(
        kspace_cutoff, cells, rcells
    )

    legacy = _legacy_concat_and_slice(
        descriptor, k_vectors, k_norm2, k_vector_batch, k0_mask,
        src_positions, src_batch, src_feats, tgt_positions, tgt_batch, volume, pbc,
    )
    new = _new_source_target(
        descriptor, k_vectors, k_norm2, k_vector_batch, k0_mask,
        src_positions, src_batch, src_feats, tgt_positions, tgt_batch, volume, pbc,
    )
    torch.testing.assert_close(new, legacy, rtol=1e-10, atol=1e-12)


def test_pbc_source_target_respects_independent_batches():
    """A source from graph 1 must not contribute to graph 0 targets."""
    _set_dtype()

    kspace_cutoff = 4.0
    descriptor = _build_descriptor(
        density_max_l=0,
        feature_max_l=1,
        feature_widths=[1.0],
        kspace_cutoff=kspace_cutoff,
    )

    src_positions = torch.tensor(
        [
            [0.1, 0.0, 0.0],
            [0.2, 0.0, 0.0],
        ]
    )
    tgt_positions = torch.tensor([[0.3, 0.0, 0.0]])
    src_batch = torch.tensor([0, 1], dtype=torch.long)
    tgt_batch = torch.tensor([0], dtype=torch.long)
    src_feats = torch.tensor([[1.0], [1000.0]])

    box = 8.0
    cells = torch.eye(3).unsqueeze(0).expand(2, 3, 3).contiguous() * box
    rcells = torch.inverse(cells)
    volume = torch.det(cells)
    pbc = torch.tensor([[True, True, True], [True, True, True]], dtype=torch.bool)
    k_vectors, k_norm2, k_vector_batch, k0_mask = compute_k_vectors_flat(
        kspace_cutoff, cells, rcells
    )

    both_sources = _new_source_target(
        descriptor,
        k_vectors,
        k_norm2,
        k_vector_batch,
        k0_mask,
        src_positions,
        src_batch,
        src_feats,
        tgt_positions,
        tgt_batch,
        volume,
        pbc,
    )
    graph0_only = _new_source_target(
        descriptor,
        k_vectors,
        k_norm2,
        k_vector_batch,
        k0_mask,
        src_positions[:1],
        src_batch[:1],
        src_feats[:1],
        tgt_positions,
        tgt_batch,
        volume,
        pbc,
    )
    torch.testing.assert_close(both_sources, graph0_only, rtol=1e-10, atol=1e-12)


# ---------------------------------------------------------------------------
# Non-PBC equivalence (molecule correction path)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("density_max_l,feature_max_l", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_nonpbc_equivalence_single_graph(density_max_l, feature_max_l):
    _set_dtype()
    torch.manual_seed(2)

    kspace_cutoff = 4.0
    descriptor = _build_descriptor(
        density_max_l=density_max_l,
        feature_max_l=feature_max_l,
        feature_widths=[0.8, 1.6],
        kspace_cutoff=kspace_cutoff,
    )

    n_src, n_tgt = 5, 3
    src_positions = torch.randn(n_src, 3)
    tgt_positions = torch.randn(n_tgt, 3) + 0.3
    src_batch = torch.zeros(n_src, dtype=torch.long)
    tgt_batch = torch.zeros(n_tgt, dtype=torch.long)
    m_dim = (density_max_l + 1) ** 2
    src_feats = torch.randn(n_src, m_dim)

    cell, rcell, volume, pbc = _make_nonpbc_geometry()
    # k_vectors / k_norm2 for non-PBC: the realspace path doesn't use them, but
    # we still need the tensors to have the right device/dtype shape.
    k_vectors, k_norm2, k_vector_batch, k0_mask = compute_k_vectors_flat(
        kspace_cutoff, cell, rcell
    )

    legacy = _legacy_concat_and_slice(
        descriptor, k_vectors, k_norm2, k_vector_batch, k0_mask,
        src_positions, src_batch, src_feats, tgt_positions, tgt_batch, volume, pbc,
    )
    new = _new_source_target(
        descriptor, k_vectors, k_norm2, k_vector_batch, k0_mask,
        src_positions, src_batch, src_feats, tgt_positions, tgt_batch, volume, pbc,
    )
    torch.testing.assert_close(new, legacy, rtol=1e-10, atol=1e-12)


def test_nonpbc_force_pbc_evaluator_equivalence():
    """Molecule + force_pbc_evaluator hits the molecule-correction branch in PBC mode."""
    _set_dtype()
    torch.manual_seed(3)

    kspace_cutoff = 4.0
    descriptor = _build_descriptor(
        density_max_l=1, feature_max_l=1,
        feature_widths=[1.0], kspace_cutoff=kspace_cutoff,
    )

    n_src, n_tgt = 5, 3
    src_positions = torch.randn(n_src, 3)
    tgt_positions = torch.randn(n_tgt, 3) + 0.5
    src_batch = torch.zeros(n_src, dtype=torch.long)
    tgt_batch = torch.zeros(n_tgt, dtype=torch.long)
    src_feats = torch.randn(n_src, 4)

    cell, rcell, volume, pbc = _make_nonpbc_geometry(box=20.0)
    k_vectors, k_norm2, k_vector_batch, k0_mask = compute_k_vectors_flat(
        kspace_cutoff, cell, rcell
    )

    legacy = _legacy_concat_and_slice(
        descriptor, k_vectors, k_norm2, k_vector_batch, k0_mask,
        src_positions, src_batch, src_feats, tgt_positions, tgt_batch, volume, pbc,
        force_pbc_evaluator=True,
    )
    new = _new_source_target(
        descriptor, k_vectors, k_norm2, k_vector_batch, k0_mask,
        src_positions, src_batch, src_feats, tgt_positions, tgt_batch, volume, pbc,
        force_pbc_evaluator=True,
    )
    torch.testing.assert_close(new, legacy, rtol=1e-10, atol=1e-12)


# ---------------------------------------------------------------------------
# Gradient flow (autograd parity)
# ---------------------------------------------------------------------------

def test_pbc_gradients_match_concat_path():
    _set_dtype()
    torch.manual_seed(4)

    kspace_cutoff = 4.0
    descriptor = _build_descriptor(
        density_max_l=0, feature_max_l=1,
        feature_widths=[1.0], kspace_cutoff=kspace_cutoff,
    )

    n_src, n_tgt = 4, 3
    src_positions_a = torch.randn(n_src, 3, requires_grad=True)
    tgt_positions_a = torch.randn(n_tgt, 3, requires_grad=True)
    src_batch = torch.zeros(n_src, dtype=torch.long)
    tgt_batch = torch.zeros(n_tgt, dtype=torch.long)
    src_feats = torch.randn(n_src, 1)

    cell, rcell, volume, pbc = _make_pbc_geometry()
    k_vectors, k_norm2, k_vector_batch, k0_mask = compute_k_vectors_flat(
        kspace_cutoff, cell, rcell
    )

    legacy = _legacy_concat_and_slice(
        descriptor, k_vectors, k_norm2, k_vector_batch, k0_mask,
        src_positions_a, src_batch, src_feats, tgt_positions_a, tgt_batch, volume, pbc,
    )
    legacy.sum().backward()
    grad_src_legacy = src_positions_a.grad.clone()
    grad_tgt_legacy = tgt_positions_a.grad.clone()

    src_positions_b = src_positions_a.detach().clone().requires_grad_(True)
    tgt_positions_b = tgt_positions_a.detach().clone().requires_grad_(True)
    new = _new_source_target(
        descriptor, k_vectors, k_norm2, k_vector_batch, k0_mask,
        src_positions_b, src_batch, src_feats, tgt_positions_b, tgt_batch, volume, pbc,
    )
    new.sum().backward()
    torch.testing.assert_close(src_positions_b.grad, grad_src_legacy, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(tgt_positions_b.grad, grad_tgt_legacy, rtol=1e-10, atol=1e-12)


# ---------------------------------------------------------------------------
# Graphs holding targets but no sources.
#
# Splitting the source and target sets breaks an invariant the symmetric path
# could rely on: there, every graph with a target necessarily had a source, so
# sizing a per-graph reduction by the highest index present in `batch` was
# always right. Here a graph can hold targets and no sources at all, and the
# reductions must still produce a (zero) row for it.
# ---------------------------------------------------------------------------


def test_slab_correction_no_leak_into_sourceless_graph():
    """A target in a graph with no sources must feel no slab correction.

    Sizing the dipole by `src_batch` alone yields one row for a two-graph batch,
    which then broadcasts graph 0's dipole onto graph 1 — silently, with no error.
    """
    from graph_longrange.slabs import slab_dipole_correction_node_fields_source_target

    node_fields = slab_dipole_correction_node_fields_source_target(
        source_feats=torch.tensor([[3.0]]),
        src_positions=torch.tensor([[0.0, 0.0, 2.0]]),
        src_batch=torch.tensor([0]),          # every source in graph 0
        tgt_positions=torch.tensor([[0.0, 0.0, 4.0]]),
        tgt_batch=torch.tensor([1]),          # the target is in graph 1
        volumes=torch.tensor([10.0, 20.0]),
    )
    torch.testing.assert_close(node_fields, torch.zeros_like(node_fields))


def test_corrective_potential_handles_sourceless_trailing_graph():
    """The highest-indexed graph having no sources must not raise."""
    from graph_longrange.slabs import CorrectivePotentialBlock

    block = CorrectivePotentialBlock(density_max_l=0)
    node_fields = block.forward_source_target(
        charge_coefficients=torch.tensor([[3.0]]),
        src_positions=torch.tensor([[0.0, 0.0, 2.0]]),
        src_batch=torch.tensor([0]),
        tgt_positions=torch.tensor([[0.0, 0.0, 4.0], [0.0, 0.0, 5.0]]),
        tgt_batch=torch.tensor([0, 1]),       # graph 1 has targets, no sources
        volumes=torch.tensor([10.0, 20.0]),
    )
    assert node_fields.shape == (2, 4)
    assert torch.isfinite(node_fields).all()


def test_corrective_potential_with_no_sources_at_all():
    """Zero sources is a legitimate input and must give a zero correction."""
    from graph_longrange.slabs import CorrectivePotentialBlock

    block = CorrectivePotentialBlock(density_max_l=0)
    node_fields = block.forward_source_target(
        charge_coefficients=torch.zeros((0, 1)),
        src_positions=torch.zeros((0, 3)),
        src_batch=torch.zeros(0, dtype=torch.long),
        tgt_positions=torch.tensor([[0.0, 0.0, 4.0], [0.0, 0.0, 5.0]]),
        tgt_batch=torch.tensor([0, 1]),
        volumes=torch.tensor([10.0, 20.0]),
    )
    torch.testing.assert_close(node_fields, torch.zeros_like(node_fields))


def test_kspace_zero_targets_returns_empty_features():
    """No targets is a legitimate input and must give an empty feature tensor.

    The k-space path flattens with reshape, and inferring the trailing dimension
    with -1 cannot work when the tensor has no elements to infer from.
    """
    _set_dtype()
    torch.manual_seed(0)

    kspace_cutoff = 4.0
    descriptor = _build_descriptor(
        density_max_l=0,
        feature_max_l=0,
        feature_widths=[0.8, 1.6],
        kspace_cutoff=kspace_cutoff,
    )

    n_src = 4
    src_positions = torch.randn(n_src, 3) * 2.0
    src_batch = torch.zeros(n_src, dtype=torch.long)
    src_feats = torch.randn(n_src, 1)
    tgt_positions = torch.zeros((0, 3))
    tgt_batch = torch.zeros(0, dtype=torch.long)

    cell, rcell, volume, pbc = _make_pbc_geometry()
    k_vectors, k_norm2, k_vector_batch, k0_mask = compute_k_vectors_flat(
        kspace_cutoff, cell, rcell
    )

    features = _new_source_target(
        descriptor, k_vectors, k_norm2, k_vector_batch, k0_mask,
        src_positions, src_batch, src_feats, tgt_positions, tgt_batch, volume, pbc,
    )
    assert features.shape[0] == 0
