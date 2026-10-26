"""Particle-mesh structure factors for large external source sets.

The direct source-target k-space path builds dense ``[n_k, n_src]`` cosine and
sine tables, which dominates memory once the external (e.g. MM) environment
holds tens of thousands of sources.  Only the per-channel structure factor

    C_m(k) = sum_s f_{s,m} exp(-i k . r_s)

is ever needed from those tables, so here it is evaluated by smooth
particle-mesh interpolation instead: the source channels are spread onto a mesh
with torch-pme's P3M (cardinal B-spline) weights, Fourier transformed, and
deconvolved by the transform of a unit source spread at the origin (Euler
exponential splines, Essmann et al., J. Chem. Phys. 103, 8577 (1995)).  The
result is read off at each existing k-vector's integer index, so the k-set,
bases and normalisation of the direct path are reused unchanged.

Memory is O(m_dim * n_mesh + n_src * p^3) rather than O(n_k * n_src).
"""

from __future__ import annotations

import math
from typing import List

import torch
from scipy.constants import pi

try:
    from torchpme.lib import MeshInterpolator
except ImportError:  # optional dependency, only needed for method="PME"
    MeshInterpolator = None

# ~1e-6 relative error against the direct sum (see docs/external_sources.md)
PME_DEFAULT_INTERPOLATION_NODES = 5
PME_DEFAULT_MESH_FACTOR = 3.0


def _require_torchpme() -> None:
    if MeshInterpolator is None:
        raise ImportError(
            "method='PME' requires torch-pme (pip install torch-pme)."
        )


def _broadcast_cell(cell: torch.Tensor, n_graphs: int) -> torch.Tensor:
    cell = cell.reshape(-1, 3, 3)
    if cell.shape[0] == 1 and n_graphs > 1:
        cell = cell.expand(n_graphs, 3, 3)
    if cell.shape[0] != n_graphs:
        raise ValueError(
            f"cell holds {cell.shape[0]} cells but the batch has {n_graphs} graphs."
        )
    return cell


def _mesh_shape(m_max: torch.Tensor, mesh_factor: float, interpolation_nodes: int):
    # N_d > 2 * m_max_d keeps every k on a distinct mesh frequency below Nyquist;
    # mesh_factor oversamples on top of that to control the aliasing error.
    return [
        max(int(math.ceil(mesh_factor * (2 * m + 1))), 2 * m + 2, interpolation_nodes)
        for m in m_max.tolist()
    ]


def precompute_pme_source_geometry(
    k_vectors: torch.Tensor,
    k_vector_batch: torch.Tensor,
    src_positions: torch.Tensor,
    src_batch: torch.Tensor,
    cell: torch.Tensor,
    n_graphs: int,
    interpolation_nodes: int,
    mesh_factor: float,
) -> List[dict]:
    """Per-graph spreading weights, mesh indices and deconvolution factors.

    Depends only on geometry, so it may be cached across changing source
    features.  Graphs without k-vectors or without sources are skipped; their
    structure factor is zero.

    Args:
        k_vectors: [n_k_total, 3], integer combinations of ``2 pi inv(cell)^T``.
        cell: [n_graphs, 3, 3] or [3, 3], rows are lattice vectors.
    """
    _require_torchpme()
    cell = _broadcast_cell(cell, n_graphs)
    graphs = []
    for g in range(n_graphs):
        k_rows = (k_vector_batch == g).nonzero(as_tuple=True)[0]
        src_rows = (src_batch == g).nonzero(as_tuple=True)[0]
        if k_rows.numel() == 0 or src_rows.numel() == 0:
            continue
        cell_g = cell[g].to(dtype=src_positions.dtype)
        with torch.no_grad():
            # A cell off by a relative delta puts an index vector off by
            # ~delta * max|n|, spread over all its components in a triclinic
            # cell, so normalise each vector's residual by its largest index.
            # Single-precision k-vectors already sit ~4e-4 off under TF32.
            m_float = (
                k_vectors[k_rows].detach().double()
                @ cell_g.detach().double().t()
                / (2 * pi)
            )
            m_int = torch.round(m_float)
            relative_residual = (m_float - m_int).abs().amax(dim=-1) / (
                m_int.abs().amax(dim=-1).clamp_min(1.0)
            )
            tolerance = 1e-6 if k_vectors.dtype == torch.float64 else 2e-3
            if relative_residual.max() > tolerance:
                raise ValueError(
                    "k_vectors are not integer combinations of 2*pi*inv(cell)^T: "
                    "pass the cell the k-vectors were generated from."
                )
            m_int = m_int.long()
            ns = _mesh_shape(m_int.abs().amax(dim=0), mesh_factor, interpolation_nodes)
            ns_mesh = torch.tensor(ns, device=k_vectors.device)
            mesh_index = m_int % ns_mesh

            # A unit source at the origin spreads to the interpolation stencil
            # alone; dividing by its transform makes the spread exact on mesh
            # points and leaves only the aliasing error.
            reference = MeshInterpolator(
                cell_g.detach(), ns_mesh, interpolation_nodes, method="P3M"
            )
            reference.compute_weights(
                torch.zeros((1, 3), dtype=cell_g.dtype, device=cell_g.device)
            )
            stencil = reference.points_to_mesh(
                torch.ones((1, 1), dtype=cell_g.dtype, device=cell_g.device)
            )[0]
            stencil_fs = torch.fft.fftn(stencil)
            deconvolution = 1.0 / stencil_fs[
                mesh_index[:, 0], mesh_index[:, 1], mesh_index[:, 2]
            ]

        interpolator = MeshInterpolator(
            cell_g, ns_mesh, interpolation_nodes, method="P3M"
        )
        interpolator.compute_weights(src_positions[src_rows])
        graphs.append(
            {
                "k_rows": k_rows,
                "src_rows": src_rows,
                "mesh_index": mesh_index,
                "deconvolution": deconvolution,
                "interpolator": interpolator,
            }
        )
    return graphs


def assemble_density_from_structure_factor(
    coeff_cos: torch.Tensor,
    coeff_sin: torch.Tensor,
    density_basis_fs: torch.Tensor,
    volume_per_k: torch.Tensor,
) -> torch.Tensor:
    """rho(k) from per-channel structure factors.

    The same assembly as ``features.assemble_fourier_series_batch`` after its
    ``cosines @ source_feats`` / ``sines @ source_feats`` products, which PME
    replaces.

    Args:
        coeff_cos: [n_k_total, m_dim], sum_s cos(k.r_s) f_s
        coeff_sin: [n_k_total, m_dim], sum_s sin(k.r_s) f_s
        density_basis_fs: [n_k_total, n_sigma(=1), m_dim, 2]
        volume_per_k: [n_k_total]
    Returns:
        density: [n_k_total, 2]
    """
    sm_dim = density_basis_fs.size(1) * density_basis_fs.size(2)
    density_basis_r = density_basis_fs[..., 0].reshape(density_basis_fs.size(0), sm_dim)
    density_basis_i = density_basis_fs[..., 1].reshape(density_basis_fs.size(0), sm_dim)

    rho_real = (density_basis_r * coeff_cos).sum(dim=-1) + (
        density_basis_i * coeff_sin
    ).sum(dim=-1)
    rho_imag = (density_basis_i * coeff_cos).sum(dim=-1) - (
        density_basis_r * coeff_sin
    ).sum(dim=-1)

    density = torch.stack([rho_real, rho_imag], dim=-1)
    return (2 * pi) ** 3 * density / volume_per_k.unsqueeze(-1)


def pme_structure_factor(
    source_feats: torch.Tensor,
    pme_geometry: List[dict],
    n_k_total: int,
):
    """Return ``(coeff_cos, coeff_sin)``, each [n_k_total, m_dim].

    These approximate ``cos(k.r_s) @ source_feats`` and
    ``sin(k.r_s) @ source_feats`` of the direct path, restricted to each
    source's own graph.
    """
    m_dim = source_feats.size(-1)
    # anchored to source_feats so an empty source set keeps an autograd path,
    # as the direct matmul does
    zero = 0.0 * source_feats.sum()
    coeff_cos = source_feats.new_zeros((n_k_total, m_dim)) + zero
    coeff_sin = source_feats.new_zeros((n_k_total, m_dim)) + zero
    for graph in pme_geometry:
        mesh = graph["interpolator"].points_to_mesh(source_feats[graph["src_rows"]])
        mesh_fs = torch.fft.fftn(mesh, dim=(-3, -2, -1))
        index = graph["mesh_index"]
        # [m_dim, n_k_graph]: sum_s f_s exp(-i k . r_s)
        structure = (
            mesh_fs[:, index[:, 0], index[:, 1], index[:, 2]] * graph["deconvolution"]
        )
        coeff_cos = coeff_cos.index_copy(0, graph["k_rows"], structure.real.t())
        coeff_sin = coeff_sin.index_copy(0, graph["k_rows"], -structure.imag.t())
    return coeff_cos, coeff_sin


__all__ = [
    "PME_DEFAULT_INTERPOLATION_NODES",
    "PME_DEFAULT_MESH_FACTOR",
    "assemble_density_from_structure_factor",
    "precompute_pme_source_geometry",
    "pme_structure_factor",
]
