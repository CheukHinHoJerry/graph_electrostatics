"""Internal damped real-space helpers for disjoint source and target sets."""

import math

import torch
from scipy.constants import pi
from mace.tools.scatter import scatter_sum
from .utils import FIELD_CONSTANT

@torch.no_grad()
def batch_bipartite_pairs(
    src_batch: torch.Tensor, tgt_batch: torch.Tensor
) -> torch.Tensor:
    """Build all (sender, receiver) edges with src_batch[s] == tgt_batch[t].

    Sources and targets live in disjoint node sets, so no self-exclusion is
    needed. Edge order is per-graph, then row-major (sender-major) within each
    graph -- matching the order the symmetric primitive produces when restricted
    to source->target pairs. On CPU that makes the scatter sums bit-identical to
    the concatenated path under zero-coefficient padding; on CUDA `scatter_add_`
    is not order-deterministic, so only round-off agreement is guaranteed there.

    Cost: the full dense n_src x n_tgt product per graph, with no distance
    cutoff. Cheaper than the symmetric (n_src + n_tgt)^2 it replaces, but still
    quadratic -- a large aperiodic MM environment will dominate memory.

    Args:
        src_batch: [N_src] graph ID per source node.
        tgt_batch: [N_tgt] graph ID per target node.

    Returns:
        edge_index: [2, E] -- row 0 = source index, row 1 = target index.
    """
    src_batch = src_batch.long()
    tgt_batch = tgt_batch.long()
    if src_batch.numel() == 0 or tgt_batch.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long, device=src_batch.device)

    G = int(max(int(src_batch.max().item()), int(tgt_batch.max().item()))) + 1
    edges = []
    for g in range(G):
        s_nodes = (src_batch == g).nonzero(as_tuple=False).view(-1)
        t_nodes = (tgt_batch == g).nonzero(as_tuple=False).view(-1)
        if s_nodes.numel() == 0 or t_nodes.numel() == 0:
            continue
        Ds = s_nodes.size(0)
        Dt = t_nodes.size(0)
        row = s_nodes.view(-1, 1).expand(-1, Dt).reshape(-1)
        col = t_nodes.view(1, -1).expand(Ds, -1).reshape(-1)
        edges.append(torch.stack([row, col], dim=0))

    if not edges:
        return torch.empty((2, 0), dtype=torch.long, device=src_batch.device)
    return torch.cat(edges, dim=1)


def charges_features_from_bipartite_graph(
    charges,                # [N_src]
    src_positions,          # [N_src, 3]
    tgt_positions,          # [N_tgt, 3]
    edge_index,             # [2, E] -- row 0 in src space, row 1 in tgt space
    n_targets,
    total_width_factors,    # [1, n_radial]
    shifts=None,            # [E, 3] lattice vector added to the sender position
):
    """Bipartite analogue of charges_features_from_graph for MM->QM fields."""
    sender, receiver = edge_index
    R_ij = src_positions[sender] - tgt_positions[receiver]
    if shifts is not None:
        R_ij = R_ij + shifts
    d_ij = torch.norm(R_ij, dim=-1, keepdim=True)
    smooth_reciprocal = torch.erf(0.5 * d_ij / total_width_factors) / (d_ij + 1e-6)
    features = scatter_sum(
        charges[sender].unsqueeze(-1) * smooth_reciprocal,
        receiver,
        dim=0,
        dim_size=n_targets,
    )
    features = FIELD_CONSTANT * features / (4 * pi)
    return features



@torch.no_grad()
def bipartite_pairs_within_cutoff(
    src_positions: torch.Tensor,
    src_batch: torch.Tensor,
    tgt_positions: torch.Tensor,
    tgt_batch: torch.Tensor,
    cutoff: float,
    cell: torch.Tensor = None,
    pbc: torch.Tensor = None,
    pair_budget: int = 1 << 22,
):
    """(sender, receiver) edges with |r_src + shift - r_tgt| <= cutoff, per graph.

    The distance search is chunked over BOTH sets so that no temporary holds
    more than ``pair_budget`` source-target pairs (default 4M pairs, i.e. about
    100 MB of float64 displacements).

    Periodic graphs (``pbc`` row has any True) include every lattice image whose
    shift keeps the pair within the cutoff, so the cutoff may exceed half the
    box. ``cell`` is [n_graphs, 3, 3] (rows are lattice vectors) or [3, 3] and is
    required for periodic graphs. Non-periodic axes get no images.

    Returns ``edge_index`` [2, E] and ``shifts`` [E, 3], the lattice vector that
    is added to the sender position (zeros without periodicity). Edges are
    ordered per graph, then per shift, then sender-major.
    """
    if not (math.isfinite(cutoff) and cutoff > 0.0):
        raise ValueError(f"cutoff must be finite and positive, got {cutoff!r}.")
    src_batch = src_batch.long()
    tgt_batch = tgt_batch.long()
    device = src_positions.device
    empty = (
        torch.empty((2, 0), dtype=torch.long, device=device),
        torch.empty((0, 3), dtype=src_positions.dtype, device=device),
    )
    if src_batch.numel() == 0 or tgt_batch.numel() == 0:
        return empty
    n_graphs = int(max(int(src_batch.max()), int(tgt_batch.max()))) + 1
    if cell is not None:
        cell = cell.to(device=device, dtype=src_positions.dtype).reshape(-1, 3, 3)
        if cell.shape[0] == 1 and n_graphs > 1:
            cell = cell.expand(n_graphs, 3, 3)
    if pbc is not None:
        pbc = pbc.to(device=device).reshape(-1, 3)
        if pbc.shape[0] == 1 and n_graphs > 1:
            pbc = pbc.expand(n_graphs, 3)

    edges, shift_rows = [], []
    for g in range(n_graphs):
        s_nodes = (src_batch == g).nonzero(as_tuple=False).view(-1)
        t_nodes = (tgt_batch == g).nonzero(as_tuple=False).view(-1)
        if s_nodes.numel() == 0 or t_nodes.numel() == 0:
            continue
        periodic = pbc is not None and bool(pbc[g].any())
        if periodic:
            if cell is None:
                raise ValueError(
                    "bipartite_pairs_within_cutoff needs the cell for periodic graphs."
                )
            cell_g = cell[g]
            # perpendicular heights of the cell along each lattice direction
            volume = torch.abs(torch.det(cell_g))
            cross = torch.stack(
                [
                    torch.cross(cell_g[1], cell_g[2], dim=0),
                    torch.cross(cell_g[2], cell_g[0], dim=0),
                    torch.cross(cell_g[0], cell_g[1], dim=0),
                ]
            )
            heights = volume / torch.linalg.norm(cross, dim=1)
            n_max = [
                int(torch.ceil(torch.tensor(cutoff) / heights[a]).item()) if bool(pbc[g][a]) else 0
                for a in range(3)
            ]
            ranges = [torch.arange(-n, n + 1, device=device) for n in n_max]
            grid = torch.cartesian_prod(*ranges).reshape(-1, 3).to(cell_g.dtype)
            shift_vectors = grid @ cell_g
        else:
            shift_vectors = torch.zeros((1, 3), dtype=src_positions.dtype, device=device)

        tgt_chunk = max(1, min(t_nodes.numel(), int(pair_budget)))
        src_chunk = max(1, int(pair_budget) // tgt_chunk)
        for t_start in range(0, t_nodes.numel(), tgt_chunk):
            t_chunk = t_nodes[t_start : t_start + tgt_chunk]
            tgt_g = tgt_positions[t_chunk]
            for start in range(0, s_nodes.numel(), src_chunk):
                s_chunk = s_nodes[start : start + src_chunk]
                disp0 = src_positions[s_chunk][:, None, :] - tgt_g[None, :, :]
                if periodic:
                    # Bring every pair to its nearest image first, so sources that
                    # drifted several cells away are handled; the neighbour shifts
                    # then cover everything within the cutoff around that image.
                    frac = disp0 @ torch.linalg.inv(cell_g)
                    wrap = torch.round(frac) * pbc[g].to(frac.dtype)
                    base_shift = -(wrap @ cell_g)
                else:
                    base_shift = torch.zeros_like(disp0)
                for shift in shift_vectors:
                    disp = disp0 + base_shift + shift
                    within = torch.linalg.norm(disp, dim=-1) <= cutoff
                    if not bool(within.any()):
                        continue
                    si, ti = within.nonzero(as_tuple=True)
                    edges.append(torch.stack([s_chunk[si], t_chunk[ti]], dim=0))
                    shift_rows.append(base_shift[si, ti] + shift)
    if not edges:
        return empty
    return torch.cat(edges, dim=1), torch.cat(shift_rows, dim=0)


def expand_edges_displaced(
    edge_index: torch.Tensor, shifts: torch.Tensor, n_src_rep: int, n_tgt_rep: int
):
    """Map node edges onto the displaced-charge representation.

    Node ``i`` becomes rows ``n_rep * i + k`` (``k < n_rep``); every pair of
    displaced rows of an edge is connected, sender-major, shifts repeated.
    """
    sender, receiver = edge_index
    ks = torch.arange(n_src_rep, device=edge_index.device)
    kt = torch.arange(n_tgt_rep, device=edge_index.device)
    new_sender = (n_src_rep * sender[:, None, None] + ks[None, :, None]).expand(
        -1, n_src_rep, n_tgt_rep
    )
    new_receiver = (n_tgt_rep * receiver[:, None, None] + kt[None, None, :]).expand(
        -1, n_src_rep, n_tgt_rep
    )
    new_shifts = shifts[:, None, None, :].expand(-1, n_src_rep, n_tgt_rep, 3)
    return (
        torch.stack([new_sender.reshape(-1), new_receiver.reshape(-1)], dim=0),
        new_shifts.reshape(-1, 3),
    )


def source_width_factors(
    source_width: float, projection_smearing_widths, like: torch.Tensor
) -> torch.Tensor:
    """``total_width_factors`` for sources of width ``source_width`` instead of the
    density width: ``sqrt((w^2 + sigma_f^2) / 2)`` per feature width."""
    widths = torch.as_tensor(projection_smearing_widths, dtype=like.dtype, device=like.device)
    return torch.sqrt((float(source_width) ** 2 + widths**2) / 2)


class _ExternalSourceRealspace:
    """Reuse a real-space block's basis without rebuilding it."""

    def __init__(self, base):
        self.base = base

    def __getattr__(self, name):
        return getattr(self.base, name)

    def call_source_target_density_0_feats_0(
        self,
        source_feats: torch.Tensor,   # [N_src, m_dim]
        src_positions: torch.Tensor,
        src_batch: torch.Tensor,
        tgt_positions: torch.Tensor,
        tgt_batch: torch.Tensor,
        edge_index: torch.Tensor = None,
        shifts: torch.Tensor = None,
        total_width_factors: torch.Tensor = None,
    ) -> torch.Tensor:
        if edge_index is None:
            edge_index = batch_bipartite_pairs(src_batch, tgt_batch)
        if total_width_factors is None:
            total_width_factors = self.total_width_factors
        feats = charges_features_from_bipartite_graph(
            charges=source_feats[:, 0],
            src_positions=src_positions,
            tgt_positions=tgt_positions,
            edge_index=edge_index,
            n_targets=tgt_positions.shape[0],
            total_width_factors=total_width_factors.unsqueeze(0),
            shifts=shifts,
        )
        return self.l0_factors * feats

    def call_source_target_density_1_feats_1(
        self,
        source_feats: torch.Tensor,   # [N_src, m_dim]
        src_positions: torch.Tensor,
        src_batch: torch.Tensor,
        tgt_positions: torch.Tensor,
        tgt_batch: torch.Tensor,
        edge_index: torch.Tensor = None,
        shifts: torch.Tensor = None,
        total_width_factors: torch.Tensor = None,
    ) -> torch.Tensor:
        if total_width_factors is None:
            total_width_factors = self.total_width_factors
        # Source 4x duplication encodes source dipoles as displaced charges.
        ext_src_positions = src_positions.repeat_interleave(4, dim=0)
        ext_src_positions[1::4] += self.x
        ext_src_positions[2::4] += self.y
        ext_src_positions[3::4] += self.z
        ext_src_batch = src_batch.repeat_interleave(4)
        src_charges = torch.zeros_like(ext_src_positions[:, 0])
        src_charges[1::4] = source_feats[:, 3] / self.offset
        src_charges[2::4] = source_feats[:, 1] / self.offset
        src_charges[3::4] = source_feats[:, 2] / self.offset
        src_charges[0::4] = source_feats[:, 0] - (
            src_charges[1::4] + src_charges[2::4] + src_charges[3::4]
        )

        # Target 4x duplication encodes the receive-side l=1 finite-difference basis.
        ext_tgt_positions = tgt_positions.repeat_interleave(4, dim=0)
        ext_tgt_positions[1::4] += self.x
        ext_tgt_positions[2::4] += self.y
        ext_tgt_positions[3::4] += self.z
        ext_tgt_batch = tgt_batch.repeat_interleave(4)

        if edge_index is None:
            edge_index = batch_bipartite_pairs(ext_src_batch, ext_tgt_batch)
        else:
            if shifts is None:
                shifts = torch.zeros((edge_index.shape[1], 3), dtype=src_positions.dtype, device=src_positions.device)
            edge_index, shifts = expand_edges_displaced(edge_index, shifts, 4, 4)
        scalar_features = charges_features_from_bipartite_graph(
            charges=src_charges,
            src_positions=ext_src_positions,
            tgt_positions=ext_tgt_positions,
            edge_index=edge_index,
            n_targets=ext_tgt_positions.shape[0],
            total_width_factors=total_width_factors.unsqueeze(0),
            shifts=shifts,
        )

        n_tgt = tgt_positions.shape[0]
        all_features = torch.zeros(
            n_tgt,
            4 * self.num_radial,
            dtype=tgt_positions.dtype,
            device=tgt_positions.device,
        )
        all_features[:, : self.num_radial] = self.l0_factors * scalar_features[0::4]
        all_features[:, self.num_radial :: 3] = self.l1_factors * (
            scalar_features[2::4] - scalar_features[0::4]
        )
        all_features[:, self.num_radial + 1 :: 3] = self.l1_factors * (
            scalar_features[3::4] - scalar_features[0::4]
        )
        all_features[:, self.num_radial + 2 :: 3] = self.l1_factors * (
            scalar_features[1::4] - scalar_features[0::4]
        )
        return all_features

    def forward_source_target(
        self,
        source_feats: torch.Tensor,    # [N_src, 1, m_dim] or [N_src, m_dim]
        src_positions: torch.Tensor,
        src_batch: torch.Tensor,
        tgt_positions: torch.Tensor,
        tgt_batch: torch.Tensor,
        edge_index: torch.Tensor = None,
        shifts: torch.Tensor = None,
        source_width: float = None,
    ) -> torch.Tensor:
        """Source-target field features at tgt_positions from sources at src_positions.

        Precondition: source and target node sets are disjoint. The block does
        not subtract a self-interaction term, so colocated source/target rows
        will produce wrong answers. Documented; not enforced (O(N^2) check).

        ``edge_index``/``shifts`` restrict the sum to given (sender, receiver)
        node pairs with lattice shifts (default: every same-graph pair, no
        shifts). ``source_width`` replaces the density smearing width on the
        source side (``None`` keeps the block's own width).
        """
        if source_feats.dim() == 3:
            source_feats_2d = source_feats.squeeze(-2)
        else:
            source_feats_2d = source_feats
        width_factors = None
        if source_width is not None:
            width_factors = source_width_factors(
                source_width, self.projection_smearing_widths, self.total_width_factors
            )
        extra = {"edge_index": edge_index, "shifts": shifts, "total_width_factors": width_factors}

        if self.density_max_l == 0 and self.projection_max_l == 0:
            return self.call_source_target_density_0_feats_0(
                source_feats_2d, src_positions, src_batch, tgt_positions, tgt_batch, **extra
            )
        if self.density_max_l == 1 and self.projection_max_l == 0:
            all_feats = self.call_source_target_density_1_feats_1(
                source_feats_2d, src_positions, src_batch, tgt_positions, tgt_batch, **extra
            )
            return all_feats[:, : self.num_radial]
        if self.density_max_l == 0 and self.projection_max_l == 1:
            padded = torch.zeros(
                source_feats_2d.shape[0],
                4,
                dtype=source_feats_2d.dtype,
                device=source_feats_2d.device,
            )
            padded[:, 0] = source_feats_2d[:, 0]
            return self.call_source_target_density_1_feats_1(
                padded, src_positions, src_batch, tgt_positions, tgt_batch, **extra
            )
        return self.call_source_target_density_1_feats_1(
            source_feats_2d, src_positions, src_batch, tgt_positions, tgt_batch, **extra
        )

    def width_correction(
        self,
        source_feats: torch.Tensor,
        src_positions: torch.Tensor,
        src_batch: torch.Tensor,
        tgt_positions: torch.Tensor,
        tgt_batch: torch.Tensor,
        edge_index: torch.Tensor,
        shifts: torch.Tensor,
        source_width: float,
        edge_chunk: int = 1 << 18,
    ) -> torch.Tensor:
        """Field of ``source_width`` sources minus the field of density-width sources,
        summed over the given pairs only. Added to a reciprocal-space result that
        treated the sources with the density width, it turns them into sources of
        width ``source_width`` (0 = point charges). The difference of the two erf
        kernels decays like erfc, which is what makes a cutoff admissible."""
        if edge_index.shape[1] == 0:
            n_out = (
                self.num_radial
                if self.projection_max_l == 0
                else 4 * self.num_radial
            )
            return tgt_positions.new_zeros((tgt_positions.shape[0], n_out))
        common = (source_feats, src_positions, src_batch, tgt_positions, tgt_batch)
        # Bounded memory: the l=1 path expands every pair 16-fold, so evaluate
        # the pairs in chunks and accumulate (the sum is linear in the edges).
        n_edges = edge_index.shape[1]
        step = max(1, int(edge_chunk))
        total = None
        for start in range(0, n_edges, step):
            sl = slice(start, min(start + step, n_edges))
            narrow = self.forward_source_target(
                *common, edge_index=edge_index[:, sl], shifts=shifts[sl], source_width=source_width
            )
            reference = self.forward_source_target(
                *common,
                edge_index=edge_index[:, sl],
                shifts=shifts[sl],
                source_width=self.density_smearing_width,
            )
            total = narrow - reference if total is None else total + (narrow - reference)
        return total

