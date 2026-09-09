import pytest
import torch

from graph_longrange.features import GTOElectrostaticFeatures
from graph_longrange.kspace import compute_k_vectors_flat


@pytest.mark.parametrize(
    ("pbc_handling", "pbc"),
    [
        ("pbc", [True, True, True]),
        ("slab", [True, True, False]),
        ("molecule_in_box", [False, False, False]),
        ("mixed_periodic", [True, True, False]),
        ("auto", [True, True, True]),
        ("realspace", [False, False, False]),
    ],
)
def test_feature_dtype_follows_inputs_not_process_default(pbc_handling, pbc):
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        dtype = torch.float64
        positions = torch.tensor(
            [[1.0, 1.0, 1.0], [2.0, 1.5, 1.2]], dtype=dtype
        )
        source_feats = torch.tensor([[0.4], [-0.4]], dtype=dtype)
        batch = torch.zeros(2, dtype=torch.long)
        cell = 6.0 * torch.eye(3, dtype=dtype).unsqueeze(0)
        r_cell = 2.0 * torch.pi * torch.linalg.inv(cell).transpose(-1, -2)
        volume = torch.linalg.det(cell)
        k_vectors, k_norm2, k_vector_batch, k0_mask = compute_k_vectors_flat(
            cutoff=1.2,
            cell_vectors=cell,
            r_cell_vectors=r_cell,
        )
        feature_block = GTOElectrostaticFeatures(
            density_max_l=0,
            density_smearing_width=0.4,
            feature_max_l=1,
            feature_smearing_widths=[0.3],
            include_self_interaction=False,
            kspace_cutoff=1.2,
            pbc_handling=pbc_handling,
        ).to(dtype=dtype)

        features = feature_block(
            k_vectors=k_vectors,
            k_norm2=k_norm2,
            k_vector_batch=k_vector_batch,
            k0_mask=k0_mask,
            source_feats=source_feats,
            node_positions=positions,
            batch=batch,
            volume=volume,
            pbc=torch.tensor([pbc], dtype=torch.bool),
        )

        assert features.dtype == dtype
        assert torch.isfinite(features).all()
    finally:
        torch.set_default_dtype(previous_dtype)
