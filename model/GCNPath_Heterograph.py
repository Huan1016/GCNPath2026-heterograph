#!/usr/bin/env python

import torch
from torch import nn

from utils.utils_gnn import MLP


def _normalized_sparse(rows, cols, values, shape, absolute_degree=False):
    values = values.float()
    degree_values = values.abs() if absolute_degree else values
    degree = torch.zeros(shape[0], dtype=values.dtype)
    degree.index_add_(0, rows, degree_values)
    normalized = values / degree[rows].clamp_min(1e-12)
    indices = torch.stack([rows, cols], dim=0)
    return torch.sparse_coo_tensor(indices, normalized, shape).coalesce()


class HeterographEncoder(nn.Module):
    def __init__(
        self,
        graph,
        hidden_dim=128,
        dim_embed_cell=256,
        dim_embed_drug=256,
        act=nn.ELU(1.0),
        drop=0.2,
    ):
        super().__init__()
        drug_x = graph["drug_x"]
        cell_x = graph["cell_x"]
        gene_x = graph["gene_x"]

        ppi = _normalized_sparse(
            graph["gene_gene_index"][1],
            graph["gene_gene_index"][0],
            graph["gene_gene_attr"],
            (gene_x.shape[0], gene_x.shape[0]),
        )
        drug_gene = _normalized_sparse(
            graph["drug_gene_index"][0],
            graph["drug_gene_index"][1],
            graph["drug_gene_attr"],
            (drug_x.shape[0], gene_x.shape[0]),
        )
        cell_gene_signed = _normalized_sparse(
            graph["cell_gene_index"][0],
            graph["cell_gene_index"][1],
            graph["cell_gene_attr"],
            (cell_x.shape[0], gene_x.shape[0]),
            absolute_degree=True,
        )
        cell_gene_magnitude = _normalized_sparse(
            graph["cell_gene_index"][0],
            graph["cell_gene_index"][1],
            graph["cell_gene_attr"].abs(),
            (cell_x.shape[0], gene_x.shape[0]),
        )

        # These graph aggregations are fixed inputs. Relation-specific learned
        # transformations below remain trainable while avoiding a 675k-edge
        # sparse traversal for every response mini-batch.
        ppi_gene_x = gene_x + torch.sparse.mm(ppi, gene_x)
        drug_gene_x = torch.sparse.mm(drug_gene, ppi_gene_x)
        cell_gene_signed_x = torch.sparse.mm(cell_gene_signed, ppi_gene_x)
        cell_gene_magnitude_x = torch.sparse.mm(cell_gene_magnitude, ppi_gene_x)

        self.register_buffer("drug_x", drug_x)
        self.register_buffer("cell_x", cell_x)
        self.register_buffer("drug_gene_x", drug_gene_x)
        self.register_buffer("cell_gene_signed_x", cell_gene_signed_x)
        self.register_buffer("cell_gene_magnitude_x", cell_gene_magnitude_x)

        gene_dim = gene_x.shape[1]
        self.gene_projection = MLP(
            [gene_dim, hidden_dim], act=act, norm="layer", drop=drop
        )
        self.drug_self = MLP(
            [drug_x.shape[1], hidden_dim], act=act, norm="layer", drop=drop
        )
        self.drug_relation = MLP(
            [hidden_dim, hidden_dim], act=act, norm="layer", drop=drop
        )
        self.drug_output = MLP(
            [2 * hidden_dim, dim_embed_drug],
            act=act,
            norm="layer",
            drop=drop,
        )

        self.cell_self = MLP(
            [cell_x.shape[1], hidden_dim], act=act, norm="layer", drop=drop
        )
        self.cell_signed_relation = MLP(
            [hidden_dim, hidden_dim], act=act, norm="layer", drop=drop
        )
        self.cell_magnitude_relation = MLP(
            [hidden_dim, hidden_dim], act=act, norm="layer", drop=drop
        )
        self.cell_output = MLP(
            [3 * hidden_dim, dim_embed_cell],
            act=act,
            norm="layer",
            drop=drop,
        )

    def forward(self, cell_idx, drug_idx):
        cell_idx = cell_idx.long()
        drug_idx = drug_idx.long()
        drug_gene = self.gene_projection(self.drug_gene_x[drug_idx])
        cell_gene_signed = self.gene_projection(self.cell_gene_signed_x[cell_idx])
        cell_gene_magnitude = self.gene_projection(self.cell_gene_magnitude_x[cell_idx])
        drug = torch.cat(
            [
                self.drug_self(self.drug_x[drug_idx]),
                self.drug_relation(drug_gene),
            ],
            dim=1,
        )
        cell = torch.cat(
            [
                self.cell_self(self.cell_x[cell_idx]),
                self.cell_signed_relation(cell_gene_signed),
                self.cell_magnitude_relation(cell_gene_magnitude),
            ],
            dim=1,
        )
        return self.cell_output(cell), self.drug_output(drug)


class GCNPathHeterograph(nn.Module):
    def __init__(
        self,
        graph,
        hidden_dim=128,
        dim_embed_cell=256,
        dim_embed_drug=256,
        dim_pred=(512, 512),
        act=nn.ELU(1.0),
        drop=0.2,
    ):
        super().__init__()
        self.attn_mode = 0
        self.encoder = HeterographEncoder(
            graph=graph,
            hidden_dim=hidden_dim,
            dim_embed_cell=dim_embed_cell,
            dim_embed_drug=dim_embed_drug,
            act=act,
            drop=drop,
        )
        self.final_layer = MLP(
            [dim_embed_cell + dim_embed_drug, *dim_pred, 1],
            act=act,
            norm=True,
            drop=drop,
            last_nn=True,
        )

    def forward(self, cell_idx, drug_idx, return_attn=False, grad_cam=False):
        if return_attn or grad_cam:
            raise ValueError("Heterograph model does not expose attention/Grad-CAM")
        cell, drug = self.encoder(cell_idx, drug_idx)
        return self.final_layer(torch.cat([cell, drug], dim=1))

    def summary(self, verbose=True):
        if verbose:
            print(f"\n### Model Summary\n{self}\n")
        total = sum(parameter.numel() for parameter in self.parameters())
        encoder = sum(parameter.numel() for parameter in self.encoder.parameters())
        prediction = sum(parameter.numel() for parameter in self.final_layer.parameters())
        print(f"# Model Parameters : {total}")
        print(f"# Model Parameters : {encoder} [Heterograph Encoder]")
        print(f"# Model Parameters : {prediction} [Final]")
