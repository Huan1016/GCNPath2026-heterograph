#!/usr/bin/env python

import hashlib
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from utils.utils_model import seed_worker


def normalize_id(value):
    if pd.isna(value):
        return None
    text = str(value).strip()
    try:
        number = float(text)
        if number.is_integer():
            return str(int(number))
    except ValueError:
        pass
    return text


class HeterographDataset(Dataset):
    def __init__(self, ic50_data, cell_to_idx, drug_to_idx, args, no_labels=False):
        self.no_labels = no_labels
        self.cell_idx = torch.tensor(
            [cell_to_idx[normalize_id(value)] for value in ic50_data[args.col_cell]],
            dtype=torch.long,
        )
        self.drug_idx = torch.tensor(
            [drug_to_idx[normalize_id(value)] for value in ic50_data[args.col_drug]],
            dtype=torch.long,
        )
        if not no_labels:
            self.ic50 = torch.tensor(
                ic50_data[args.col_ic50].to_numpy(), dtype=torch.float32
            )

    def __len__(self):
        return len(self.cell_idx)

    def __getitem__(self, idx):
        if self.no_labels:
            return self.cell_idx[idx], self.drug_idx[idx]
        return self.cell_idx[idx], self.drug_idx[idx], self.ic50[idx]


def load_heterograph_data(root, response_data, args):
    root = Path(root)
    drug_map = pd.read_csv(root / "node_mapping/drug_node_mapping.csv")
    cell_map = pd.read_csv(root / "node_mapping/cell_node_mapping.csv")
    gene_map = pd.read_csv(root / "node_mapping/gene_node_mapping.csv")

    drug_x = torch.load(
        root / "node_features/drug_Morgan1024_x.pt", map_location="cpu"
    ).float()
    cell_x = torch.load(
        root / "node_features/cell_RNA_x.pt", map_location="cpu"
    ).float()
    gene_x = torch.load(
        root / "node_features/gene_ESM2_x.pt", map_location="cpu"
    ).float()

    drug_gene = pd.read_csv(root / "edges/01_drug_gene_Ki_pAffinity_edges.csv")
    cell_gene = pd.read_csv(root / "edges/02_cell_gene_CRISPR_edges.csv")
    gene_gene = pd.read_csv(
        root / "edges/03_gene_gene_STRING_PPI_bidirectional_edges.csv"
    )

    expected_indices = {
        "drug": (drug_map, drug_x),
        "cell": (cell_map, cell_x),
        "gene": (gene_map, gene_x),
    }
    for node_type, (mapping, features) in expected_indices.items():
        expected = list(range(len(mapping)))
        if mapping["node_idx"].tolist() != expected:
            raise ValueError(f"{node_type} node_idx is not contiguous from zero")
        if len(mapping) != features.shape[0]:
            raise ValueError(
                f"{node_type} mapping/features mismatch: "
                f"{len(mapping)} != {features.shape[0]}"
            )

    drug_to_idx = {
        normalize_id(row.GDSC_Drug): int(row.node_idx)
        for row in drug_map.itertuples()
    }
    cell_to_idx = {
        normalize_id(row.GDSC_Cell): int(row.node_idx)
        for row in cell_map.itertuples()
    }
    response_drugs = set(response_data[args.col_drug].map(normalize_id))
    response_cells = set(response_data[args.col_cell].map(normalize_id))
    missing_drugs = sorted(response_drugs - set(drug_to_idx))
    missing_cells = sorted(response_cells - set(cell_to_idx))
    extra_drugs = sorted(set(drug_to_idx) - response_drugs)
    extra_cells = sorted(set(cell_to_idx) - response_cells)
    if missing_drugs or missing_cells or extra_drugs or extra_cells:
        raise ValueError(
            "Heterograph/response node mismatch: "
            f"missing_drugs={missing_drugs}, missing_cells={missing_cells}, "
            f"extra_drugs={extra_drugs}, extra_cells={extra_cells}"
        )

    graph = {
        "drug_x": drug_x,
        "cell_x": cell_x,
        "gene_x": gene_x,
        "drug_gene_index": torch.tensor(
            drug_gene[["drug_idx", "gene_idx"]].to_numpy().T, dtype=torch.long
        ),
        "drug_gene_attr": torch.tensor(
            drug_gene["pAffinity"].to_numpy(), dtype=torch.float32
        ),
        "cell_gene_index": torch.tensor(
            cell_gene[["cell_idx", "gene_idx"]].to_numpy().T, dtype=torch.long
        ),
        "cell_gene_attr": torch.tensor(
            cell_gene["GeneEffect"].to_numpy(), dtype=torch.float32
        ),
        "gene_gene_index": torch.tensor(
            gene_gene[["gene1_idx", "gene2_idx"]].to_numpy().T,
            dtype=torch.long,
        ),
        "gene_gene_attr": torch.tensor(
            gene_gene["PPI_score"].to_numpy(), dtype=torch.float32
        ),
        "cell_to_idx": cell_to_idx,
        "drug_to_idx": drug_to_idx,
    }
    return graph


def load_heterograph_pairs(
    ic50_data,
    graph,
    args,
    no_labels=False,
    batch_size=256,
    num_workers=0,
    shuffle=False,
    fix_seed=False,
):
    ic50_data.reset_index(drop=True, inplace=True)
    dataset = HeterographDataset(
        ic50_data,
        graph["cell_to_idx"],
        graph["drug_to_idx"],
        args,
        no_labels=no_labels,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        worker_init_fn=seed_worker if fix_seed else None,
        pin_memory=args.pin_memory,
    )


def _frame_signature(frame, columns):
    canonical = frame[columns].copy()
    for column in columns:
        canonical[column] = canonical[column].map(normalize_id)
    canonical.sort_values(columns, inplace=True, kind="mergesort")
    payload = canonical.to_csv(index=False, header=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_split_signature(train_data, valid_data, test_data, args):
    columns = [args.col_cell, args.col_drug, args.col_ic50]
    signature = {
        "response": str(args.ic50),
        "choice": int(args.choice),
        "fold": int(args.nth),
        "seed_split": int(args.seed_split),
        "train_rows": len(train_data),
        "valid_rows": len(valid_data),
        "test_rows": len(test_data),
        "train_sha256": _frame_signature(train_data, columns),
        "valid_sha256": _frame_signature(valid_data, columns),
        "test_sha256": _frame_signature(test_data, columns),
        "test_drugs_sha256": _frame_signature(
            test_data.drop_duplicates(args.col_drug), [args.col_drug]
        ),
    }
    path = Path(args.dir_out) / f"split_signature_{args.nth}.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as handle:
            existing = json.load(handle)
        comparable = {key: value for key, value in signature.items() if key != "response"}
        existing_comparable = {
            key: value for key, value in existing.items() if key != "response"
        }
        if existing_comparable != comparable:
            raise ValueError(f"Split signature mismatch with existing file: {path}")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(signature, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"# Split signature: {path}")
    return signature
