#!/usr/bin/env python3
"""Align heterograph nodes, features, and edges to the response table."""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch


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


def load_tensor(path):
    return torch.load(path, map_location="cpu")


def compact_nodes(mapping, features, keep_mask, index_column):
    kept = mapping.loc[keep_mask].copy()
    source_indices = kept[index_column].astype(int).tolist()
    if source_indices and max(source_indices) >= features.shape[0]:
        raise ValueError(
            f"{index_column} references feature row {max(source_indices)}, "
            f"but the tensor has only {features.shape[0]} rows"
        )

    old_to_new = {
        old_idx: new_idx for new_idx, old_idx in enumerate(source_indices)
    }
    compact_features = features[source_indices].clone()
    kept[index_column] = range(len(kept))
    kept.reset_index(drop=True, inplace=True)
    index_map = pd.DataFrame(
        {
            f"old_{index_column}": source_indices,
            f"new_{index_column}": range(len(source_indices)),
        }
    )
    return kept, compact_features, old_to_new, index_map


def remap_column(frame, column, index_map):
    frame = frame[frame[column].isin(index_map)].copy()
    frame[column] = frame[column].map(index_map).astype(int)
    return frame


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "heterograph_GDSC",
    )
    parser.add_argument(
        "--response",
        default="response/IC50_GDSC_异构图.txt",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = args.root.resolve()

    paths = {
        "drug_map": root / "node_mapping/drug_node_mapping.csv",
        "cell_map": root / "node_mapping/cell_node_mapping.csv",
        "gene_map": root / "node_mapping/gene_node_mapping.csv",
        "drug_x": root / "node_features/drug_Morgan1024_x.pt",
        "cell_x": root / "node_features/cell_RNA_x.pt",
        "gene_x": root / "node_features/gene_ESM2_x.pt",
        "drug_gene": root / "edges/01_drug_gene_Ki_pAffinity_edges.csv",
        "cell_gene": root / "edges/02_cell_gene_CRISPR_edges.csv",
        "gene_gene": root / "edges/03_gene_gene_STRING_PPI_bidirectional_edges.csv",
        "metadata": root / "metadata/graph_metadata.json",
        "response": root / args.response,
    }

    drug_map = pd.read_csv(paths["drug_map"])
    cell_map = pd.read_csv(paths["cell_map"])
    gene_map = pd.read_csv(paths["gene_map"])
    drug_x = load_tensor(paths["drug_x"])
    cell_x = load_tensor(paths["cell_x"])
    gene_x = load_tensor(paths["gene_x"])
    drug_gene = pd.read_csv(paths["drug_gene"])
    cell_gene = pd.read_csv(paths["cell_gene"])
    gene_gene = pd.read_csv(paths["gene_gene"])
    response = pd.read_csv(paths["response"], sep="\t")

    response_drugs = set(response["Drug"].map(normalize_id))
    response_cells = set(response["Cell"].map(normalize_id))
    graph_drugs = set(drug_map["GDSC_Drug"].map(normalize_id))
    graph_cells = set(cell_map["GDSC_Cell"].map(normalize_id))
    missing_drugs = sorted(response_drugs - graph_drugs)
    missing_cells = sorted(response_cells - graph_cells)
    if missing_drugs or missing_cells:
        raise ValueError(
            f"Response nodes missing from graph: drugs={missing_drugs}, "
            f"cells={missing_cells}"
        )

    original_counts = {
        "drug": len(drug_map),
        "cell": len(cell_map),
        "gene": len(gene_map),
        "drug_gene": len(drug_gene),
        "cell_gene": len(cell_gene),
        "gene_gene": len(gene_gene),
    }

    drug_keep = drug_map["GDSC_Drug"].map(normalize_id).isin(response_drugs)
    cell_keep = cell_map["GDSC_Cell"].map(normalize_id).isin(response_cells)
    removed_drugs = drug_map.loc[~drug_keep].copy()
    removed_cells = cell_map.loc[~cell_keep].copy()

    drug_map, drug_x, drug_idx_map, drug_idx_report = compact_nodes(
        drug_map, drug_x, drug_keep, "node_idx"
    )
    cell_map, cell_x, cell_idx_map, cell_idx_report = compact_nodes(
        cell_map, cell_x, cell_keep, "node_idx"
    )
    drug_gene = remap_column(drug_gene, "drug_idx", drug_idx_map)
    cell_gene = remap_column(cell_gene, "cell_idx", cell_idx_map)

    incident_genes = (
        set(drug_gene["gene_idx"])
        | set(cell_gene["gene_idx"])
        | set(gene_gene["gene1_idx"])
        | set(gene_gene["gene2_idx"])
    )
    gene_keep = gene_map["node_idx"].isin(incident_genes)
    removed_genes = gene_map.loc[~gene_keep].copy()
    gene_map, gene_x, gene_idx_map, gene_idx_report = compact_nodes(
        gene_map, gene_x, gene_keep, "node_idx"
    )
    drug_gene = remap_column(drug_gene, "gene_idx", gene_idx_map)
    cell_gene = remap_column(cell_gene, "gene_idx", gene_idx_map)
    gene_gene = remap_column(gene_gene, "gene1_idx", gene_idx_map)
    gene_gene = remap_column(gene_gene, "gene2_idx", gene_idx_map)

    isolated_drugs = set(drug_map["node_idx"]) - set(drug_gene["drug_idx"])
    isolated_cells = set(cell_map["node_idx"]) - set(cell_gene["cell_idx"])
    gene_incident = (
        set(drug_gene["gene_idx"])
        | set(cell_gene["gene_idx"])
        | set(gene_gene["gene1_idx"])
        | set(gene_gene["gene2_idx"])
    )
    isolated_genes = set(gene_map["node_idx"]) - gene_incident
    if isolated_drugs or isolated_cells or isolated_genes:
        raise ValueError(
            "Isolated response-aligned nodes remain: "
            f"drug={sorted(isolated_drugs)}, cell={sorted(isolated_cells)}, "
            f"gene={sorted(isolated_genes)}"
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = root / "backups" / f"before_response_alignment_{timestamp}"
    for key in (
        "drug_map", "cell_map", "gene_map", "drug_x", "cell_x", "gene_x",
        "drug_gene", "cell_gene", "gene_gene", "metadata",
    ):
        source = paths[key]
        destination = backup_dir / source.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    drug_map.to_csv(paths["drug_map"], index=False)
    cell_map.to_csv(paths["cell_map"], index=False)
    gene_map.to_csv(paths["gene_map"], index=False)
    torch.save(drug_x, paths["drug_x"])
    torch.save(cell_x, paths["cell_x"])
    torch.save(gene_x, paths["gene_x"])
    drug_gene.to_csv(paths["drug_gene"], index=False)
    cell_gene.to_csv(paths["cell_gene"], index=False)
    gene_gene.to_csv(paths["gene_gene"], index=False)

    report_dir = root / "response_alignment_report"
    report_dir.mkdir(parents=True, exist_ok=True)
    drug_idx_report.to_csv(report_dir / "drug_idx_old_to_new.csv", index=False)
    cell_idx_report.to_csv(report_dir / "cell_idx_old_to_new.csv", index=False)
    gene_idx_report.to_csv(report_dir / "gene_idx_old_to_new.csv", index=False)
    removed_drugs.to_csv(report_dir / "removed_drugs_not_in_response.csv", index=False)
    removed_cells.to_csv(report_dir / "removed_cells_not_in_response.csv", index=False)
    removed_genes.to_csv(report_dir / "removed_isolated_genes.csv", index=False)

    with open(paths["metadata"], "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    metadata["node_counts"] = {
        "drug": len(drug_map), "cell": len(cell_map), "gene": len(gene_map)
    }
    metadata["edge_counts"] = {
        "drug__binds_ki__gene": len(drug_gene),
        "gene__rev_binds_ki__drug": len(drug_gene),
        "cell__crispr__gene": len(cell_gene),
        "gene__rev_crispr__cell": len(cell_gene),
        "gene__ppi__gene": len(gene_gene),
    }
    metadata["missing_or_isolated"] = {
        "missing_crispr_pairs": len(cell_map) * len(gene_map) - len(cell_gene),
        "drugs_without_ki_edge": 0,
        "cells_without_crispr_edge": 0,
        "genes_without_any_edge": 0,
    }
    metadata["response_alignment"] = {
        "response_file": str(paths["response"].relative_to(root)),
        "response_rows": len(response),
        "response_drugs": len(response_drugs),
        "response_cells": len(response_cells),
        "aligned_at": timestamp,
        "backup_directory": str(backup_dir.relative_to(root)),
    }
    with open(paths["metadata"], "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    final_counts = {
        "drug": len(drug_map),
        "cell": len(cell_map),
        "gene": len(gene_map),
        "drug_gene": len(drug_gene),
        "cell_gene": len(cell_gene),
        "gene_gene": len(gene_gene),
    }
    lines = [
        "Heterograph Response Alignment",
        "=" * 80,
        f"Response rows: {len(response)}",
        f"Response drugs: {len(response_drugs)}",
        f"Response cells: {len(response_cells)}",
        f"Backup: {backup_dir}",
        "",
    ]
    for name in final_counts:
        lines.append(f"{name}: {original_counts[name]} -> {final_counts[name]}")
    lines.extend(
        [
            "",
            "Isolated drug nodes: 0",
            "Isolated cell nodes: 0",
            "Isolated gene nodes: 0",
        ]
    )
    (report_dir / "alignment_statistics.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
