import os
import pandas as pd
import torch


ROOT = (
    os.path.dirname(os.path.abspath(__file__))
    + "/data/heterograph_GDSC"
)


def norm_id(x):
    if pd.isna(x):
        return None

    x = str(x).strip()

    try:
        f = float(x)
        if f.is_integer():
            return str(int(f))
    except Exception:
        pass

    return x


# ============================================================
# 1. Node mappings
# ============================================================

drug_map = pd.read_csv(
    os.path.join(
        ROOT,
        "node_mapping",
        "drug_node_mapping.csv"
    )
)

cell_map = pd.read_csv(
    os.path.join(
        ROOT,
        "node_mapping",
        "cell_node_mapping.csv"
    )
)

gene_map = pd.read_csv(
    os.path.join(
        ROOT,
        "node_mapping",
        "gene_node_mapping.csv"
    )
)


# ============================================================
# 2. Node features
# ============================================================

drug_x = torch.load(
    os.path.join(
        ROOT,
        "node_features",
        "drug_Morgan1024_x.pt"
    ),
    map_location="cpu"
)

cell_x = torch.load(
    os.path.join(
        ROOT,
        "node_features",
        "cell_RNA_x.pt"
    ),
    map_location="cpu"
)

gene_x = torch.load(
    os.path.join(
        ROOT,
        "node_features",
        "gene_ESM2_x.pt"
    ),
    map_location="cpu"
)


# ============================================================
# 3. Edges
# ============================================================

drug_gene = pd.read_csv(
    os.path.join(
        ROOT,
        "edges",
        "01_drug_gene_Ki_pAffinity_edges.csv"
    )
)

cell_gene = pd.read_csv(
    os.path.join(
        ROOT,
        "edges",
        "02_cell_gene_CRISPR_edges.csv"
    )
)

gene_gene = pd.read_csv(
    os.path.join(
        ROOT,
        "edges",
        "03_gene_gene_STRING_PPI_bidirectional_edges.csv"
    )
)


# ============================================================
# 4. Response
# ============================================================

response = pd.read_csv(
    os.path.join(
        ROOT,
        "response",
        "IC50_GDSC_异构图.txt"
    ),
    sep="\t"
)


# ============================================================
# 5. 基本统计
# ============================================================

print("=" * 80)
print("NODE MAPPING")
print("=" * 80)

print("Drug:", drug_map.shape)
print("Cell:", cell_map.shape)
print("Gene:", gene_map.shape)


print("\n" + "=" * 80)
print("NODE FEATURES")
print("=" * 80)

print("Drug X:", tuple(drug_x.shape))
print("Cell X:", tuple(cell_x.shape))
print("Gene X:", tuple(gene_x.shape))

print("Drug NaN:", torch.isnan(drug_x).sum().item())
print("Cell NaN:", torch.isnan(cell_x).sum().item())
print("Gene NaN:", torch.isnan(gene_x).sum().item())


print("\n" + "=" * 80)
print("EDGES")
print("=" * 80)

print("Drug-Gene:", drug_gene.shape)
print("Cell-Gene:", cell_gene.shape)
print("Gene-Gene:", gene_gene.shape)

print("\nDrug-Gene columns:")
print(drug_gene.columns.tolist())

print("\nCell-Gene columns:")
print(cell_gene.columns.tolist())

print("\nGene-Gene columns:")
print(gene_gene.columns.tolist())


print("\n" + "=" * 80)
print("RESPONSE")
print("=" * 80)

print("Response shape:", response.shape)
print("Response columns:", response.columns.tolist())

print(
    "Response Drug:",
    response["Drug"].nunique()
)

print(
    "Response Cell:",
    response["Cell"].nunique()
)

print("\nLN_IC50:")
print(response["LN_IC50"].describe())


# ============================================================
# 6. Mapping / feature alignment
# ============================================================

print("\n" + "=" * 80)
print("NODE / FEATURE ALIGNMENT")
print("=" * 80)

assert len(drug_map) == drug_x.shape[0], (
    "Drug mapping 与 Morgan 行数不一致"
)

assert len(cell_map) == cell_x.shape[0], (
    "Cell mapping 与 RNA 行数不一致"
)

assert len(gene_map) == gene_x.shape[0], (
    "Gene mapping 与 ESM2 行数不一致"
)

print("[OK] Drug mapping <-> Morgan")
print("[OK] Cell mapping <-> RNA")
print("[OK] Gene mapping <-> ESM2")


# ============================================================
# 7. Edge index 检查
# ============================================================

print("\n" + "=" * 80)
print("EDGE INDEX")
print("=" * 80)

assert drug_gene["drug_idx"].min() >= 0
assert drug_gene["drug_idx"].max() < len(drug_map)

assert drug_gene["gene_idx"].min() >= 0
assert drug_gene["gene_idx"].max() < len(gene_map)

print("[OK] Drug-Gene")


assert cell_gene["cell_idx"].min() >= 0
assert cell_gene["cell_idx"].max() < len(cell_map)

assert cell_gene["gene_idx"].min() >= 0
assert cell_gene["gene_idx"].max() < len(gene_map)

print("[OK] Cell-Gene")


gene_idx_cols = [
    col
    for col in gene_gene.columns
    if "idx" in col.lower()
]

print(
    "Gene-Gene index columns:",
    gene_idx_cols
)

for col in gene_idx_cols:
    assert gene_gene[col].min() >= 0
    assert gene_gene[col].max() < len(gene_map)

print("[OK] Gene-Gene")


# ============================================================
# 8. Response 是否全部存在于 graph
# ============================================================

print("\n" + "=" * 80)
print("RESPONSE VS GRAPH")
print("=" * 80)

graph_drugs = set(
    drug_map["GDSC_Drug"]
    .map(norm_id)
    .dropna()
)

graph_cells = set(
    cell_map["GDSC_Cell"]
    .astype(str)
    .str.strip()
)

response_drugs = set(
    response["Drug"]
    .map(norm_id)
    .dropna()
)

response_cells = set(
    response["Cell"]
    .astype(str)
    .str.strip()
)

missing_drugs = (
    response_drugs -
    graph_drugs
)

missing_cells = (
    response_cells -
    graph_cells
)

print(
    "Response Drug 不在图中:",
    len(missing_drugs)
)

print(
    "Response Cell 不在图中:",
    len(missing_cells)
)

if missing_drugs:
    print(
        "Missing Drugs:",
        sorted(missing_drugs)[:20]
    )

if missing_cells:
    print(
        "Missing Cells:",
        sorted(missing_cells)[:20]
    )

extra_drugs = graph_drugs - response_drugs
extra_cells = graph_cells - response_cells

assert len(missing_drugs) == 0
assert len(missing_cells) == 0
assert len(extra_drugs) == 0
assert len(extra_cells) == 0

print("[OK] 所有 response Drug 均存在于异构图")
print("[OK] 所有 response Cell 均存在于异构图")
print("[OK] 图中不存在 response 之外的 Drug/Cell")

isolated_drugs = set(drug_map["node_idx"]) - set(drug_gene["drug_idx"])
isolated_cells = set(cell_map["node_idx"]) - set(cell_gene["cell_idx"])
incident_genes = (
    set(drug_gene["gene_idx"])
    | set(cell_gene["gene_idx"])
    | set(gene_gene["gene1_idx"])
    | set(gene_gene["gene2_idx"])
)
isolated_genes = set(gene_map["node_idx"]) - incident_genes

assert len(isolated_drugs) == 0
assert len(isolated_cells) == 0
assert len(isolated_genes) == 0

assert len(response) == 89840
assert response["Drug"].nunique() == 165
assert response["Cell"].nunique() == 592

print("[OK] Drug/Cell/Gene 均无孤立节点")
print("[OK] Response 为 89840 rows / 165 Drugs / 592 Cells")

# ============================================================
# 9. 最终结果
# ============================================================

print("\n" + "=" * 80)
print("ALL CHECKS PASSED")
print("=" * 80)