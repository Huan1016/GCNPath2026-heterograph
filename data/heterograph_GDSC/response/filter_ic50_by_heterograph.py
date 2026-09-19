import os
import csv
import shutil
import pandas as pd


# ============================================================
# 1. 路径
# ============================================================

ROOT = "/homeb/lihuanhuan/20260714/260918/GCNPath2026"

# GCNPath 原始 IC50
IC50_FILE = os.path.join(
    ROOT,
    "data",
    "original",
    "ic50_data",
    "IC50_GDSC.txt"
)

# 异构图 Drug / Cell mapping
DRUG_MAP_FILE = os.path.join(
    ROOT,
    "data",
    "heterograph_GDSC",
    "node_mapping",
    "drug_node_mapping.csv"
)

CELL_MAP_FILE = os.path.join(
    ROOT,
    "data",
    "heterograph_GDSC",
    "node_mapping",
    "cell_node_mapping.csv"
)

# 输出
OUTPUT_FILE = os.path.join(
    ROOT,
    "data",
    "heterograph_GDSC",
    "response",
    "IC50_GDSC_异构图.txt"
)


# ============================================================
# 2. ID 标准化
# ============================================================

def normalize_id(x):
    """
    统一 Drug ID 格式：

    176870
    176870.0
    "176870"

    -> "176870"
    """

    if pd.isna(x):
        return None

    x = str(x).strip()

    if x == "":
        return None

    try:
        f = float(x)

        if f.is_integer():
            return str(int(f))

    except Exception:
        pass

    return x


def normalize_cell(x):
    """
    Cell ID 统一去除两端空格。
    """

    if pd.isna(x):
        return None

    return str(x).strip()


# ============================================================
# 3. 检查输入文件
# ============================================================

print("=" * 80)
print("检查输入文件")
print("=" * 80)

for path in [
    IC50_FILE,
    DRUG_MAP_FILE,
    CELL_MAP_FILE,
]:
    print(path)

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"文件不存在:\n{path}"
        )

print("\n[OK] 输入文件全部存在")


# ============================================================
# 4. 读取 GCNPath 原始 IC50
# ============================================================

print("\n" + "=" * 80)
print("读取 GCNPath 原始 IC50")
print("=" * 80)

# 从你截图中的格式来看，是 tab 分隔
ic50 = pd.read_csv(
    IC50_FILE,
    sep="\t"
)

print("文件:")
print(IC50_FILE)

print("\nshape:")
print(ic50.shape)

print("\ncolumns:")
print(ic50.columns.tolist())

print("\n前5行:")
print(ic50.head())


required_columns = [
    "Cell",
    "Drug",
    "LN_IC50"
]

missing = [
    col
    for col in required_columns
    if col not in ic50.columns
]

if missing:
    raise ValueError(
        f"IC50 文件缺少必要字段: {missing}"
    )


# ============================================================
# 5. 读取 Drug mapping
# ============================================================

print("\n" + "=" * 80)
print("读取 Drug mapping")
print("=" * 80)

drug_map = pd.read_csv(
    DRUG_MAP_FILE
)

print("shape:", drug_map.shape)
print("columns:", drug_map.columns.tolist())

if "GDSC_Drug" not in drug_map.columns:
    raise ValueError(
        "drug_node_mapping.csv 中不存在 GDSC_Drug"
    )

graph_drugs = set(
    drug_map["GDSC_Drug"]
    .map(normalize_id)
    .dropna()
)

print("异构图 Drug 数:", len(graph_drugs))


# ============================================================
# 6. 读取 Cell mapping
# ============================================================

print("\n" + "=" * 80)
print("读取 Cell mapping")
print("=" * 80)

cell_map = pd.read_csv(
    CELL_MAP_FILE
)

print("shape:", cell_map.shape)
print("columns:", cell_map.columns.tolist())

if "GDSC_Cell" not in cell_map.columns:
    raise ValueError(
        "cell_node_mapping.csv 中不存在 GDSC_Cell"
    )

graph_cells = set(
    cell_map["GDSC_Cell"]
    .map(normalize_cell)
    .dropna()
)

print("异构图 Cell 数:", len(graph_cells))


# ============================================================
# 7. 标准化 IC50 中的 Drug 和 Cell
# ============================================================

ic50["_Drug_norm"] = (
    ic50["Drug"]
    .map(normalize_id)
)

ic50["_Cell_norm"] = (
    ic50["Cell"]
    .map(normalize_cell)
)


# ============================================================
# 8. 原始数据统计
# ============================================================

print("\n" + "=" * 80)
print("原始 IC50_GDSC.txt 统计")
print("=" * 80)

original_samples = len(ic50)

original_drugs = (
    ic50["_Drug_norm"]
    .nunique()
)

original_cells = (
    ic50["_Cell_norm"]
    .nunique()
)

print(f"样本数 : {original_samples:,}")
print(f"Drug数 : {original_drugs:,}")
print(f"Cell数 : {original_cells:,}")


# ============================================================
# 9. 检查与异构图交集
# ============================================================

drug_mask = (
    ic50["_Drug_norm"]
    .isin(graph_drugs)
)

cell_mask = (
    ic50["_Cell_norm"]
    .isin(graph_cells)
)

both_mask = (
    drug_mask &
    cell_mask
)


print("\n" + "=" * 80)
print("异构图覆盖情况")
print("=" * 80)

print(
    f"Drug 匹配样本 : "
    f"{drug_mask.sum():,} / {original_samples:,} "
    f"({drug_mask.mean() * 100:.2f}%)"
)

print(
    f"Cell 匹配样本 : "
    f"{cell_mask.sum():,} / {original_samples:,} "
    f"({cell_mask.mean() * 100:.2f}%)"
)

print(
    f"Drug + Cell 同时匹配 : "
    f"{both_mask.sum():,} / {original_samples:,} "
    f"({both_mask.mean() * 100:.2f}%)"
)


# ============================================================
# 10. 正式筛选
# ============================================================

filtered = (
    ic50.loc[both_mask]
    .copy()
)

# 删除辅助列
filtered.drop(
    columns=[
        "_Drug_norm",
        "_Cell_norm"
    ],
    inplace=True
)


# ============================================================
# 11. 最终统计
# ============================================================

final_samples = len(filtered)

final_drugs = (
    filtered["Drug"]
    .map(normalize_id)
    .nunique()
)

final_cells = (
    filtered["Cell"]
    .map(normalize_cell)
    .nunique()
)


print("\n" + "=" * 80)
print("筛选后统计")
print("=" * 80)

print(
    f"样本数 : "
    f"{original_samples:,} -> {final_samples:,}"
)

print(
    f"Drug数 : "
    f"{original_drugs:,} -> {final_drugs:,}"
)

print(
    f"Cell数 : "
    f"{original_cells:,} -> {final_cells:,}"
)

print(
    f"删除样本 : "
    f"{original_samples - final_samples:,}"
)


# ============================================================
# 12. 查看实际进入 response 的 Drug / Cell
# ============================================================

response_drugs = set(
    filtered["Drug"]
    .map(normalize_id)
    .dropna()
)

response_cells = set(
    filtered["Cell"]
    .map(normalize_cell)
    .dropna()
)

graph_drugs_no_response = (
    graph_drugs -
    response_drugs
)

graph_cells_no_response = (
    graph_cells -
    response_cells
)

print("\n" + "=" * 80)
print("Graph 与最终 Response 的节点覆盖")
print("=" * 80)

print(
    f"异构图 Drug : {len(graph_drugs)}"
)

print(
    f"Response Drug : {len(response_drugs)}"
)

print(
    f"图中但没有 Response 的 Drug : "
    f"{len(graph_drugs_no_response)}"
)

print(
    f"\n异构图 Cell : {len(graph_cells)}"
)

print(
    f"Response Cell : {len(response_cells)}"
)

print(
    f"图中但没有 Response 的 Cell : "
    f"{len(graph_cells_no_response)}"
)


# ============================================================
# 13. 严格检查最终 response
# ============================================================

missing_drugs = (
    response_drugs -
    graph_drugs
)

missing_cells = (
    response_cells -
    graph_cells
)

assert len(missing_drugs) == 0, (
    f"存在不属于异构图的 Drug: {missing_drugs}"
)

assert len(missing_cells) == 0, (
    f"存在不属于异构图的 Cell: {missing_cells}"
)

print("\n[OK] 所有最终 Drug 都存在于异构图")
print("[OK] 所有最终 Cell 都存在于异构图")


# ============================================================
# 14. LN_IC50 检查
# ============================================================

print("\n" + "=" * 80)
print("LN_IC50")
print("=" * 80)

print(
    filtered["LN_IC50"]
    .describe()
)

print(
    "\nLN_IC50 NaN:",
    filtered["LN_IC50"]
    .isna()
    .sum()
)


# ============================================================
# 15. 如果旧结果存在，先备份
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

if os.path.exists(OUTPUT_FILE):

    backup_file = (
        OUTPUT_FILE +
        ".backup"
    )

    shutil.copy2(
        OUTPUT_FILE,
        backup_file
    )

    print("\n旧文件已备份:")
    print(backup_file)


# ============================================================
# 16. 保存
#
# 使用 QUOTE_NONNUMERIC：
# 字符串带双引号，数值不带双引号，
# 尽量保持你截图中 GCNPath 原始文件的格式。
# ============================================================

filtered.to_csv(
    OUTPUT_FILE,
    sep="\t",
    index=False,
    quoting=csv.QUOTE_NONNUMERIC
)


print("\n" + "=" * 80)
print("保存完成")
print("=" * 80)

print(OUTPUT_FILE)

print("\n最终 shape:")
print(filtered.shape)

print("\n最终 columns:")
print(filtered.columns.tolist())

print("\n前5行:")
print(filtered.head())

print("\nALL DONE")