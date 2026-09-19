# GCNPath2026 Drug–Cell–Gene 异构图改造交接文档

## 0. 项目位置与 Git 基线

项目根目录：

```text
/homeb/lihuanhuan/20260714/260918/GCNPath2026
```

原始仓库：

```text
https://github.com/MinhoLee-DGU/GCNPath2026.git
```

原始 GCNPath baseline 的 Git tag：

```text
heterograph-baseline
```

Baseline commit：

```text
f4fa82d
```

当前开发分支：

```text
heterograph-gdsc
```

在修改任何内容之前，请先执行：

```bash
cd /homeb/lihuanhuan/20260714/260918/GCNPath2026

git rev-parse --abbrev-ref HEAD
git status --short
git log --oneline -10
git diff --name-status heterograph-baseline
git diff --stat heterograph-baseline
git diff heterograph-baseline -- '*.py'
```

将 `heterograph-baseline` 视为**不可修改的原始 GCNPath 参考实现**。

不要 reset、删除或覆盖用户数据。

---

# 1. 总体目标

在原始 GCNPath 实现基础上，增加对新的 **Drug–Cell–Gene 异构图输入**的支持。

本次修改必须尽量小。

**只允许修改以下两个方面：**

1. 数据输入
2. 模型结构

**不要重新设计训练框架。**

本次实验的目的，是在完全一致的实验条件下比较：

- A. 原始 GCNPath baseline 架构
- B. 新的 Drug–Cell–Gene 异构图架构

两者必须严格保持以下内容一致：

- response 样本
- train / validation / test 划分
- fold 定义
- 随机种子
- label
- loss
- optimizer
- scheduler
- epoch 数
- early stopping
- 评价指标
- 输出格式

**唯一希望存在的实验差异，应当是输入表示方式和模型表示方式。**

---

# 2. 最关键的公平比较要求

Baseline 和 heterograph 模型必须使用**完全相同的 response 文件**。

新的默认 IC50 文件为：

```text
/homeb/lihuanhuan/20260714/260918/GCNPath2026/data/heterograph_GDSC/response/IC50_GDSC_异构图.txt
```

**禁止** baseline 使用原始完整数据：

```text
data/ic50_data/IC50_GDSC.txt
```

而 heterograph 使用经过筛选的数据。

如果这样做，两者结果将不再公平可比。

Baseline 和 heterograph 都必须默认使用：

```text
data/heterograph_GDSC/response/IC50_GDSC_异构图.txt
```

当前 response 数据统计：

- 样本数：90,005
- 药物数：165
- 有 response 的细胞系数：593
- 预测目标：`LN_IC50`
- `LN_IC50` 缺失值：0

Response 文件列：

- `Cell`
- `Cell_BROAD`
- `Cell_COSMIC`
- `Drug`
- `LN_IC50`

全部 165 个 response 药物均存在于异构图中。

全部 593 个 response 细胞系均存在于异构图中。

---

# 3. 数据划分规则：必须保持原始 GCNPath 逻辑

**不要修改 GCNPath 原有的数据划分算法。**

原始 split 实现必须继续作为唯一的数据划分来源。

无论 GCNPath 当前支持哪些 split 模式，都必须继续保持其原始语义，包括已有的：

- random split
- blind split
- Drug-Blind
- Cell-Blind
- 其他原有 split

**模型类型不能影响数据划分。**

正确流程必须保持为：

```text
IC50_GDSC_异构图.txt
        |
        v
原始 GCNPath split 逻辑
        |
        +------------+-------------+
        |            |             |
      train        valid          test
        |            |             |
        +------------+-------------+
                     |
                     v
               同一套 split
                     |
           +---------+---------+
           |                   |
     baseline 模型        heterograph 模型
```

对于完全相同的：

- seed
- fold
- split mode

baseline 与 heterograph 必须接收到**完全相同的 train / validation / test 样本**。

禁止：

- 新建独立的 heterograph 专用划分逻辑
- split 后再次按模型类型筛样本
- baseline 与 heterograph 使用不同 shuffle
- 根据模型类型重新生成 fold

---

# 4. Baseline 实验要求

Baseline 模型结构必须保持为**原始 GCNPath 模型**。

Baseline 条件下，唯一希望发生的数据相关改动是：

旧的默认 response 来源：

```text
原始 GCNPath IC50 数据
```

新的默认 response 来源：

```text
/homeb/lihuanhuan/20260714/260918/GCNPath2026/data/heterograph_GDSC/response/IC50_GDSC_异构图.txt
```

不要修改 baseline 的：

- encoder 架构
- hidden dimension
- loss
- optimizer
- prediction head
- training loop
- evaluation code

除非为了兼容新的 response 子集而必须做极小的兼容性修改。

Baseline 是本实验中的**对照组**。

---

# 5. 异构图数据位置

异构图根目录：

```text
/homeb/lihuanhuan/20260714/260918/GCNPath2026/data/heterograph_GDSC
```

目录结构：

```text
data/heterograph_GDSC/
|
├── response/
│   └── IC50_GDSC_异构图.txt
|
├── node_mapping/
│   ├── drug_node_mapping.csv
│   ├── cell_node_mapping.csv
│   └── gene_node_mapping.csv
|
├── node_features/
│   ├── drug_Morgan1024_x.pt
│   ├── cell_RNA_x.pt
│   ├── gene_ESM2_x.pt
│   ├── drug_Morgan_feature_names.csv
│   └── cell_RNA_feature_names.csv
|
├── edges/
│   ├── 01_drug_gene_Ki_pAffinity_edges.csv
│   ├── 02_cell_gene_CRISPR_edges.csv
│   └── 03_gene_gene_STRING_PPI_bidirectional_edges.csv
|
└── metadata/
    ├── graph_metadata.json
    └── graph_statistics.txt
```

---

# 6. 异构图节点定义

异构图包含三种节点类型：

1. Drug
2. Cell
3. Gene

---

## 6.1 Drug 节点

节点数量：

```text
165
```

节点映射：

```text
data/heterograph_GDSC/node_mapping/drug_node_mapping.csv
```

重要列：

- `node_idx`
- `GDSC_Drug`

Drug 节点特征：

```text
data/heterograph_GDSC/node_features/drug_Morgan1024_x.pt
```

形状：

```text
165 × 1024
```

特征含义：

```text
Morgan fingerprint，1024 维
```

重要不变量：

```text
drug_Morgan1024_x.pt 第 i 行
必须对应 drug_node_mapping.csv 中 node_idx == i
```

Drug 特征中无 NaN。

---

## 6.2 Cell 节点

节点数量：

```text
617
```

节点映射：

```text
data/heterograph_GDSC/node_mapping/cell_node_mapping.csv
```

重要列包括：

- `node_idx`
- `GDSC_Cell`
- `DepMap_ModelID`
- `SIDM`
- `COSMICID`
- `CCLEName`

Cell 节点特征：

```text
data/heterograph_GDSC/node_features/cell_RNA_x.pt
```

形状：

```text
617 × 1410
```

特征含义：

```text
RNA expression
```

重要不变量：

```text
cell_RNA_x.pt 第 i 行
必须对应 cell_node_mapping.csv 中 node_idx == i
```

无 NaN。

目前 617 个 Cell 节点中：

```text
593 个细胞系存在 LN_IC50 response
24 个细胞系没有 LN_IC50 response
```

**不要自动删除这 24 个细胞系。**

它们仍然是合法图节点，即使不直接参与 response 监督，也可以通过异构图消息传播参与表示学习。

---

## 6.3 Gene 节点

节点数量：

```text
1157
```

节点映射：

```text
data/heterograph_GDSC/node_mapping/gene_node_mapping.csv
```

重要列：

- `node_idx`
- `Entrez_ID`

Gene 节点特征：

```text
data/heterograph_GDSC/node_features/gene_ESM2_x.pt
```

形状：

```text
1157 × 1280
```

特征含义：

```text
ESM2 蛋白序列 embedding
```

无 NaN。

重要不变量：

```text
gene_ESM2_x.pt 第 i 行
必须对应 gene_node_mapping.csv 中 node_idx == i
```

---

# 7. 异构图边定义

存在三类生物关系。

---

## 7.1 Drug → Gene

文件：

```text
data/heterograph_GDSC/edges/01_drug_gene_Ki_pAffinity_edges.csv
```

边数量：

```text
602
```

字段：

- `GDSC_Drug`
- `Entrez_ID`
- `Ki_nM`
- `source_rows`
- `pAffinity`
- `drug_idx`
- `gene_idx`

使用：

```text
drug_idx -> gene_idx
```

关键边属性：

```text
pAffinity
```

`pAffinity` 表示经过对数变换后的亲和力信息，应作为 Drug–Gene 关系属性使用。

**不要使用 LN_IC50 替代 pAffinity。**

---

## 7.2 Cell → Gene

文件：

```text
data/heterograph_GDSC/edges/02_cell_gene_CRISPR_edges.csv
```

边数量：

```text
703,597
```

字段：

- `cell_idx`
- `GDSC_Cell`
- `DepMap_ModelID`
- `gene_idx`
- `Entrez_ID`
- `GeneEffect`

使用：

```text
cell_idx -> gene_idx
```

关键边属性：

```text
GeneEffect
```

`GeneEffect` 是**带符号的数值**。

不要：

- 取绝对值
- 删除正负号
- 静默将其二值化

CRISPR GeneEffect 应作为 Cell 与 Gene 之间的关系信息。

---

## 7.3 Gene → Gene

文件：

```text
data/heterograph_GDSC/edges/03_gene_gene_STRING_PPI_bidirectional_edges.csv
```

有向边数量：

```text
14,156
```

字段：

- `gene1_idx`
- `Gene1_Entrez_ID`
- `gene2_idx`
- `Gene2_Entrez_ID`
- `combined_score`
- `PPI_score`

使用：

```text
gene1_idx -> gene2_idx
```

该文件已经是**双向边文件**。

因此不要再次无条件复制反向边。

只有在检查原代码后，确认当前 loader 明确要求“单向输入后内部转无向”时，才可以做对应处理。

关键边属性：

```text
PPI_score
```

---

# 8. 当前图统计总结

节点：

```text
Drug:
165
feature = Morgan 1024

Cell:
617
feature = RNA 1410

Gene:
1157
feature = ESM2 1280
```

关系：

```text
Drug --pAffinity--> Gene
602 条边

Cell --GeneEffect--> Gene
703,597 条边

Gene --PPI_score--> Gene
14,156 条有向边
```

当前图中没有完全孤立的节点。

---

# 9. 最关键的 label / graph 分离原则

`LN_IC50` 是监督学习目标。

**Drug–Cell response 绝对不能作为消息传递关系插入异构图。**

正确设计：

```text
异构图输入：

Drug --Ki/pAffinity--> Gene
Cell --CRISPR--------> Gene
Gene --PPI-----------> Gene
```

监督目标：

```text
Drug_i + Cell_j -> LN_IC50_ij
```

禁止构建：

```text
Drug --LN_IC50--> Cell
```

并把它作为消息传播边。

否则会造成 response label leakage。

因此必须始终保持：

```text
生物异构图
```

与：

```text
response 监督数据
```

在概念和代码结构上分离。

---

# 10. 模型修改策略

**不要替换整个 GCNPath 训练系统。**

优先修改或增强：

```text
产生 Drug embedding 和 Cell embedding 的部分
```

原始结构可抽象为：

```text
Original Drug Encoder ----> drug embedding ----\
                                                 \
                                                  prediction module
                                                 /
Original Cell Encoder ----> cell embedding ----/
```

新的 heterograph 模型应大致变为：

```text
Drug Morgan 1024 ----\
                      \
Cell RNA 1410 ---------> Drug-Cell-Gene Heterograph Encoder
                      /
Gene ESM2 1280 -------/

Drug-Gene pAffinity
Cell-Gene GeneEffect
Gene-Gene PPI_score

            |
            v

heterograph drug embeddings
heterograph cell embeddings

            |
            v

尽可能复用原始 GCNPath downstream prediction / fusion 逻辑

            |
            v

predicted LN_IC50
```

下游预测逻辑必须尽量复用原始 GCNPath。

不要重新设计 regression 任务。

---

# 11. 节点输入投影要求

三种节点的原始维度不同：

```text
Drug: 1024
Cell: 1410
Gene: 1280
```

因此需要使用**节点类型特异的输入投影层**，例如：

```text
Drug 1024 -> hidden_dim
Cell 1410 -> hidden_dim
Gene 1280 -> hidden_dim
```

`hidden_dim` 应优先选择能够兼容原始 GCNPath 下游模块的维度。

不要为了异构图而无必要地重写 prediction head。

---

# 12. 关系特异的消息传递

三类关系的生物意义不同，而且边数量差异巨大：

```text
Drug-Gene:       602
Cell-Gene:   703,597
Gene-Gene:    14,156
```

CRISPR 关系远比 Drug-Gene 稠密。

因此：

**不要把全部边合并成一个同质图 adjacency。**

应使用关系特异的变换或 message function，例如：

```text
Drug -> Gene : 一套 relation-specific 参数
Cell -> Gene : 一套 relation-specific 参数
Gene -> Gene : 一套 relation-specific 参数
```

根据模型需要，可以显式加入反向信息流：

```text
Gene -> Drug
Gene -> Cell
```

但反向关系必须明确建模。

不要把所有关系当作同一种关系处理。

---

# 13. 边属性不能被静默丢弃

以下边属性属于本实验输入设计的一部分：

```text
Drug-Gene:
pAffinity

Cell-Gene:
GeneEffect

Gene-Gene:
PPI_score
```

异构图模型应实际利用这些数值进行消息传播。

除非有明确且记录在案的实现原因，否则不要静默把图变成 0/1 二值邻接矩阵。

特别注意：

```text
GeneEffect 可以为负数
```

不要直接把负的 GeneEffect 当作普通正邻接权重传入不支持 signed edge weight 的算子。

如果所选 GNN operator 不能安全使用带符号 scalar edge weight，应实现：

- relation-specific edge encoder
- edge gating
- message modulation

等最小方案。

例如：

```text
先对节点消息做 relation-specific transformation
再使用 edge scalar 对 message 进行可学习调制
```

要求：

- 保留 GeneEffect 的符号
- 不丢失 pAffinity
- 不丢失 PPI_score
- 实现尽量简单
- 技术理由清晰

---

# 14. 监督样本中的 Drug / Cell ID 映射

Response 表中使用：

```text
Drug
Cell
```

必须通过映射文件获得 graph node index：

```text
Drug
  -> drug_node_mapping.csv
  -> drug_idx
```

```text
Cell
  -> cell_node_mapping.csv
  -> cell_idx
```

禁止：

- 依赖 IC50 文件中的行号
- 假设 GDSC Drug ID 连续
- 假设 Cell ID 本身就是整数图索引

以下 mapping CSV 是唯一权威来源：

```text
drug_node_mapping.csv
cell_node_mapping.csv
gene_node_mapping.csv
```

---

# 15. Batch 行为

保留原始 response-pair 训练语义。

每个 batch 仍然应该概念上包含：

```text
Drug
Cell
LN_IC50
```

例如：

```text
Drug_i, Cell_j, LN_IC50_ij
Drug_k, Cell_m, LN_IC50_km
...
```

异构图 encoder 可以先计算图节点表示，然后每个 batch 根据：

```text
drug_idx
cell_idx
```

提取：

```text
z_drug[drug_idx]
z_cell[cell_idx]
```

再送入原始预测模块。

**不要把监督任务改造成：**

- node classification
- edge classification

输出仍然是：

```text
标量 LN_IC50 回归
```

---

# 16. Baseline 与 Heterograph 的实验条件

必须存在两个可以直接运行的模型条件。

---

## Condition A：baseline

- response：
  `IC50_GDSC_异构图.txt`
- 原始 GCNPath 模型架构
- 原始训练逻辑

---

## Condition B：heterograph

- 与 baseline 完全相同的 response：
  `IC50_GDSC_异构图.txt`
- 完全相同的 split
- 新异构图 encoder / model
- 原始训练 / evaluation 逻辑

唯一希望存在的差异：

```text
模型与输入表示
```

概念上的公平比较应满足：

```text
same response file
same fold
same seed
same train rows
same validation rows
same test rows
same labels
same loss
same optimizer
same epochs
same metrics

        |
        +---------------------------+
        |                           |
 Original GCNPath             Heterograph GCNPath
```

---

# 17. Split 一致性验证

每次实验都需要验证 baseline 与 heterograph 的 split membership 完全一致。

至少比较：

- train 样本数
- validation 样本数
- test 样本数
- train 唯一药物数
- validation 唯一药物数
- test 唯一药物数
- train 唯一细胞系数
- validation 唯一细胞系数
- test 唯一细胞系数

对于相同的：

- split type
- fold
- seed

这些统计必须在 baseline 与 heterograph 之间一致。

优先增加：

```text
assert / verification
```

而不是重新写一套 split。

不要为了验证方便而改变原有 split 语义。

---

# 18. 原则上禁止修改的部分

除非为了兼容性而绝对必要，否则不要修改：

- train / validation / test split algorithm
- fold generation
- Drug-Blind definition
- Cell-Blind definition
- 其他原有 blind split 定义
- loss function
- optimizer
- scheduler
- learning rate policy
- epoch count
- early stopping
- evaluation metrics
- metric implementation
- checkpoint policy
- prediction / result 输出格式
- seed logic

如果发现必须修改数据输入和模型结构之外的部分：

**先说明为什么必须修改，再实施。**

---

# 19. 不允许修改 LN_IC50

Response 文件已经包含 GCNPath 当前使用的最终 target。

不要对 `LN_IC50` 做：

- 新的 normalization
- 按 drug mean 中心化
- residualization
- rank transform
- clipping
- aggregation
- 不同方式的 deduplication
- 单位变换

预测目标必须保持当前 `LN_IC50` 原值。

本实验的目的就是：

```text
在不改变预测目标的前提下，
单独验证异构图表示是否提升性能。
```

---

# 20. 当前 response 文件的生成方式

当前：

```text
data/heterograph_GDSC/response/IC50_GDSC_异构图.txt
```

来源于原始 GCNPath：

```text
data/original/ic50_data/IC50_GDSC.txt
```

只保留满足以下条件的样本：

```text
Drug 存在于 drug_node_mapping.csv

AND

Cell 存在于 cell_node_mapping.csv
```

没有对 `LN_IC50` 做额外变换。

原始 GCNPath 数据：

```text
373,681 samples
432 drugs
978 cells
```

筛选后的 heterograph-compatible response：

```text
90,005 samples
165 drugs
593 cells
```

所有 response Drug / Cell 均通过 graph mapping 验证。

---

# 21. 已有数据验证脚本

已有验证脚本：

```text
/homeb/lihuanhuan/20260714/260918/GCNPath2026/check_heterograph_data.py
```

已经确认：

```text
Drug mapping: 165
Drug feature: 165 x 1024

Cell mapping: 617
Cell feature: 617 x 1410

Gene mapping: 1157
Gene feature: 1157 x 1280

Drug-Gene edges: 602
Cell-Gene edges: 703597
Gene-Gene edges: 14156

Response:
90005 samples
165 drugs
593 cells

Response drugs missing from graph: 0
Response cells missing from graph: 0

NaN in Drug feature: 0
NaN in Cell feature: 0
NaN in Gene feature: 0
```

在调试模型之前，应优先运行这个验证脚本。

---

# 22. 实现优先级

必须按照下面顺序推进。

---

## Step 1 —— 先检查原始代码

编辑之前，先准确定位：

- IC50 文件路径在哪里选择
- Dataset / DataLoader 在哪里定义
- train / validation / test split 在哪里执行
- Drug 输入在哪里读取
- Cell 输入在哪里读取
- 原始模型 class 在哪里定义
- Drug embedding 在哪里产生
- Cell embedding 在哪里产生
- Drug / Cell embedding 在哪里进入最终 predictor

**不要一开始就重写 `train.py`。**

---

## Step 2 —— 修改默认 response 输入

把默认 GDSC response 文件改为：

```text
data/heterograph_GDSC/response/IC50_GDSC_异构图.txt
```

Baseline 和 heterograph 模式都必须使用它。

不要修改 split 逻辑。

首先确认：

```text
原始 baseline
```

可以在这 90,005 条 response 数据上正常训练和评估。

这是非常重要的对照实验。

---

## Step 3 —— 增加 heterograph loader

新增最小必要的数据加载模块，用于加载：

- node mappings
- node features
- edge indices
- edge attributes

优先新增独立模块。

不要把几百行 heterograph loading 逻辑全部塞进 `train.py`。

Loader 应返回结构清晰的数据。

例如概念上：

```python
x_dict = {
    "drug": ...,
    "cell": ...,
    "gene": ...,
}

edge_index_dict = {
    ("drug", "binds", "gene"): ...,
    ("cell", "crispr", "gene"): ...,
    ("gene", "ppi", "gene"): ...,
}

edge_attr_dict = {
    ("drug", "binds", "gene"): pAffinity,
    ("cell", "crispr", "gene"): GeneEffect,
    ("gene", "ppi", "gene"): PPI_score,
}
```

具体实现应尽量使用当前 GCNPath 已有依赖和代码风格。

---

## Step 4 —— 增加 heterograph encoder

实现一个新的模型 variant，使其从 Drug–Cell–Gene 异构图中产生：

- Drug embedding
- Cell embedding

然后尽可能复用原始 GCNPath downstream prediction module。

不要对 encoder 之外的架构做不必要修改。

---

## Step 5 —— 增加模型选择开关

需要能够简单切换：

```text
baseline
```

和：

```text
heterograph
```

同时其他实验参数保持完全一致。

不要复制出两套独立训练脚本，导致：

- split 重复实现
- training loop 重复
- metrics 重复
- seed 行为不一致

优先使用：

```text
同一训练 pipeline + model/input switch
```

---

## Step 6 —— 验证公平比较

正式训练前，必须确认 baseline 与 heterograph：

```text
response file == identical
split mode    == identical
seed          == identical
fold          == identical
train set     == identical
valid set     == identical
test set      == identical
```

只有通过这些检查后，再进行性能比较。

---

# 23. 当前实验假设

本阶段将异构图视为：

```text
固定的外部生物学 side information
```

图本身**不包含 LN_IC50 response 边**。

本阶段不要为每个 fold 单独重新构建图，除非后续另行明确要求。

当前首要目标是建立：

```text
Original GCNPath
vs
GCNPath + Drug / Cell / Gene biological graph representation
```

并且保证：

```text
完全相同的 response 数据
完全相同的 split
完全相同的训练与评估 pipeline
```

未来可以单独做更严格的 inductive 实验，例如：

```text
held-out Drug 节点在图中完全移除
```

但那属于**另一个实验问题**。

不要把它混进本次第一阶段实现。

---

# 24. 期望运行方式

最终项目至少应该支持两个直接可比较的运行模式。

概念示例：

```bash
# baseline
python train.py ... --model baseline

# heterograph
python train.py ... --model heterograph
```

实际 CLI 参数名称应遵循现有项目风格。

关键要求不是参数名称，而是：

```text
仅切换 model
```

不能改变：

- response file
- split
- seed
- fold
- optimizer
- loss
- metrics

---

# 25. 必须完成的 sanity checks

正式训练前，必须检查：

```text
[1] Response file loaded:
    90,005 rows

[2] Unique response drugs:
    165

[3] Unique response cells:
    593

[4] Graph:
    165 drug nodes
    617 cell nodes
    1157 gene nodes

[5] Drug mapping coverage:
    100%

[6] Cell mapping coverage:
    100%

[7] Node feature shapes:
    Drug 165 x 1024
    Cell 617 x 1410
    Gene 1157 x 1280

[8] Edge counts:
    Drug-Gene 602
    Cell-Gene 703597
    Gene-Gene 14156

[9] LN_IC50 is NOT a graph edge.

[10] baseline and heterograph have identical split membership.
```

任何一个关键不变量不满足，都应立即：

```text
fail early
```

并输出清晰错误信息。

---

# 26. Git 与变更控制要求

原始参考基线：

```text
heterograph-baseline
```

修改完成后检查：

```bash
git diff --name-status heterograph-baseline
git diff --stat heterograph-baseline
git diff heterograph-baseline -- '*.py'
```

理想情况下，diff 应主要集中于：

- 数据输入路径 / heterograph loading
- 模型定义 / model selection

如果发现以下部分被大幅修改，应视为高风险并重新审查：

- training
- splitting
- metrics
- loss

禁止：

- 对整个项目做大范围自动格式化
- 因代码风格而修改无关文件
- 进行与本任务无关的重构

最终 diff 必须：

```text
小
可审查
可解释
```

---

# 27. 编码原则

本任务的目标**不是**新建一个独立项目。

目标是：

> 保留 GCNPath 作为完整实验框架，  
> 替换/扩展输入表示与模型 encoder，  
> 其他实验因素尽可能全部控制不变。

当遇到两种方案：

```text
A. 大范围重写 GCNPath
```

和：

```text
B. 增加一个小型 adapter / loader / encoder
```

优先选择：

```text
B
```

---

# 28. 最终验收标准

只有以下条件全部满足，任务才算完成：

1. 原始 GCNPath baseline 可以使用  
   `IC50_GDSC_异构图.txt` 正常运行。

2. Heterograph 模型可以使用完全相同的  
   `IC50_GDSC_异构图.txt` 正常运行。

3. Baseline 与 heterograph 在相同：
   - seed
   - fold
   - split mode  
   下使用完全相同的数据划分。

4. 原始 split 逻辑保持不变。

5. 原始 loss 保持不变。

6. 原始 optimizer / training logic 保持不变。

7. 原始 metrics 保持不变。

8. `LN_IC50` 不作为 graph edge 输入。

9. Heterograph 模型实际使用：
   - Morgan Drug 特征
   - RNA Cell 特征
   - ESM2 Gene 特征
   - Drug–Gene `pAffinity`
   - Cell–Gene `GeneEffect`
   - Gene–Gene `PPI_score`

10. 165 个 Drug 和 593 个有监督 Cell 全部可以成功映射到 graph node。

11. 相对 `heterograph-baseline` 的最终 Git diff 尽量小，主要集中在：
    - data input
    - model code

12. Baseline 结果和 heterograph 结果可以在同一套 evaluation pipeline 下直接公平比较。

---

# 29. 编码前必须先汇报修改方案

正式修改代码之前，先检查整个仓库，并汇报：

1. 当前哪个文件决定 IC50 路径。
2. 哪个文件负责 response 加载。
3. 哪个文件负责数据划分。
4. 哪个文件定义原始 GCNPath 模型。
5. 哪些 function / class 需要修改。
6. 哪些文件必须完全不动。
7. 预计的最小文件级 diff。

在完成这一步检查之前：

**不要直接开始大规模修改。**

本任务优先级：

```text
最小修改
可控修改
公平实验
可复现
易审查
```

---

# 30. 给 Codex 的最终总指令

将本文件放到项目根目录后，可以直接对 Codex 发送下面这段：

```text
请先完整阅读 CODEX_HANDOFF_HETEROGRAPH_中文版.md，然后检查 heterograph-baseline 与当前 heterograph-gdsc 分支。

第一步不要修改代码。请先定位 GCNPath 当前的：
1. IC50 加载位置；
2. response Dataset/DataLoader；
3. train/validation/test 数据划分；
4. Drug 输入；
5. Cell 输入；
6. 原始模型定义；
7. Drug/Cell embedding 产生位置；
8. 最终 prediction module。

然后先给出“哪些文件需要改、哪些文件绝对不改”的最小修改方案，并给出预计的文件级 diff。

后续严格按照交接文档执行。

核心要求：

Baseline 和 heterograph 必须默认使用同一个：

data/heterograph_GDSC/response/IC50_GDSC_异构图.txt

并且必须共享完全相同的：
- split
- seed
- fold
- train/valid/test 样本
- label
- loss
- optimizer
- scheduler
- epoch
- early stopping
- metrics
- output format

只允许将实验差异集中在：
1. 数据输入表示；
2. 模型结构。

先确保原始 baseline 能在 90,005 条 response 数据上完整跑通，再接入 heterograph 模型。

不要将 LN_IC50 作为 Drug–Cell 图边。
不要重新设计 split。
不要重新设计 loss。
不要重写整个 train.py。
不要新建一套独立训练框架。

优先通过最小 adapter / loader / encoder 修改完成任务。
```

---

# 31. 本阶段实验核心原则总结

一句话总结：

> **只改输入和模型，不改训练实验框架。**

公平比较原则：

```text
相同 response
相同 split
相同 seed
相同 fold
相同训练参数
相同 loss
相同 optimizer
相同 metrics
        |
        v
只比较：
Original GCNPath
vs
Heterograph GCNPath
```

尤其要避免：

```text
baseline 使用 373,681 条样本
heterograph 使用 90,005 条样本
```

否则性能变化无法归因于异构图本身。

因此本次改造必须先完成：

```text
原始 baseline
+
90,005 条异构图兼容 response
```

确认该控制组正常运行后，再加入 heterograph encoder。

这样最终性能差异才能主要归因于：

```text
Drug / Cell / Gene 生物图信息与新的表示学习方式
```

而不是数据规模、split 或训练流程差异。
