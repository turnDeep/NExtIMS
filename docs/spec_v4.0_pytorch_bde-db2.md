# QC-GN2oMS2-EI システム詳細技術仕様書 v4.0

> 歴史資料: BDE/DGL/BonDNetはv1.1.0の実行環境から削除されています。現行の手順はルートREADME、検討の経緯は[研究履歴](research/HISTORY.md)を参照してください。
## PyTorch統一環境・BonDNet BDE-db2デフォルト版

**作成日**: 2025-12-02
**対象システム**: NExtIMS (NIST EI-MS Prediction System)
**基盤アーキテクチャ**: QC-GN2oMS2 (PNNL)
**ハードウェア**: NVIDIA GeForce RTX 5070 Ti (Blackwell sm_120)

---

## 📋 目次

1. [主要変更点（v3.0 → v4.0）](#主要変更点v30--v40)
2. [システム概要](#システム概要)
3. [アーキテクチャ設計](#アーキテクチャ設計)
4. [Phase 0: BDE-db2環境構築](#phase-0-bde-db2環境構築)
5. [Phase 1: データ準備](#phase-1-データ準備)
6. [Phase 2: GNN学習](#phase-2-gnn学習)
7. [Phase 3: 評価](#phase-3-評価)
8. [設定ファイル詳細](#設定ファイル詳細)
9. [開発環境構築](#開発環境構築)
10. [タイムライン](#タイムライン)
11. [参考文献](#参考文献)

---

## 主要変更点（v3.0 → v4.0）

### ❌ 削除された機能

| 削除項目 | 理由 |
|---------|------|
| **xTB GPU計算** | BDEを直接計算できない（Hessian行列のみ）。間接的なエネルギー差分法は10秒/結合と遅く、ラジカル計算の数値不安定性も問題 |
| **ALFABET** | TensorFlow依存。PyTorch統一環境と競合するため不適合 |
| **プラガブルBDEバックエンド** | xTB/ALFABET削除により単一バックエンド（BonDNet）に集約 |

### ✅ 新規追加・変更された機能

| 項目 | 詳細 |
|------|------|
| **BonDNet BDE-db2デフォルト化** | 531,244件のBDEデータで再学習したBonDNetをデフォルトバックエンドに設定 |
| **Pure PyTorch環境** | TensorFlow依存を完全削除。PyTorch 2.10.0+ nightly (cu128) のみ使用 |
| **Phase 0の追加** | BDE-db2ダウンロード→BonDNet再学習をデータ準備前の必須フェーズとして追加 |
| **ハロゲン・硫黄・リン対応** | BDE-db2により10元素（C,H,O,N,F,S,P,Cl,Br,I）をサポート |
| **設定ファイル簡素化** | BonDNet単一バックエンド化により`config.yml`のBDEセクションを簡略化 |

---

## システム概要

### 目的

NIST 17 EI-MSデータベース（約300,000スペクトル）を用いて、**物理化学的に解釈可能なGraph Neural Network**によるEI-MSスペクトル予測システムを構築する。

### 基盤技術

**QC-GN2oMS2**（PNNL, 2024）:
- 論文: "Quantum Chemistry-Informed Graph Neural Network for Mass Spectrum Prediction"
- 特徴: 量子化学計算（BDE）をエッジ特徴量として使用
- 元の対象: MS/MS（タンデム質量分析）
- **本プロジェクトでの適用**: EI-MS（電子イオン化質量分析、70eV固定）

### アーキテクチャ概要図

```
┌─────────────────────────────────────────────────────────────┐
│ Phase 0: BDE-db2環境構築（必須前処理）                        │
├─────────────────────────────────────────────────────────────┤
│ 1. BDE-db2ダウンロード (531,244 reactions)                   │
│ 2. BonDNet再学習 (2-3日, RTX 5070 Ti)                        │
│ 3. 学習済みモデル検証 (MAE < 1.0 kcal/mol目標)               │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase 1: データ準備                                          │
├─────────────────────────────────────────────────────────────┤
│ NIST 17 EI-MS → BonDNet BDE計算 → PyG Graph → HDF5         │
│ (300,000 spectra × 75 min = 5日間)                          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase 2: GNN学習                                             │
├─────────────────────────────────────────────────────────────┤
│ 10-layer GATv2Conv + Residual Connections                   │
│ RTX 5070 Ti (16GB GDDR7) × 約48時間                          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase 3: 評価                                                │
├─────────────────────────────────────────────────────────────┤
│ Cosine Similarity, Top-10 Recall, Physical Interpretability │
└─────────────────────────────────────────────────────────────┘
```

---

## アーキテクチャ設計

### BDE計算バックエンド: BonDNet (BDE-db2再学習版)

#### 選定理由

| 基準 | BonDNet (BDE-db2) | ALFABET | xTB GPU |
|------|------------------|---------|---------|
| **フレームワーク** | PyTorch ✅ | TensorFlow ❌ | Fortran (nvfortran) |
| **速度** | 15ms/分子 ✅ | 5ms/分子 | 1.5秒/分子 ❌ |
| **精度** | MAE 0.51 kcal/mol ✅ | MAE 0.45 kcal/mol | MAE 3-5 kcal/mol ❌ |
| **対応元素** | 10元素 (BDE-db2) ✅ | 6元素 | 全元素 |
| **BDE直接計算** | 可能 ✅ | 可能 | 不可（エネルギー差分法のみ）❌ |
| **環境統一性** | PyTorch統一 ✅ | TF/PyTorch混在 ❌ | 別プロセス起動 |
| **学習コスト** | 2-3日 (初回のみ) | 学習済み | N/A |

#### BDE-db2データセット詳細

**Paton Group BDE-db2**:
- 総データ数: **531,244 BDE値**
- 元素種: C, H, O, N, F, S, P, Cl, Br, I（10元素）
- データソース: B3LYP/6-31G(d) DFT計算
- 論文: "A comprehensive database of bond dissociation enthalpies" (Paton et al.)

**BDNCM（BonDNet公式）との比較**:
| 項目 | BDNCM (公式) | BDE-db2 (本プロジェクト) |
|------|-------------|------------------------|
| データ数 | 64,312 | 531,244 |
| 元素数 | 5 (C,H,O,F,Li) | 10 (C,H,O,N,F,S,P,Cl,Br,I) |
| 用途 | 有機リチウム電池 | 汎用有機化合物 ✅ |
| ハロゲン対応 | Fのみ | Cl, Br, I対応 ✅ |

**NIST 17との適合性**:
- NIST 17の95%以上が10元素内に収まる
- 環状化合物、ヘテロ環化合物の多様性に対応
- ハロゲン化合物（農薬、医薬品）のカバレッジ向上

#### BonDNet再学習の必要性

BonDNet公式モデル（BDNCM学習済み）は以下の理由で不十分:

1. **元素不足**: S, P, Cl, Br, I が未対応
2. **データ分布**: 有機リチウム電池用途に最適化（NIST EI-MSとドメイン乖離）
3. **精度**: BDE-db2での再学習により、NIST分子に対するMAE改善が期待される

**再学習による改善目標**:
- MAE: 0.51 kcal/mol → **0.8 kcal/mol以下**（BDE-db2再学習後）
- カバレッジ: 85% → **95%以上**（10元素対応により）

---

### GNNアーキテクチャ: 10-layer GATv2Conv

#### モデル構成

```python
import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, global_mean_pool

class QCGN2oEI(nn.Module):
    """
    QC-GN2oMS2 Architecture for EI-MS Prediction

    Key changes from original:
    - MS/MS → EI-MS (fragmentation energy: variable → 70eV fixed)
    - Edge features: BDE from BonDNet (BDE-db2 retrained)
    - Output: 1000-bin intensity distribution (m/z 50-1000)
    """

    def __init__(
        self,
        node_dim: int = 128,       # Atom feature dimension
        edge_dim: int = 64,        # Edge feature dimension (includes BDE)
        hidden_dim: int = 256,     # Hidden layer dimension
        num_layers: int = 10,      # GATv2Conv layers
        num_heads: int = 8,        # Attention heads
        output_dim: int = 1000,    # Output spectrum bins
        dropout: float = 0.1
    ):
        super().__init__()

        # Node embedding
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Edge embedding (BDE + bond features)
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 10-layer GATv2Conv with residual connections
        self.gat_layers = nn.ModuleList()
        self.residual_layers = nn.ModuleList()

        for i in range(num_layers):
            # GATv2Conv layer
            self.gat_layers.append(
                GATv2Conv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // num_heads,
                    heads=num_heads,
                    edge_dim=hidden_dim,  # Edge features
                    dropout=dropout,
                    concat=True,          # Concatenate heads
                    residual=True         # PyG 2.6.1+ feature
                )
            )

            # Residual connection projection
            self.residual_layers.append(
                nn.Linear(hidden_dim, hidden_dim)
            )

        # Global pooling + prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1)  # Normalize to intensity distribution
        )

    def forward(self, data):
        """
        Args:
            data: PyG Data object
                - x: Node features [num_nodes, node_dim]
                - edge_index: Graph connectivity [2, num_edges]
                - edge_attr: Edge features (includes BDE) [num_edges, edge_dim]
                - batch: Batch assignment [num_nodes]

        Returns:
            spectrum: Predicted intensity [batch_size, 1000]
        """
        # Encode nodes and edges
        x = self.node_encoder(data.x)
        edge_attr = self.edge_encoder(data.edge_attr)

        # 10-layer GATv2Conv with residual connections
        for gat, residual in zip(self.gat_layers, self.residual_layers):
            x_res = residual(x)  # Residual projection
            x = gat(x, data.edge_index, edge_attr)
            x = x + x_res  # Residual addition
            x = torch.relu(x)

        # Global mean pooling
        x = global_mean_pool(x, data.batch)

        # Predict spectrum
        spectrum = self.prediction_head(x)

        return spectrum
```

#### ノード特徴量（128次元）

| カテゴリ | 次元 | 内容 |
|---------|------|------|
| **原子種** | 10 | C, H, O, N, F, S, P, Cl, Br, I (one-hot) |
| **ハイブリダイゼーション** | 5 | SP, SP2, SP3, SP3D, SP3D2 (one-hot) |
| **形式電荷** | 3 | -1, 0, +1 (one-hot) |
| **芳香族性** | 1 | Binary (aromatic/aliphatic) |
| **環構造** | 1 | Binary (in ring/not in ring) |
| **水素結合数** | 5 | 0, 1, 2, 3, 4+ (one-hot) |
| **次数（degree）** | 6 | 0, 1, 2, 3, 4, 5+ (one-hot) |
| **ラジカル電子** | 3 | 0, 1, 2 (one-hot) |
| **キラリティ** | 3 | None, R, S (one-hot) |
| **部分電荷** | 1 | Gasteiger charge (continuous) |
| **原子量** | 1 | Normalized atomic mass (continuous) |
| **ファン der Waals半径** | 1 | Normalized vdW radius (continuous) |
| **電気陰性度** | 1 | Pauling electronegativity (continuous) |
| **予備** | 87 | 将来の拡張用 |

**実装例**:
```python
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import numpy as np

def get_atom_features(atom: Chem.Atom) -> np.ndarray:
    """Extract 128-dimensional atom features"""

    # Atom type (10-dim one-hot)
    atom_types = ['C', 'H', 'O', 'N', 'F', 'S', 'P', 'Cl', 'Br', 'I']
    atom_type = one_hot(atom.GetSymbol(), atom_types)

    # Hybridization (5-dim one-hot)
    hybridizations = [
        Chem.HybridizationType.SP,
        Chem.HybridizationType.SP2,
        Chem.HybridizationType.SP3,
        Chem.HybridizationType.SP3D,
        Chem.HybridizationType.SP3D2
    ]
    hybrid = one_hot(atom.GetHybridization(), hybridizations)

    # Formal charge (3-dim one-hot)
    charge = one_hot(atom.GetFormalCharge(), [-1, 0, 1])

    # Binary features
    aromatic = [int(atom.GetIsAromatic())]
    in_ring = [int(atom.IsInRing())]

    # Hydrogen count (5-dim one-hot)
    num_h = one_hot(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])

    # Degree (6-dim one-hot)
    degree = one_hot(atom.GetDegree(), [0, 1, 2, 3, 4, 5])

    # Radical electrons (3-dim one-hot)
    radical = one_hot(atom.GetNumRadicalElectrons(), [0, 1, 2])

    # Chirality (3-dim one-hot)
    chiralities = [
        Chem.ChiralType.CHI_UNSPECIFIED,
        Chem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.ChiralType.CHI_TETRAHEDRAL_CCW
    ]
    chirality = one_hot(atom.GetChiralTag(), chiralities)

    # Continuous features
    partial_charge = [atom.GetDoubleProp('_GasteigerCharge') if atom.HasProp('_GasteigerCharge') else 0.0]
    atomic_mass = [atom.GetMass() / 100.0]  # Normalize
    vdw_radius = [Chem.GetPeriodicTable().GetRvdw(atom.GetSymbol()) / 2.0]  # Normalize
    electronegativity = [Chem.GetPeriodicTable().GetElectronegativity(atom.GetSymbol()) / 4.0]  # Normalize

    # Concatenate (total: 10+5+3+1+1+5+6+3+3+1+1+1+1 = 41 dims)
    # Pad to 128 with zeros
    features = np.concatenate([
        atom_type, hybrid, charge, aromatic, in_ring,
        num_h, degree, radical, chirality,
        partial_charge, atomic_mass, vdw_radius, electronegativity
    ])

    padded = np.zeros(128)
    padded[:len(features)] = features

    return padded

def one_hot(value, choices):
    """One-hot encoding with out-of-vocabulary handling"""
    encoding = [0] * len(choices)
    if value in choices:
        encoding[choices.index(value)] = 1
    return encoding
```

#### エッジ特徴量（64次元）

| カテゴリ | 次元 | 内容 |
|---------|------|------|
| **BDE（重要）** | 1 | Bond Dissociation Energy from BonDNet (kcal/mol, normalized) |
| **結合次数** | 4 | Single, Double, Triple, Aromatic (one-hot) |
| **環内結合** | 1 | Binary (in ring/not in ring) |
| **共役** | 1 | Binary (conjugated/not conjugated) |
| **立体化学** | 3 | None, E, Z (one-hot) |
| **回転可能性** | 1 | Binary (rotatable/rigid) |
| **結合距離** | 1 | Normalized bond length (Å) |
| **予備** | 52 | 将来の拡張用 |

**BDE正規化**:
```python
def normalize_bde(bde_kcal_mol: float) -> float:
    """
    Normalize BDE to [0, 1] range

    Typical BDE ranges:
    - C-C single: 85 kcal/mol
    - C=C double: 146 kcal/mol
    - C-H: 105 kcal/mol
    - O-H: 110 kcal/mol
    - Aromatic C-C: 120 kcal/mol

    Range: 50-200 kcal/mol
    """
    return (bde_kcal_mol - 50.0) / 150.0
```

**実装例**:
```python
def get_bond_features(bond: Chem.Bond, bde_value: float) -> np.ndarray:
    """Extract 64-dimensional bond features"""

    # BDE (normalized)
    bde = [normalize_bde(bde_value)]

    # Bond type (4-dim one-hot)
    bond_types = [
        Chem.BondType.SINGLE,
        Chem.BondType.DOUBLE,
        Chem.BondType.TRIPLE,
        Chem.BondType.AROMATIC
    ]
    bond_type = one_hot(bond.GetBondType(), bond_types)

    # Binary features
    in_ring = [int(bond.IsInRing())]
    conjugated = [int(bond.GetIsConjugated())]

    # Stereochemistry (3-dim one-hot)
    stereo = one_hot(bond.GetStereo(), [
        Chem.BondStereo.STEREONONE,
        Chem.BondStereo.STEREOE,
        Chem.BondStereo.STEREOZ
    ])

    # Rotatable
    rotatable = [int(bond.GetBondDir() == Chem.BondDir.NONE and not bond.IsInRing())]

    # Bond length (requires 3D conformer)
    mol = bond.GetOwningMol()
    if mol.GetNumConformers() > 0:
        conf = mol.GetConformer()
        pos_i = conf.GetAtomPosition(bond.GetBeginAtomIdx())
        pos_j = conf.GetAtomPosition(bond.GetEndAtomIdx())
        length = [(pos_i - pos_j).Length() / 2.0]  # Normalize to ~[0, 1]
    else:
        length = [0.75]  # Default typical bond length

    # Concatenate (total: 1+4+1+1+3+1+1 = 12 dims)
    # Pad to 64 with zeros
    features = np.concatenate([
        bde, bond_type, in_ring, conjugated, stereo, rotatable, length
    ])

    padded = np.zeros(64)
    padded[:len(features)] = features

    return padded
```

---

## Phase 0: BDE-db2環境構築

### 概要

BonDNetをBDE-db2データセットで再学習し、NIST 17分子に最適化されたBDEモデルを構築する。

### タイムライン

| ステップ | 推定時間 | 詳細 |
|---------|---------|------|
| **0.1 BDE-db2ダウンロード** | 30分 | GitHub LFS経由でダウンロード（約50GB） |
| **0.2 データ前処理** | 3時間 | BonDNet形式への変換、train/val/test分割 |
| **0.3 BonDNet再学習** | 48-72時間 | RTX 5070 Ti × 531,244サンプル |
| **0.4 モデル検証** | 1時間 | MAE計算、エラー分析 |
| **合計** | **2-3日** | - |

### 0.1 BDE-db2ダウンロード

```bash
#!/bin/bash
# scripts/download_bde_db2.sh

set -e

echo "Downloading BDE-db2 dataset..."

# Create directory
mkdir -p data/external/bde-db2

# Clone BDE-db2 repository
cd data/external/bde-db2
git clone https://github.com/patongroup/BDE-db2.git .

# Verify download
if [ -f "bde_data.csv" ]; then
    echo "✅ BDE-db2 downloaded successfully"
    wc -l bde_data.csv
else
    echo "❌ Download failed"
    exit 1
fi
```

**データ形式**:
```csv
smiles,bond_index_1,bond_index_2,bde_kcal_mol,method,basis_set
CCO,0,1,85.3,B3LYP,6-31G(d)
CCO,1,2,104.2,B3LYP,6-31G(d)
c1ccccc1,0,1,119.8,B3LYP,6-31G(d)
...
```

### 0.2 データ前処理

```python
# scripts/preprocess_bde_db2.py
"""
Convert BDE-db2 to BonDNet training format
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from bondnet.data.dataset import prepare_reaction_graphs
import pickle
from pathlib import Path

def convert_bde_db2_to_bondnet(
    input_csv: str = "data/external/bde-db2/bde_data.csv",
    output_dir: str = "data/processed/bondnet_bde_db2"
):
    """
    Convert BDE-db2 CSV to BonDNet graph format

    BonDNet expects reaction SMILES: reactant>>product
    For BDE: parent_molecule >> radical1.radical2
    """

    print("Loading BDE-db2...")
    df = pd.read_csv(input_csv)
    print(f"Total entries: {len(df)}")

    # Filter valid SMILES
    df = df[df['smiles'].apply(lambda x: Chem.MolFromSmiles(x) is not None)]
    print(f"Valid SMILES: {len(df)}")

    # Create reaction SMILES for BonDNet
    reactions = []
    bde_values = []

    for idx, row in df.iterrows():
        smiles = row['smiles']
        bond_idx_1 = int(row['bond_index_1'])
        bond_idx_2 = int(row['bond_index_2'])
        bde = float(row['bde_kcal_mol'])

        # Create fragmented SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        # Fragment at specified bond
        frag = Chem.FragmentOnBonds(
            mol,
            [mol.GetBondBetweenAtoms(bond_idx_1, bond_idx_2).GetIdx()],
            addDummies=False
        )

        # Get fragment SMILES
        frags = Chem.GetMolFrags(frag, asMols=True)
        if len(frags) != 2:
            continue

        frag1_smi = Chem.MolToSmiles(frags[0])
        frag2_smi = Chem.MolToSmiles(frags[1])

        # Reaction format: parent >> radical1.radical2
        reaction_smi = f"{smiles}>>{frag1_smi}.{frag2_smi}"

        reactions.append(reaction_smi)
        bde_values.append(bde)

        if idx % 10000 == 0:
            print(f"Processed {idx}/{len(df)}")

    print(f"Total reactions: {len(reactions)}")

    # Train/Val/Test split (80/10/10)
    np.random.seed(42)
    indices = np.random.permutation(len(reactions))

    n_train = int(0.8 * len(reactions))
    n_val = int(0.1 * len(reactions))

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:]

    # Create BonDNet datasets
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for split, idx in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
        split_reactions = [reactions[i] for i in idx]
        split_bde = [bde_values[i] for i in idx]

        # Save as BonDNet format
        dataset = prepare_reaction_graphs(split_reactions, split_bde)

        with open(output_path / f"{split}.pkl", 'wb') as f:
            pickle.dump(dataset, f)

        print(f"{split}: {len(split_reactions)} reactions")

    print("✅ BDE-db2 conversion complete")

if __name__ == "__main__":
    convert_bde_db2_to_bondnet()
```

**実行**:
```bash
python scripts/preprocess_bde_db2.py
```

**出力**:
```
data/processed/bondnet_bde_db2/
├── train.pkl  (424,995 reactions)
├── val.pkl    (53,125 reactions)
└── test.pkl   (53,124 reactions)
```

### 0.3 BonDNet再学習

```python
# scripts/train_bondnet_bde_db2.py
"""
Train BonDNet on BDE-db2 dataset
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from bondnet.model.gated_reaction_network import GatedGCNReactionNetwork
from bondnet.data.dataloader import DataLoaderReaction
import yaml
from pathlib import Path
import wandb

def train_bondnet_bde_db2(
    config_path: str = "config/bondnet_training.yml"
):
    """
    Train BonDNet on BDE-db2 dataset

    Expected training time: 48-72 hours on RTX 5070 Ti
    """

    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Initialize wandb
    wandb.init(
        project="bondnet-bde-db2",
        config=config
    )

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Data loaders
    train_loader = DataLoaderReaction(
        dataset_path="data/processed/bondnet_bde_db2/train.pkl",
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4
    )

    val_loader = DataLoaderReaction(
        dataset_path="data/processed/bondnet_bde_db2/val.pkl",
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4
    )

    # Model
    model = GatedGCNReactionNetwork(
        in_feats=config['model']['in_feats'],
        embedding_size=config['model']['embedding_size'],
        gated_num_layers=config['model']['gated_num_layers'],
        gated_hidden_size=config['model']['gated_hidden_size'],
        gated_num_fc_layers=config['model']['gated_num_fc_layers'],
        gated_graph_norm=config['model']['gated_graph_norm'],
        gated_batch_norm=config['model']['gated_batch_norm'],
        gated_activation=config['model']['gated_activation'],
        gated_residual=config['model']['gated_residual'],
        gated_dropout=config['model']['gated_dropout'],
        num_lstm_iters=config['model']['num_lstm_iters'],
        num_lstm_layers=config['model']['num_lstm_layers'],
        set2set_ntypes_direct=config['model']['set2set_ntypes_direct'],
        fc_num_layers=config['model']['fc_num_layers'],
        fc_hidden_size=config['model']['fc_hidden_size'],
        fc_batch_norm=config['model']['fc_batch_norm'],
        fc_activation=config['model']['fc_activation'],
        fc_dropout=config['model']['fc_dropout'],
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Optimizer & Scheduler
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        verbose=True
    )

    # Loss function
    criterion = nn.L1Loss()  # MAE loss

    # Training loop
    best_val_mae = float('inf')

    for epoch in range(config['training']['num_epochs']):
        # Train
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            batch = batch.to(device)

            optimizer.zero_grad()
            pred = model(batch)
            loss = criterion(pred, batch.y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch)
                loss = criterion(pred, batch.y)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Logging
        wandb.log({
            'epoch': epoch,
            'train_mae': train_loss,
            'val_mae': val_loss,
            'learning_rate': optimizer.param_groups[0]['lr']
        })

        print(f"Epoch {epoch+1}/{config['training']['num_epochs']}: "
              f"Train MAE={train_loss:.4f}, Val MAE={val_loss:.4f}")

        # Save best model
        if val_loss < best_val_mae:
            best_val_mae = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mae': val_loss,
            }, "models/bondnet_bde_db2_best.pth")
            print(f"✅ Best model saved (Val MAE: {val_loss:.4f})")

    print(f"Training complete. Best Val MAE: {best_val_mae:.4f}")

if __name__ == "__main__":
    train_bondnet_bde_db2()
```

**設定ファイル**:
```yaml
# config/bondnet_training.yml

model:
  in_feats: 54  # Atom feature dimension
  embedding_size: 128
  gated_num_layers: 4
  gated_hidden_size: [128, 128, 64]
  gated_num_fc_layers: 2
  gated_graph_norm: True
  gated_batch_norm: True
  gated_activation: "ReLU"
  gated_residual: True
  gated_dropout: 0.0
  num_lstm_iters: 6
  num_lstm_layers: 3
  set2set_ntypes_direct: ["atom", "bond", "global"]
  fc_num_layers: 2
  fc_hidden_size: [64, 32]
  fc_batch_norm: False
  fc_activation: "ReLU"
  fc_dropout: 0.0

training:
  num_epochs: 200
  batch_size: 256  # RTX 5070 Ti (16GB) optimal batch size
  learning_rate: 0.001
  weight_decay: 0.0
  early_stopping_patience: 30
```

**実行**:
```bash
python scripts/train_bondnet_bde_db2.py
```

**予想出力（48時間後）**:
```
Epoch 200/200: Train MAE=0.612, Val MAE=0.784
✅ Best model saved (Val MAE: 0.784)
Training complete. Best Val MAE: 0.784
```

### 0.4 モデル検証

```python
# scripts/evaluate_bondnet_bde_db2.py
"""
Evaluate trained BonDNet model
"""

import torch
from bondnet.model.gated_reaction_network import GatedGCNReactionNetwork
from bondnet.data.dataloader import DataLoaderReaction
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_bondnet(
    model_path: str = "models/bondnet_bde_db2_best.pth",
    test_data: str = "data/processed/bondnet_bde_db2/test.pkl"
):
    """Evaluate BonDNet on test set"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    checkpoint = torch.load(model_path)
    model = GatedGCNReactionNetwork(...)  # Same config as training
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Test loader
    test_loader = DataLoaderReaction(
        dataset_path=test_data,
        batch_size=256,
        shuffle=False
    )

    # Inference
    predictions = []
    targets = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(batch)
            predictions.extend(pred.cpu().numpy())
            targets.extend(batch.y.cpu().numpy())

    predictions = np.array(predictions)
    targets = np.array(targets)

    # Metrics
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    r2 = 1 - np.sum((predictions - targets) ** 2) / np.sum((targets - np.mean(targets)) ** 2)

    print("=" * 60)
    print("BonDNet BDE-db2 Test Results")
    print("=" * 60)
    print(f"MAE:  {mae:.4f} kcal/mol")
    print(f"RMSE: {rmse:.4f} kcal/mol")
    print(f"R²:   {r2:.4f}")
    print("=" * 60)

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Parity plot
    axes[0].scatter(targets, predictions, alpha=0.3, s=1)
    axes[0].plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--')
    axes[0].set_xlabel("True BDE (kcal/mol)")
    axes[0].set_ylabel("Predicted BDE (kcal/mol)")
    axes[0].set_title(f"Parity Plot (R²={r2:.4f})")

    # Error distribution
    errors = predictions - targets
    axes[1].hist(errors, bins=50, alpha=0.7)
    axes[1].axvline(0, color='r', linestyle='--')
    axes[1].set_xlabel("Prediction Error (kcal/mol)")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"Error Distribution (MAE={mae:.4f})")

    plt.tight_layout()
    plt.savefig("results/bondnet_bde_db2_evaluation.png", dpi=300)
    print("✅ Evaluation plot saved to results/bondnet_bde_db2_evaluation.png")

if __name__ == "__main__":
    evaluate_bondnet()
```

**目標精度**:
- MAE < 0.8 kcal/mol
- RMSE < 1.5 kcal/mol
- R² > 0.95

---

## Phase 1: データ準備

### 1.1 NIST 17データ読み込み

```python
# src/data/nist_loader.py
"""
NIST 17 EI-MS Data Loader
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from pathlib import Path
from typing import List, Dict, Tuple

class NIST17Loader:
    """
    NIST 17 EI-MS Database Loader

    Expected format: MSP file with NIST spectra
    """

    def __init__(self, nist_path: str = "data/external/nist17/mainlib"):
        self.nist_path = Path(nist_path)

    def parse_msp(self, msp_file: str) -> List[Dict]:
        """
        Parse NIST MSP file

        Returns:
            List of dicts with keys: name, smiles, spectrum
        """
        spectra = []
        current_spectrum = {}
        current_peaks = []

        with open(msp_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()

                if line.startswith("Name:"):
                    if current_spectrum:
                        current_spectrum['spectrum'] = current_peaks
                        spectra.append(current_spectrum)
                    current_spectrum = {'name': line.split(":", 1)[1].strip()}
                    current_peaks = []

                elif line.startswith("SMILES:"):
                    current_spectrum['smiles'] = line.split(":", 1)[1].strip()

                elif line.startswith("Num Peaks:"):
                    num_peaks = int(line.split(":")[1].strip())

                elif line and line[0].isdigit():
                    # Peak data: "m/z intensity; m/z intensity; ..."
                    for peak in line.split(";"):
                        peak = peak.strip()
                        if peak:
                            mz, intensity = peak.split()
                            current_peaks.append((int(mz), int(intensity)))

        # Add last spectrum
        if current_spectrum:
            current_spectrum['spectrum'] = current_peaks
            spectra.append(current_spectrum)

        return spectra

    def load_all_spectra(self) -> pd.DataFrame:
        """Load all NIST 17 spectra into DataFrame"""

        all_spectra = []

        # NIST 17 has multiple MSP files
        msp_files = list(self.nist_path.glob("*.msp"))

        for msp_file in msp_files:
            print(f"Loading {msp_file.name}...")
            spectra = self.parse_msp(msp_file)
            all_spectra.extend(spectra)

        df = pd.DataFrame(all_spectra)

        # Filter valid SMILES
        df = df[df['smiles'].notna()]
        df = df[df['smiles'].apply(lambda x: Chem.MolFromSmiles(x) is not None)]

        print(f"Total spectra: {len(df)}")

        return df

    def spectrum_to_array(self, peaks: List[Tuple[int, int]],
                          mz_range: Tuple[int, int] = (50, 1000)) -> np.ndarray:
        """
        Convert peak list to fixed-size array

        Args:
            peaks: List of (m/z, intensity) tuples
            mz_range: (min_mz, max_mz)

        Returns:
            spectrum_array: [950] array for m/z 50-1000
        """
        min_mz, max_mz = mz_range
        spectrum = np.zeros(max_mz - min_mz)

        for mz, intensity in peaks:
            if min_mz <= mz < max_mz:
                spectrum[mz - min_mz] = intensity

        # Normalize to [0, 1]
        if spectrum.max() > 0:
            spectrum = spectrum / spectrum.max()

        return spectrum
```

### 1.2 BDE前計算（BonDNet BDE-db2）

```python
# src/data/bde_calculator.py
"""
BDE Calculation using BonDNet (BDE-db2 retrained model)
"""

import torch
import numpy as np
from rdkit import Chem
from bondnet.model.gated_reaction_network import GatedGCNReactionNetwork
from bondnet.prediction.predictor import predict_single_molecule
from typing import Dict
import h5py
from pathlib import Path

class BDECalculator:
    """
    Bond Dissociation Energy Calculator

    Uses BonDNet model retrained on BDE-db2 dataset
    """

    def __init__(
        self,
        model_path: str = "models/bondnet_bde_db2_best.pth",
        device: str = "cuda"
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load BonDNet model
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = GatedGCNReactionNetwork(...)  # Config from training
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        print(f"✅ BonDNet model loaded from {model_path}")
        print(f"   Using device: {self.device}")

    def calculate_bde(self, smiles: str, charge: int = 0) -> Dict[int, float]:
        """
        Calculate BDE for all bonds in a molecule

        Args:
            smiles: SMILES string
            charge: Molecular charge (default: 0)

        Returns:
            bde_dict: {bond_idx: bde_value (kcal/mol)}
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}

        bde_dict = {}

        with torch.no_grad():
            for bond in mol.GetBonds():
                bond_idx = bond.GetIdx()
                atom_i = bond.GetBeginAtomIdx()
                atom_j = bond.GetEndAtomIdx()

                # Create reaction SMILES for BonDNet
                frag = Chem.FragmentOnBonds(mol, [bond_idx], addDummies=False)
                frags = Chem.GetMolFrags(frag, asMols=True)

                if len(frags) != 2:
                    # Cyclic bond: use default estimate
                    if bond.GetIsAromatic():
                        bde_dict[bond_idx] = 120.0  # Aromatic C-C
                    else:
                        bde_dict[bond_idx] = 85.0   # Aliphatic C-C in ring
                    continue

                frag1_smi = Chem.MolToSmiles(frags[0])
                frag2_smi = Chem.MolToSmiles(frags[1])
                reaction_smi = f"{smiles}>>{frag1_smi}.{frag2_smi}"

                # Predict BDE using BonDNet
                try:
                    bde_pred = predict_single_molecule(
                        self.model,
                        reaction_smi,
                        charge=charge,
                        device=self.device
                    )
                    bde_dict[bond_idx] = float(bde_pred)
                except Exception as e:
                    # Fallback to rule-based estimate
                    bond_type = bond.GetBondType()
                    if bond_type == Chem.BondType.SINGLE:
                        bde_dict[bond_idx] = 85.0
                    elif bond_type == Chem.BondType.DOUBLE:
                        bde_dict[bond_idx] = 146.0
                    elif bond_type == Chem.BondType.TRIPLE:
                        bde_dict[bond_idx] = 200.0
                    else:
                        bde_dict[bond_idx] = 100.0

        return bde_dict

    def batch_calculate(
        self,
        smiles_list: list,
        output_hdf5: str = "data/processed/bde_cache.h5",
        batch_size: int = 64
    ):
        """
        Batch BDE calculation with HDF5 caching

        Args:
            smiles_list: List of SMILES strings
            output_hdf5: Output HDF5 file path
            batch_size: Batch size for GPU inference
        """
        from tqdm import tqdm

        Path(output_hdf5).parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(output_hdf5, 'w') as f:
            for i, smiles in enumerate(tqdm(smiles_list, desc="Calculating BDE")):
                bde_dict = self.calculate_bde(smiles)

                # Store in HDF5
                grp = f.create_group(str(i))
                grp.attrs['smiles'] = smiles

                for bond_idx, bde_value in bde_dict.items():
                    grp.create_dataset(str(bond_idx), data=bde_value)

                if (i + 1) % 1000 == 0:
                    f.flush()

        print(f"✅ BDE calculation complete: {output_hdf5}")
```

**実行時間見積もり**:
```
300,000 molecules × 15ms/molecule = 4,500 seconds = 75 minutes
```

### 1.3 PyG Graph生成

```python
# src/data/graph_generator.py
"""
PyTorch Geometric Graph Generator
"""

import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import numpy as np
from typing import Dict, List
import h5py

class GraphGenerator:
    """Generate PyTorch Geometric graphs with BDE edge features"""

    def __init__(self, bde_cache_path: str = "data/processed/bde_cache.h5"):
        self.bde_cache = h5py.File(bde_cache_path, 'r')

    def smiles_to_graph(
        self,
        smiles: str,
        spectrum: np.ndarray,
        molecule_idx: int
    ) -> Data:
        """
        Convert SMILES to PyG Data object

        Args:
            smiles: SMILES string
            spectrum: Target spectrum [1000]
            molecule_idx: Index for BDE cache lookup

        Returns:
            PyG Data object
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Add hydrogens for complete graph
        mol = Chem.AddHs(mol)

        # Generate 3D conformer for bond lengths
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)

        # Compute Gasteiger charges
        AllChem.ComputeGasteigerCharges(mol)

        # Get BDE values from cache
        bde_dict = {}
        if str(molecule_idx) in self.bde_cache:
            grp = self.bde_cache[str(molecule_idx)]
            for bond_idx in grp.keys():
                bde_dict[int(bond_idx)] = float(grp[bond_idx][()])

        # Node features
        node_features = []
        for atom in mol.GetAtoms():
            node_features.append(self.get_atom_features(atom))

        x = torch.tensor(node_features, dtype=torch.float)

        # Edge features
        edge_index = []
        edge_attr = []

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bond_idx = bond.GetIdx()

            # Get BDE value
            bde_value = bde_dict.get(bond_idx, 100.0)  # Default if not in cache

            # Bidirectional edges
            edge_index.append([i, j])
            edge_index.append([j, i])

            bond_features = self.get_bond_features(bond, bde_value)
            edge_attr.append(bond_features)
            edge_attr.append(bond_features)  # Same features for both directions

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        # Target spectrum
        y = torch.tensor(spectrum, dtype=torch.float)

        # Create PyG Data
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            smiles=smiles
        )

        return data

    def get_atom_features(self, atom: Chem.Atom) -> List[float]:
        """Extract 128-dimensional atom features"""
        # (Implementation same as in architecture section)
        pass

    def get_bond_features(self, bond: Chem.Bond, bde_value: float) -> List[float]:
        """Extract 64-dimensional bond features"""
        # (Implementation same as in architecture section)
        pass
```

### 1.4 HDF5データセット保存

```python
# scripts/prepare_dataset.py
"""
Prepare complete dataset with BDE and PyG graphs
"""

from src.data.nist_loader import NIST17Loader
from src.data.bde_calculator import BDECalculator
from src.data.graph_generator import GraphGenerator
import h5py
from pathlib import Path
from tqdm import tqdm

def prepare_full_dataset():
    """
    Full pipeline: NIST → BDE → PyG Graph → HDF5
    """

    # Step 1: Load NIST 17
    print("Step 1: Loading NIST 17 data...")
    nist_loader = NIST17Loader("data/external/nist17/mainlib")
    df = nist_loader.load_all_spectra()
    print(f"Loaded {len(df)} spectra")

    # Step 2: Calculate BDE for all molecules
    print("\nStep 2: Calculating BDE (BonDNet BDE-db2)...")
    bde_calc = BDECalculator(
        model_path="models/bondnet_bde_db2_best.pth",
        device="cuda"
    )
    bde_calc.batch_calculate(
        smiles_list=df['smiles'].tolist(),
        output_hdf5="data/processed/bde_cache.h5"
    )

    # Step 3: Generate PyG graphs
    print("\nStep 3: Generating PyG graphs...")
    graph_gen = GraphGenerator("data/processed/bde_cache.h5")

    graphs = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        spectrum_array = nist_loader.spectrum_to_array(row['spectrum'])
        graph = graph_gen.smiles_to_graph(
            smiles=row['smiles'],
            spectrum=spectrum_array,
            molecule_idx=idx
        )
        if graph is not None:
            graphs.append(graph)

    # Step 4: Train/Val/Test split
    print("\nStep 4: Splitting dataset...")
    from sklearn.model_selection import train_test_split

    train_graphs, temp_graphs = train_test_split(graphs, test_size=0.2, random_state=42)
    val_graphs, test_graphs = train_test_split(temp_graphs, test_size=0.5, random_state=42)

    print(f"Train: {len(train_graphs)}, Val: {len(val_graphs)}, Test: {len(test_graphs)}")

    # Step 5: Save to HDF5
    print("\nStep 5: Saving to HDF5...")
    import torch

    for split, split_graphs in [('train', train_graphs), ('val', val_graphs), ('test', test_graphs)]:
        output_path = f"data/processed/nist17_{split}.pt"
        torch.save(split_graphs, output_path)
        print(f"✅ Saved {split}: {output_path}")

    print("\n✅ Dataset preparation complete!")

if __name__ == "__main__":
    prepare_full_dataset()
```

**実行**:
```bash
python scripts/prepare_dataset.py
```

**推定時間**:
- NIST読み込み: 30分
- BDE計算: 75分
- PyG Graph生成: 60分
- **合計: 約3時間**

---

## Phase 2: GNN学習

### 2.1 学習スクリプト

```python
# scripts/train_gnn.py
"""
Train QC-GN2oEI model
"""

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from src.models.qcgn2oei import QCGN2oEI
import wandb
import yaml
from pathlib import Path
from tqdm import tqdm

def cosine_similarity_loss(pred, target):
    """
    Cosine Similarity Loss

    Same as QC-GN2oMS2 paper
    """
    pred_norm = pred / (pred.norm(dim=1, keepdim=True) + 1e-8)
    target_norm = target / (target.norm(dim=1, keepdim=True) + 1e-8)

    cosine_sim = (pred_norm * target_norm).sum(dim=1)

    # Return 1 - cosine_similarity (minimize loss)
    return (1 - cosine_sim).mean()

def train_qcgn2oei(config_path: str = "config/training.yml"):
    """Train QC-GN2oEI model"""

    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Initialize wandb
    wandb.init(project="qcgn2oei", config=config)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data loaders
    train_data = torch.load("data/processed/nist17_train.pt")
    val_data = torch.load("data/processed/nist17_val.pt")

    train_loader = DataLoader(
        train_data,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4
    )

    val_loader = DataLoader(
        val_data,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4
    )

    # Model
    model = QCGN2oEI(
        node_dim=config['model']['node_dim'],
        edge_dim=config['model']['edge_dim'],
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads'],
        output_dim=config['model']['output_dim'],
        dropout=config['model']['dropout']
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Optimizer (RAdam from QC-GN2oMS2)
    optimizer = torch.optim.RAdam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['num_epochs'],
        eta_min=1e-6
    )

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(config['training']['num_epochs']):
        # Train
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            batch = batch.to(device)

            optimizer.zero_grad()
            pred = model(batch)
            loss = cosine_similarity_loss(pred, batch.y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                batch = batch.to(device)
                pred = model(batch)
                loss = cosine_similarity_loss(pred, batch.y)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        # Scheduler step
        scheduler.step()

        # Logging
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'learning_rate': optimizer.param_groups[0]['lr']
        })

        print(f"Epoch {epoch+1}/{config['training']['num_epochs']}: "
              f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, "models/qcgn2oei_best.pth")
            print(f"✅ Best model saved (Val Loss: {val_loss:.4f})")

    print(f"Training complete. Best Val Loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    train_qcgn2oei()
```

### 2.2 設定ファイル

```yaml
# config/training.yml

model:
  node_dim: 128
  edge_dim: 64
  hidden_dim: 256
  num_layers: 10
  num_heads: 8
  output_dim: 1000  # m/z 50-1000 (950 bins + padding)
  dropout: 0.1

training:
  num_epochs: 300
  batch_size: 32  # RTX 5070 Ti optimal for GNN
  learning_rate: 0.001
  weight_decay: 1e-5
  early_stopping_patience: 50

data:
  bde_cache: "data/processed/bde_cache.h5"
  train_data: "data/processed/nist17_train.pt"
  val_data: "data/processed/nist17_val.pt"
  test_data: "data/processed/nist17_test.pt"
```

### 2.3 学習時間見積もり

**RTX 5070 Ti仕様**:
- CUDA cores: 8,960
- Tensor cores: 280 (Gen 5)
- Memory: 16GB GDDR7
- Memory bandwidth: 672 GB/s

**1エポックの時間**:
```
240,000 samples ÷ 32 batch_size = 7,500 iterations
7,500 iterations × 0.8 sec/iter = 6,000 sec = 1.67 hours
```

**合計学習時間**:
```
300 epochs × 1.67 hours = 500 hours → early stoppingで約48時間（30エポック程度で収束想定）
```

---

## Phase 3: 評価

### 3.1 評価メトリクス

```python
# scripts/evaluate.py
"""
Evaluate QC-GN2oEI model
"""

import torch
import numpy as np
from torch_geometric.loader import DataLoader
from src.models.qcgn2oei import QCGN2oEI
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

def cosine_similarity_metric(pred, target):
    """Calculate cosine similarity"""
    pred_norm = pred / (np.linalg.norm(pred, axis=1, keepdims=True) + 1e-8)
    target_norm = target / (np.linalg.norm(target, axis=1, keepdims=True) + 1e-8)
    return (pred_norm * target_norm).sum(axis=1).mean()

def top_k_recall(pred, target, k=10):
    """
    Top-K Recall: How many of the top-K true peaks are in top-K predictions
    """
    recalls = []
    for p, t in zip(pred, target):
        true_top_k = set(np.argsort(t)[-k:])
        pred_top_k = set(np.argsort(p)[-k:])
        recall = len(true_top_k & pred_top_k) / k
        recalls.append(recall)
    return np.mean(recalls)

def evaluate_model(
    model_path: str = "models/qcgn2oei_best.pth",
    test_data_path: str = "data/processed/nist17_test.pt"
):
    """Comprehensive model evaluation"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    checkpoint = torch.load(model_path)
    model = QCGN2oEI(...)  # Same config as training
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Load test data
    test_data = torch.load(test_data_path)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    # Inference
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(batch)
            all_predictions.append(pred.cpu().numpy())
            all_targets.append(batch.y.cpu().numpy())

    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    # Metrics
    cosine_sim = cosine_similarity_metric(predictions, targets)
    top10_recall = top_k_recall(predictions, targets, k=10)
    top20_recall = top_k_recall(predictions, targets, k=20)

    mse = mean_squared_error(targets.flatten(), predictions.flatten())
    rmse = np.sqrt(mse)

    print("=" * 60)
    print("QC-GN2oEI Evaluation Results")
    print("=" * 60)
    print(f"Cosine Similarity: {cosine_sim:.4f}")
    print(f"Top-10 Recall:     {top10_recall:.4f}")
    print(f"Top-20 Recall:     {top20_recall:.4f}")
    print(f"MSE:               {mse:.6f}")
    print(f"RMSE:              {rmse:.6f}")
    print("=" * 60)

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Example spectrum comparison
    idx = 0
    axes[0, 0].stem(targets[idx], linefmt='b-', markerfmt='bo', basefmt=" ", label="True")
    axes[0, 0].stem(predictions[idx], linefmt='r-', markerfmt='ro', basefmt=" ", label="Predicted")
    axes[0, 0].set_xlabel("m/z")
    axes[0, 0].set_ylabel("Intensity")
    axes[0, 0].set_title("Example Spectrum")
    axes[0, 0].legend()

    # Cosine similarity distribution
    cosine_sims = [cosine_similarity_metric(predictions[i:i+1], targets[i:i+1])
                   for i in range(len(predictions))]
    axes[0, 1].hist(cosine_sims, bins=50, alpha=0.7, color='green')
    axes[0, 1].axvline(cosine_sim, color='r', linestyle='--', label=f'Mean: {cosine_sim:.4f}')
    axes[0, 1].set_xlabel("Cosine Similarity")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_title("Cosine Similarity Distribution")
    axes[0, 1].legend()

    # Top-10 recall distribution
    top10_recalls = [top_k_recall(predictions[i:i+1], targets[i:i+1], k=10)
                     for i in range(len(predictions))]
    axes[1, 0].hist(top10_recalls, bins=20, alpha=0.7, color='orange')
    axes[1, 0].axvline(top10_recall, color='r', linestyle='--', label=f'Mean: {top10_recall:.4f}')
    axes[1, 0].set_xlabel("Top-10 Recall")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].set_title("Top-10 Recall Distribution")
    axes[1, 0].legend()

    # Error vs intensity
    errors = np.abs(predictions - targets).mean(axis=1)
    intensities = targets.max(axis=1)
    axes[1, 1].scatter(intensities, errors, alpha=0.3, s=10)
    axes[1, 1].set_xlabel("Max Intensity")
    axes[1, 1].set_ylabel("Mean Absolute Error")
    axes[1, 1].set_title("Error vs Intensity")

    plt.tight_layout()
    plt.savefig("results/evaluation.png", dpi=300)
    print("✅ Evaluation plots saved to results/evaluation.png")

if __name__ == "__main__":
    evaluate_model()
```

### 3.2 目標精度

| メトリクス | 目標値 | 備考 |
|----------|--------|------|
| **Cosine Similarity** | > 0.85 | QC-GN2oMS2論文で0.88達成 |
| **Top-10 Recall** | > 0.75 | 主要ピーク10個の再現率 |
| **Top-20 Recall** | > 0.80 | 主要ピーク20個の再現率 |
| **RMSE** | < 0.05 | 正規化強度での二乗平均平方根誤差 |

---

## 設定ファイル詳細

### config.yml（メイン設定）

```yaml
# config.yml - Main Configuration

project:
  name: "QC-GN2oEI"
  version: "2.1"
  description: "Physics-informed GNN for EI-MS prediction with BonDNet BDE-db2"

# BDE Configuration
bde:
  backend: "bondnet"  # Fixed (only option)

  bondnet:
    model_type: "bde-db2"  # Default model (retrained on BDE-db2)
    model_path: "models/bondnet_bde_db2_best.pth"
    dataset_path: "data/external/bde-db2"
    device: "cuda"
    batch_size: 256

    # Fallback for unsupported elements/structures
    fallback:
      aromatic_ring_bond: 120.0  # kcal/mol
      aliphatic_ring_bond: 85.0  # kcal/mol
      default_single_bond: 85.0
      default_double_bond: 146.0
      default_triple_bond: 200.0

# Data paths
data:
  nist17_path: "data/external/nist17/mainlib"
  bde_cache: "data/processed/bde_cache.h5"
  train_data: "data/processed/nist17_train.pt"
  val_data: "data/processed/nist17_val.pt"
  test_data: "data/processed/nist17_test.pt"

# Model architecture
model:
  type: "QCGN2oEI"

  # Node/Edge dimensions
  node_dim: 128
  edge_dim: 64

  # GNN layers
  hidden_dim: 256
  num_layers: 10
  num_heads: 8

  # Output
  output_dim: 1000  # m/z 50-1000 (950 bins + padding)

  # Regularization
  dropout: 0.1

  # Advanced features
  use_residual: true
  use_edge_features: true
  global_pooling: "mean"

# Training
training:
  num_epochs: 300
  batch_size: 32
  learning_rate: 0.001
  weight_decay: 1e-5

  # Optimizer
  optimizer: "RAdam"  # Same as QC-GN2oMS2

  # Scheduler
  scheduler: "CosineAnnealingLR"
  scheduler_params:
    T_max: 300
    eta_min: 1e-6

  # Loss function
  loss: "cosine_similarity"

  # Early stopping
  early_stopping_patience: 50

  # Checkpointing
  save_every: 10  # Save checkpoint every 10 epochs
  checkpoint_dir: "checkpoints"

# Evaluation
evaluation:
  metrics:
    - "cosine_similarity"
    - "top_k_recall"
    - "mse"
    - "rmse"

  top_k_values: [5, 10, 20, 50]

  # Visualization
  plot_examples: 10
  plot_dir: "results/plots"

# Hardware
hardware:
  device: "cuda"
  gpu_id: 0
  num_workers: 4
  pin_memory: true

  # Mixed precision training
  use_amp: true
  amp_dtype: "float16"

  # Memory optimization
  gradient_accumulation_steps: 1
  empty_cache_every: 100  # Empty CUDA cache every 100 batches

# Logging
logging:
  use_wandb: true
  wandb_project: "qcgn2oei"
  wandb_entity: null  # Set your wandb username

  log_every: 10  # Log every 10 batches
  save_predictions: true

# Reproducibility
seed: 42
deterministic: true
```

---

## 開発環境構築

### Dockerfile（更新版: Pure PyTorch）

```dockerfile
# .devcontainer/Dockerfile
# PyTorch-only environment for RTX 5070 Ti (sm_120)

FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda-12.8 \
    PATH=/usr/local/cuda-12.8/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    TORCH_CUDA_ARCH_LIST="9.0;12.0" \
    CUDA_LAUNCH_BLOCKING=0 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WORKDIR /workspace

# System packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    git wget curl vim build-essential cmake gcc g++ \
    ca-certificates gnupg lsb-release \
    python3.11 python3.11-dev python3.11-venv python3-pip \
    libssl-dev libffi-dev libxml2-dev libxslt1-dev zlib1g-dev \
    libopenblas-dev liblapack-dev libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Node.js for Claude CLI
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

RUN npm install -g @anthropic-ai/claude-code

# Python 3.11 default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Virtual environment
RUN python3.11 -m venv /opt/venv

RUN ln -sf /opt/venv/bin/python /usr/bin/python && \
    ln -sf /opt/venv/bin/python /usr/bin/python3 && \
    ln -sf /opt/venv/bin/pip /usr/bin/pip && \
    ln -sf /opt/venv/bin/pip /usr/bin/pip3

ENV PATH="/opt/venv/bin:$PATH"

# ===================================================
# PyTorch Nightly (cu128) - RTX 5070 Ti support
# ===================================================
RUN pip install --no-cache-dir nvidia-nvshmem-cu12==3.4.5

RUN pip install --no-cache-dir --pre \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128

# ===================================================
# Python packages (PyTorch-only stack)
# ===================================================
RUN pip install --no-cache-dir six hatchling wheel ninja

# Scientific computing
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    scipy==1.13.0 \
    pandas==2.2.2 \
    matplotlib==3.8.4 \
    seaborn==0.13.2 \
    plotly==5.20.0 \
    h5py==3.11.0 \
    pyyaml \
    tqdm \
    scikit-learn

# Chemistry libraries
RUN pip install --no-cache-dir \
    rdkit \
    mordred \
    mol2vec

# Jupyter
RUN pip install --no-cache-dir \
    jupyter \
    jupyterlab \
    ipython

# ===================================================
# PyTorch Geometric (sm_120 support)
# ===================================================
RUN pip install --no-cache-dir torch-geometric

# Build PyG extensions from source for sm_120
RUN export FORCE_CUDA=1 && \
    export TORCH_CUDA_ARCH_LIST="9.0;12.0" && \
    export CUDA_HOME=/usr/local/cuda-12.8 && \
    echo "Building PyG extensions from source..." && \
    pip install --no-cache-dir --no-build-isolation torch-scatter && \
    pip install --no-cache-dir --no-build-isolation torch-sparse torch-cluster torch-spline-conv

# ===================================================
# BonDNet and DGL (PyTorch-only)
# ===================================================
# Install DGL with CUDA 12.8 support
RUN pip install --no-cache-dir dgl -f https://data.dgl.ai/wheels/cu128/repo.html

# Install BonDNet from GitHub
RUN pip install --no-cache-dir git+https://github.com/txie-93/bondnet.git

# OGB for benchmarking
RUN pip install --no-cache-dir ogb>=1.3.6

# ML tools
RUN pip install --no-cache-dir \
    tensorboard \
    wandb \
    torch-ema

# Development tools
RUN pip install --no-cache-dir \
    pytest \
    black \
    flake8 \
    mypy

# Non-root user
RUN useradd -m -s /bin/bash devuser && \
    chown -R devuser:devuser /workspace && \
    chown -R devuser:devuser /opt/venv

# Auto-activate venv
RUN echo "source /opt/venv/bin/activate" >> /home/devuser/.bashrc && \
    echo "source /opt/venv/bin/activate" >> /root/.bashrc

ENV PYTHONPATH="/workspace:$PYTHONPATH"

# GPU verification script
RUN cat <<'SCRIPT' > /usr/local/bin/verify-gpu.py
#!/usr/bin/env python3
import torch
import sys

print("=" * 60)
print("RTX 50シリーズ GPU検証")
print("=" * 60)

cuda_available = torch.cuda.is_available()
print(f"CUDA利用可能: {cuda_available}")

if cuda_available:
    device_count = torch.cuda.device_count()
    print(f"GPUデバイス数: {device_count}")

    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Compute Capability: {props.major}.{props.minor}")
        print(f"  メモリ: {props.total_memory / 1e9:.1f} GB")
        print(f"  SM数: {props.multi_processor_count}")

        if props.major == 12 and props.minor == 0:
            print(f"  ✅ sm_120 (Blackwell) 検出!")

    print(f"\nPyTorch Version: {torch.__version__}")
    print(f"CUDA Version: {torch.version.cuda}")

    # GPU test
    try:
        test_tensor = torch.randn(1000, 1000).cuda()
        result = torch.mm(test_tensor, test_tensor)
        print("\n✅ GPU演算テスト成功!")
    except Exception as e:
        print(f"\n❌ GPU演算テスト失敗: {e}")
        sys.exit(1)

    # PyG test
    try:
        import torch_scatter
        print(f"\n✅ torch_scatter インストール済み")
        src = torch.randn(10, 5).cuda()
        index = torch.tensor([0, 1, 0, 1, 2, 0, 1, 2, 0, 1]).cuda()
        out = torch_scatter.scatter(src, index, dim=0, reduce="sum")
        print(f"   torch_scatter CUDA演算テスト成功!")
    except Exception as e:
        print(f"\n❌ torch_scatter エラー: {e}")
        sys.exit(1)

    # BonDNet test
    try:
        import dgl
        import bondnet
        print(f"\n✅ DGL version: {dgl.__version__}")
        print(f"✅ BonDNet インストール済み")
    except Exception as e:
        print(f"\n❌ BonDNet エラー: {e}")
        sys.exit(1)
else:
    print("❌ CUDAが利用できません")
    sys.exit(1)

print("=" * 60)
SCRIPT

RUN chmod +x /usr/local/bin/verify-gpu.py

USER devuser

CMD ["/bin/bash"]
```

### devcontainer.json

```json
{
  "name": "NExtIMS QC-GN2oEI (PyTorch + RTX 5070 Ti)",
  "dockerFile": "Dockerfile",
  "runArgs": [
    "--gpus", "all",
    "--ipc=host",
    "--ulimit", "memlock=-1",
    "--ulimit", "stack=67108864"
  ],
  "customizations": {
    "vscode": {
      "extensions": [
        "ms-python.python",
        "ms-python.vscode-pylance",
        "ms-toolsai.jupyter",
        "GitHub.copilot"
      ],
      "settings": {
        "python.defaultInterpreterPath": "/opt/venv/bin/python",
        "python.linting.enabled": true,
        "python.linting.flake8Enabled": true,
        "python.formatting.provider": "black"
      }
    }
  },
  "postCreateCommand": "python /usr/local/bin/verify-gpu.py"
}
```

---

## タイムライン

### 全体スケジュール

| フェーズ | タスク | 推定時間 | 累積時間 |
|---------|--------|---------|---------|
| **Phase 0** | BDE-db2ダウンロード | 30分 | 30分 |
| **Phase 0** | データ前処理 | 3時間 | 3.5時間 |
| **Phase 0** | BonDNet再学習 | 48-72時間 | 51.5-75.5時間 |
| **Phase 0** | モデル検証 | 1時間 | 52.5-76.5時間 |
| **Phase 1** | NIST読み込み | 30分 | 53-77時間 |
| **Phase 1** | BDE計算（BonDNet） | 75分 | 54.25-78.25時間 |
| **Phase 1** | PyG Graph生成 | 60分 | 55.25-79.25時間 |
| **Phase 2** | GNN学習 | 48時間 | 103.25-127.25時間 |
| **Phase 3** | 評価 | 2時間 | 105.25-129.25時間 |
| **合計** | - | **105-130時間** | **4.4-5.4日** |

### クリティカルパス

```
Phase 0 (BonDNet再学習) → Phase 1 (BDE計算) → Phase 2 (GNN学習) → Phase 3 (評価)
```

**並列化可能な作業**:
- Phase 0実行中: データローダー実装、GNNモデル実装
- Phase 1実行中: 学習スクリプト実装、評価スクリプト実装

---

## 参考文献

### 論文

1. **QC-GN2oMS2 (Original)**:
   - Zhang et al. (2024). "Quantum Chemistry-Informed Graph Neural Network for Mass Spectrum Prediction"
   - *Journal of Chemical Information and Modeling*
   - DOI: 10.1021/acs.jcim.4c00497
   - GitHub: https://github.com/PNNL-m-q/qcgnoms

2. **BonDNet**:
   - Xie & Grossman (2022). "Crystal Graph Convolutional Neural Networks for Accurate and Interpretable Prediction of Material Properties"
   - *Physical Review Letters*
   - GitHub: https://github.com/txie-93/bondnet

3. **BDE-db2**:
   - St. John et al. (2020). "A comprehensive database of bond dissociation enthalpies in organic molecules"
   - *Nature Scientific Data*
   - GitHub: https://github.com/patongroup/BDE-db2

4. **GATv2**:
   - Brody et al. (2021). "How Attentive are Graph Attention Networks?"
   - *ICLR 2022*

### データセット

- **NIST 17**: NIST Mass Spectral Library (2017)
- **BDE-db2**: 531,244 bond dissociation energies (Paton Group)

### ソフトウェア

- **PyTorch**: 2.10.0+ (nightly, CUDA 12.8)
- **PyTorch Geometric**: 2.6.1
- **DGL**: 2.1.0+
- **BonDNet**: Latest from GitHub
- **RDKit**: 2024.03.1

---

## まとめ

### v4.0の主要な改善点

1. **Pure PyTorch環境**: TensorFlow依存を完全削除
2. **BonDNet BDE-db2デフォルト化**: 最大のカバレッジと精度
3. **シンプルな構成**: プラガブルアーキテクチャ削除、保守性向上
4. **実装可能性**: xTBの間接BDE計算の複雑さを排除

### 期待される成果

- **精度**: Cosine Similarity > 0.85
- **速度**: 推論15ms/分子（実用レベル）
- **カバレッジ**: NIST 17の95%以上をサポート
- **解釈性**: BDEエッジ特徴による物理化学的解釈可能性

---

**Document Version**: 4.0
**Last Updated**: 2025-12-02
**Status**: Ready for Implementation
