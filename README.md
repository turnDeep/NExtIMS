# NExtIMS - Minimal Configuration EI-MS Prediction System

次世代の電子衝撃イオン化マススペクトル（EI-MS）予測システム。QC-GN2oMS2の実証済みアプローチに基づく最小構成から開始し、反復改善で性能を向上させます。

**設計哲学**: "Start Simple, Iterate Based on Evidence"

## ✨ 特徴

### コア機能

- **GNN Architecture**: 10層GATv2Conv、8 attention heads、256 hidden dimension
- **Node Features**: 16次元（原子種、芳香族性、環構造、ハイブリダイゼーション、部分電荷）
- **Edge Features**: 3次元（BDE、結合次数、環内結合）
- **Loss Function**: Cosine similarity loss（QC-GN2oMS2と同じ）
- **Optimizer**: RAdam（lr=5e-5）
- **評価メトリクス**: Cosine Similarity, Top-K Recall, MSE/RMSE, Spectral Angle
- **RTX 50シリーズ対応**: RTX 5070 Ti (16GB)最適化

## 🎯 性能目標

| メトリック | 目標値 | 判定基準 |
|-----------|--------|---------|
| **Cosine Similarity** | ≥ 0.85 | EXCELLENT（採用完了） |
| Cosine Similarity | 0.80-0.85 | GOOD（軽微な改善検討） |
| Cosine Similarity | 0.75-0.80 | MODERATE（特徴量拡張推奨） |
| Cosine Similarity | < 0.75 | INSUFFICIENT（大幅拡張必要） |
| **Top-10 Recall** | ≥ 0.95 | 目標達成 |

**参考**: QC-GN2oMS2がMS/MSで**Cosine Similarity 0.88**を達成（16次元ノード、2次元エッジ）

## 📋 システム要件

### 推論環境（最小要件）
- **CPU**: 4コア以上
- **RAM**: 8GB以上
- **GPU**: オプション（CPU推論可能）
- **ストレージ**: 500MB以上

### 学習環境（推奨構成）
- **CPU**: AMD Ryzen 7700 (8コア/16スレッド) 以上
- **GPU**: NVIDIA RTX 5070 Ti (16GB VRAM) 以上
- **RAM**: 32GB DDR5
- **ストレージ**: 1TB NVMe SSD
- **OS**: Ubuntu 22.04+ / Windows 11 with WSL2
- **CUDA**: 12.8+
- **PyTorch**: 2.7.0+ (nightly)
- **Python**: 3.11+

## 🚀 クイックスタート

### インストール

```bash
# 1. リポジトリのクローン
git clone https://github.com/turnDeep/NExtIMS.git
cd NExtIMS

# 2. 依存関係のインストール
pip install -r requirements.txt

# 2a. Git LFS（BDE-db2使用時のみ必要）
# Ubuntu/Debian
sudo apt-get install git-lfs
git lfs install

# macOS
# brew install git-lfs
# git lfs install

# 2b. DGL & BonDNet（BDE計算に必要）
# CUDA版DGL
pip install dgl -f https://data.dgl.ai/wheels/torch-2.5/cu128/repo.html
# BonDNet
pip install git+https://github.com/mjwen/bondnet.git

# 3. データのダウンロード（NIST17）
# NIST EI-MSスペクトルデータベースを配置
# - NIST17.MSP: マススペクトルデータ（ピーク情報）を data/ に配置
# - mol_files/: 化学構造データ（MOLファイル）を data/mol_files/ に配置
# - ID番号でリンク: MSP内のIDとMOLファイル名（ID12345.MOL）が対応

# 4. BonDNet BDEモデル準備（Phase 0）

# Option A: 公式Pre-trained modelを使用（推奨、学習不要）
# BonDNet公式の学習済みモデル (bdncm/20200808) を自動ダウンロード
# NIST17カバレッジ: ~95%
# ※ 以降のスクリプトで --bondnet-model 未指定時に自動使用

# Option B: BDE-db2で再学習（より高カバレッジ、48-72時間必要）
# より多くの元素（Cl, Br, Iなど）をサポート

# ステップ1: データセットダウンロード
python scripts/download_bde_db2.py --output data/external/bde-db2

# ステップ2: BonDNet形式に変換（1-2時間）
python scripts/convert_bde_db2_to_bondnet.py \
    --input data/external/bde-db2/bde-db2.csv \
    --output data/processed/bondnet_training/

# ステップ3: BonDNet学習（48-72時間）
# メモリ効率化のためHDF5を使用することを推奨します

# 3a. HDF5データセット作成（数分〜数時間）
python scripts/bondnet_hdf5_dataset.py \
    --molecule-file data/processed/bondnet_training/molecules.sdf \
    --molecule-attributes data/processed/bondnet_training/molecule_attributes.yaml \
    --reaction-file data/processed/bondnet_training/reactions.yaml \
    --output data/processed/bondnet_training/bondnet_data.h5

# 3b. HDF5を使用して学習
python scripts/train_bondnet_bde_db2.py \
    --data-dir data/processed/bondnet_training/ \
    --use-hdf5 \
    --output models/bondnet_bde_db2_best.pth

# NIST17カバレッジ: ~99%+ (ハロゲン含有化合物対応)
```

### トレーニング（Phase 2）

```bash
# GNN学習（約40時間、RTX 5070 Ti）

# Option A使用時（公式Pre-trained model）
python scripts/train_gnn_minimal.py \
    --nist-msp data/NIST17.MSP \
    --bde-cache data/processed/bde_cache/nist17_bde_cache.h5 \
    --output models/qcgn2oei_minimal_best.pth \
    --epochs 200 \
    --batch-size 32
# --bondnet-model 未指定でbdncm/20200808を自動使用

# Option B使用時（再学習済みBonDNet）
python scripts/train_gnn_minimal.py \
    --nist-msp data/NIST17.MSP \
    --bde-cache data/processed/bde_cache/nist17_bde_cache.h5 \
    --bondnet-model models/bondnet_bde_db2_best.pth \
    --output models/qcgn2oei_minimal_best.pth \
    --epochs 200 \
    --batch-size 32
```

### 評価（Phase 3）

```bash
# モデル評価
python scripts/evaluate_minimal.py \
    --model models/qcgn2oei_minimal_best.pth \
    --nist-msp data/NIST17.MSP \
    --visualize --benchmark \
    --output-dir results/evaluation
```

### 推論（Phase 5）

```bash
# 単一分子予測
python scripts/predict_single.py "CCO" \
    --model models/qcgn2oei_minimal_best.pth \
    --visualize --output ethanol.png

# バッチ予測
python scripts/predict_batch.py \
    --input molecules.csv \
    --output predictions.csv \
    --model models/qcgn2oei_minimal_best.pth \
    --batch-size 64
```

## 📁 プロジェクト構造

```
NExtIMS/
├── config.yaml                        # システム設定（v2.0 legacy）
├── requirements.txt                   # Python依存関係
├── README.md                          # このファイル
│
├── docs/
│   ├── spec_v4.2_minimal_iterative.md    # v4.2仕様書
│   ├── PREDICTION_GUIDE.md               # 予測ガイド
│   ├── ARCHITECTURE_V2.md                # アーキテクチャ（v2.0 legacy）
│   └── BDE_AWARE_PREDICTION.md           # BDE対応予測（v2.1）
│
├── scripts/
│   ├── download_bde_db2.py               # Phase 0: BDE-db2ダウンロード
│   ├── train_bondnet_bde_db2.py          # Phase 0: BonDNet学習
│   ├── train_gnn_minimal.py              # Phase 2: GNN学習
│   ├── evaluate_minimal.py               # Phase 3: 評価
│   ├── predict_single.py                 # Phase 5: 単一予測
│   └── predict_batch.py                  # Phase 5: バッチ予測
│
├── src/
│   ├── models/
│   │   ├── qcgn2oei_minimal.py          # 最小構成GNNモデル
│   │   └── modules.py                    # 共通モジュール
│   ├── data/
│   │   ├── nist_dataset.py              # NIST17データローダー
│   │   ├── graph_generator_minimal.py   # グラフ生成（16/3次元）
│   │   ├── features_qcgn.py             # 特徴量抽出
│   │   ├── filters.py                   # データフィルタリング
│   │   └── bde_generator.py             # BDE計算
│   └── training/
│       └── losses.py                     # 損失関数
│
├── tests/
│   ├── test_evaluation_metrics.py       # 評価メトリクステスト
│   └── test_prediction.py               # 予測機能テスト
│
└── data/
    ├── NIST17.MSP                        # NIST EI-MSデータ
    ├── external/
    │   └── bde-db2/                      # BDE-db2データセット
    └── processed/
        └── bde_cache/                    # BDE計算キャッシュ
```

## 🔬 使用例

### 例1: エタノールのスペクトル予測

```bash
python scripts/predict_single.py "CCO" \
    --model models/qcgn2oei_minimal_best.pth \
    --visualize --output ethanol_spectrum.png
```

**出力**:
```
======================================================================
NExtIMS v4.2: Single Molecule Prediction Results
======================================================================
SMILES: CCO
Formula: C2H6O
Molecular Weight: 46.07 Da
Atoms: 9 (3 heavy)
----------------------------------------------------------------------
Inference Time: 45.23 ms
Max Intensity: 0.9234
Number of Peaks (>1%): 15
----------------------------------------------------------------------

Top 10 Predicted Peaks:
----------------------------------------------------------------------
Rank   m/z      Intensity    Relative %
----------------------------------------------------------------------
1      46       0.9234        100.0%
2      31       0.7821         84.7%
3      45       0.5432         58.8%
...
======================================================================
```

### 例2: バッチ予測（CSV入力）

**molecules.csv**:
```csv
smiles,id,name
CCO,mol_001,ethanol
CC(C)O,mol_002,isopropanol
CC(=O)C,mol_003,acetone
c1ccccc1,mol_004,benzene
```

```bash
python scripts/predict_batch.py \
    --input molecules.csv \
    --output predictions.csv \
    --model models/qcgn2oei_minimal_best.pth \
    --save-spectra spectra.npy
```

### 例3: モデル評価と性能判定

```bash
python scripts/evaluate_minimal.py \
    --model models/qcgn2oei_minimal_best.pth \
    --nist-msp data/NIST17.MSP \
    --visualize --num-visualize 20 \
    --benchmark
```

**出力**:
```
======================================================================
QC-GN2oEI Minimal Configuration Evaluation
======================================================================
Model: models/qcgn2oei_minimal_best.pth
Device: cuda
Node features: 16 dims
Edge features: 3 dims
Hidden dimension: 256
Number of layers: 10
----------------------------------------------------------------------

Spectral Similarity Metrics:
  Cosine Similarity:  0.8734
  Spectral Angle:     28.45°

Top-K Recall:
  Top-5 Recall:       0.8912
  Top-10 Recall:      0.9456
  Top-20 Recall:      0.9823
  Top-50 Recall:      0.9967

Error Metrics:
  MSE:                0.001234
  RMSE:               0.035128
  MAE:                0.012456

Performance:
  Avg Inference Time: 45.23 ms/sample
  Total Samples:      28,000
======================================================================

======================================================================
Performance Assessment
======================================================================
✅ EXCELLENT: Cosine Similarity >= 0.85
   Recommendation: Adopt v4.2 minimal configuration!
   No feature expansion needed.
======================================================================
```

## 📊 反復改善プロセス

v4.2は評価結果に基づいて段階的に改善します：

```
Phase 3: 評価
     ↓
┌────────────────────────────────────┐
│ Cosine Similarity = ?              │
└────────────────────────────────────┘
     ↓
     ├─ ≥ 0.85 → ✅ 完了！v4.2採用
     │
     ├─ 0.80-0.85 → v4.3 (軽微拡張)
     │                ├─ ノード: 16 → 25 (+9)
     │                └─ 再評価
     │
     ├─ 0.75-0.80 → v4.3 (中度拡張)
     │                ├─ ノード: 16 → 30 (+14)
     │                ├─ エッジ: 3 → 4 (+1)
     │                └─ 再評価
     │
     └─ < 0.75 → v4.3 (中間構成)
                    ├─ ノード: 16 → 64 (+48)
                    ├─ エッジ: 3 → 32 (+29)
                    └─ 再評価
```

## 🔧 設定ファイル

主な設定は `config.yaml` で管理されています：

```yaml
# モデル設定（最小構成）
model:
  teacher:
    gnn:
      num_layers: 10         # GATv2Conv層数
      hidden_dim: 256        # 隠れ層次元
      node_features: 16      # ノード特徴次元
      edge_features: 3       # エッジ特徴次元
      dropout: 0.1

# 学習設定
training:
  teacher_multitask:
    batch_size: 32           # バッチサイズ（RTX 5070 Ti最適化）
    num_epochs: 200
    learning_rate: 5.0e-5    # RAdam学習率
    optimizer: "RAdam"
    scheduler: "CosineAnnealingLR"

# GPU設定（RTX 5070 Ti）
gpu:
  use_cuda: true
  device_ids: [0]
  mixed_precision: true
```

## 🧪 テスト

```bash
# 評価メトリクステスト
python tests/test_evaluation_metrics.py

# 予測機能テスト（RDKit必要）
python tests/test_prediction.py

# モデルテスト（PyTorch Geometric必要）
python tests/test_models.py
```

## 📖 ドキュメント

- **[v4.2仕様書](docs/spec_v4.2_minimal_iterative.md)**: 完全な技術仕様
- **[予測ガイド](docs/PREDICTION_GUIDE.md)**: 推論機能の使い方
- **[BDE対応予測](docs/BDE_AWARE_PREDICTION.md)**: BDE統合の詳細（v2.1）

## ⚡ パフォーマンス

### 学習時間（RTX 5070 Ti 16GB）

#### Option A: 公式Pre-trained BonDNet使用

| フェーズ | 時間 | 説明 |
|---------|------|------|
| Phase 0 | **0時間** | BonDNet公式モデル (bdncm/20200808) 自動使用 |
| Phase 1 | 2時間 | データ準備（NIST17, 280K spectra） |
| Phase 2 | 40時間 | GNN学習（200 epochs, early stopping） |
| Phase 3 | 2時間 | 評価・可視化 |
| **合計** | **約2日** | すぐに始められる |

#### Option B: BDE-db2で再学習

| フェーズ | 時間 | 説明 |
|---------|------|------|
| Phase 0 | **48-72時間** | BonDNet BDE-db2再学習（より高カバレッジ） |
| Phase 1 | 2時間 | データ準備（NIST17, 280K spectra） |
| Phase 2 | 40時間 | GNN学習（200 epochs, early stopping） |
| Phase 3 | 2時間 | 評価・可視化 |
| **合計** | **約5-6日** | 上級者向け |

### 推論速度

| バッチサイズ | スループット | 用途 |
|-------------|-------------|------|
| 1 | 20 molecules/sec | 単一予測 |
| 32 | ~700 molecules/sec | 標準バッチ |
| 64 | ~4,000 molecules/sec | 高速バッチ（トレーニング） |
| 128 | ~8,000 molecules/sec | 最適化バッチ（推論） |

### メモリ使用量

- **学習時**: ~8-12 GB VRAM (batch_size=32)
- **推論時**: ~4-8 GB VRAM (batch_size=128)
- **CPU推論**: ~2-4 GB RAM

## 🐛 トラブルシューティング

### CUDA Out of Memory

```bash
# バッチサイズを削減
python scripts/train_gnn_minimal.py \
    --batch-size 16  # 32 → 16に削減
```

### 学習が遅い

```bash
# BDEキャッシュを使用
python scripts/precompute_bde.py \
    --nist-msp data/NIST17.MSP \
    --output data/processed/bde_cache/nist17_bde_cache.h5

# キャッシュを使って学習
python scripts/train_gnn_minimal.py \
    --bde-cache data/processed/bde_cache/nist17_bde_cache.h5
```

### サポートされていない元素

サポート元素: **C, H, O, N, F, S, P, Cl, Br, I**

```python
# フィルタリング例
from src.data.filters import SUPPORTED_ELEMENTS

for smiles in smiles_list:
    mol = Chem.MolFromSmiles(smiles)
    if all(atom.GetSymbol() in SUPPORTED_ELEMENTS for atom in mol.GetAtoms()):
        # 予測実行
        predict(smiles)
```

## 📚 参考文献

### QC-GN2oMS2（基盤アーキテクチャ）
- Ruwf et al., "QC-GN2oMS2: a Graph Neural Net for High Resolution Mass Spectra Prediction", *JCIM* (2024)
- GitHub: https://github.com/PNNL-m-q/QC-GN2oMS2

### BonDNet & BDE-db2（BDE計算）
- Kim et al., "BonDNet: A Graph Neural Network for the Prediction of Bond Dissociation Energies", *Chemical Science* (2021)
- St. John et al., "Expansion of bond dissociation prediction with machine learning", *Digital Discovery* (2023)
- BDE-db2: 531,244 reactions

### NEIMS（ベースライン）
- Wei et al., "Rapid Prediction of Electron-Ionization Mass Spectrometry Using Neural Networks", *ACS Central Science* (2019)

## 📝 ライセンス

MIT License

## 🤝 貢献

Issue、Pull Requestを歓迎します！

## 📧 連絡先

- GitHub Issues: https://github.com/turnDeep/NExtIMS/issues

---

**バージョン**: NExtIMS v4.2 (Minimal Configuration)
**最終更新**: 2025-12-03
**ステータス**: Ready for Training & Evaluation
