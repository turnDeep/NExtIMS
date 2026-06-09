# NExtIMS v1.0 - Neural EI-MS Prediction System

次世代の電子衝撃イオン化マススペクトル（EI-MS）予測システム。QC-GN2oMS2の実証済みアプローチに基づく強力なGNNアーキテクチャを採用し、立体化学（Stereochemistry）と結合解離エネルギー（BDE）を統合した高精度な予測を実現します。

**設計哲学**: "Precision through Chemistry-Aware Deep Learning"

## ✨ 特徴

### コア機能

- **GNN Architecture**: GATv2Conv (デフォルト: 14層, 768 hidden dimension, 24 heads - RTX 5070 Ti最適化)
- **OpenMythos ACT Recurrent Refinement**: グラフプーリング後の表現を、適応的計算時間（ACT: Adaptive Computation Time）、LTI風安定化、ループ埋め込みを用いて再帰的に細分化・改良する機能。
- **Graph-level Chemistry Features**: mean/add/max poolingに加え、分子量・精密質量・元素組成カウントをグラフ全体の特徴量として統合。
- **Node Features**: 34次元（原子種、ハイブリダイゼーション、芳香族性、電荷、ラジカル、**立体化学/Chirality**）
- **Edge Features**: 10次元（結合次数、**BDE**、共役系、**結合立体化学/Stereo**）
- **Loss Function**: Cosine similarity loss + ACT Ponder Loss (推論時間調整用) + 分子イオン補助損失 + 中間スペクトル補助損失 (Auxiliary Loss)
- **Optimizer**: RAdam
- **評価メトリクス**: Cosine Similarity, Top-K Recall, MSE/RMSE, Spectral Angle
- **RTX 50シリーズ対応**: RTX 5070 Ti (16GB) 最適化アーキテクチャ。RTX 5090ではモデルを巨大化するより、batch sizeを増やして学習速度と勾配安定性を高めることを推奨。

## 🎯 性能目標

| メトリック | 目標値 | 判定基準 |
|-----------|--------|---------|
| **Cosine Similarity** | ≥ 0.85 | EXCELLENT |
| **Top-10 Recall** | ≥ 0.95 | EXCELLENT |

## 📋 システム要件

### 推論環境（最小要件）
- **CPU**: 4コア以上
- **RAM**: 8GB以上
- **GPU**: オプション（CPU推論可能）
- **ストレージ**: 1GB以上

### 学習環境（推奨構成）
- **CPU**: AMD Ryzen 7700 (8コア/16スレッド) 以上
- **GPU**: NVIDIA RTX 5070 Ti (16GB VRAM) 以上
- **RAM**: 32GB DDR5
- **ストレージ**: 1TB NVMe SSD
- **OS**: Ubuntu 22.04+ / Windows 11 with WSL2
- **CUDA**: 12.8+
- **PyTorch**: 2.7.0+
- **Python**: 3.10+

## 🚀 クイックスタート

### インストール

```bash
# 1. リポジトリのクローン
git clone https://github.com/turnDeep/NExtIMS.git
cd NExtIMS

# 2. 依存関係のインストール
pip install -r requirements.txt
pip install -e .  # パッケージとしてインストール

# 2a. Git LFS（BDE-db2使用時のみ必要）
# Ubuntu/Debian
sudo apt-get install git-lfs
git lfs install

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

# Option A: 公式Pre-trained modelを使用（推奨・高速）
# BonDNet公式の学習済みモデル (bdncm/20200808) を自動ダウンロード
python scripts/download_bondnet_pretrained.py --output models/bondnet_pretrained.pth

# Option B: BDE-db2で再学習（より高カバレッジ）
# 手順省略（scripts/train_bondnet_bde_db2.py を使用）
```

### トレーニング

```bash
# ACT Recurrent Refinement機能を利用したGNN学習
# 初回はACTの早期終了を抑制するため、--act-loss-weight を 0.001 から始めることを推奨します
python scripts/train_gnn_minimal.py \
    --nist-msp data/NIST17.MSP \
    --bde-cache data/processed/bde_cache/nist17_bde_cache.h5 \
    --output models/qcgn2oei_stable_act_refine.pth \
    --epochs 200 \
    --batch-size 32 \
    --hidden-dim 768 \
    --num-layers 14 \
    --num-heads 24 \
    --pooling-mode mean_add_max \
    --molecular-ion-loss-weight 0.1 \
    --use-recurrent-refinement \
    --refinement-steps 4 \
    --refinement-max-steps 8 \
    --use-lti-stabilization \
    --use-loop-embedding \
    --use-act \
    --act-loss-weight 0.001
```

> [!TIP]
> **段階的な機能検証**
> 1. ベースライン（Refinementなし）
> 2. ACTなし固定ループ（`--use-recurrent-refinement --refinement-steps 4 --refinement-max-steps 8 --use-lti-stabilization --use-loop-embedding`）
> 3. ACTあり（上記＋`--use-act --act-loss-weight 0.001`）
> の順に学習を行い、各機能の寄与を確認することを推奨します。

### 評価

```bash
# モデル評価（ACT搭載モデルも自動で構成を読み込みます）
python scripts/evaluate_minimal.py \
    --model models/qcgn2oei_stable_act_refine.pth \
    --nist-msp data/NIST17.MSP \
    --visualize --benchmark \
    --output-dir results/evaluation
```

### 推論

```bash
# ACT搭載モデルによる単一分子予測（画像出力付き）
python scripts/predict_gnn_minimal.py \
    --checkpoint models/qcgn2oei_stable_act_refine.pth \
    --smiles "CC(=O)OC1=CC=CC=C1C(=O)O" \
    --bde-cache data/processed/bde_cache/nist17_bde_cache.h5 \
    --top-k 20 \
    --output-spectrum outputs/pred_spectrum.csv \
    --output-peaks outputs/top_peaks.csv \
    --visualize \
    --output-image outputs/predicted.png

# 学習時4loop、推論時8loopでの確認（※学習時に refinement-max-steps=8 以上を指定している必要があります）
python scripts/predict_gnn_minimal.py \
    --checkpoint models/qcgn2oei_stable_act_refine.pth \
    --smiles "CC(=O)OC1=CC=CC=C1C(=O)O" \
    --bde-cache data/processed/bde_cache/nist17_bde_cache.h5 \
    --refinement-steps 8 \
    --top-k 20

# ACTを無効化し、固定8ループで推論（比較検証用）
python scripts/predict_gnn_minimal.py \
    --checkpoint models/qcgn2oei_stable_act_refine.pth \
    --smiles "CC(=O)OC1=CC=CC=C1C(=O)O" \
    --bde-cache data/processed/bde_cache/nist17_bde_cache.h5 \
    --refinement-steps 8 \
    --force-no-act \
    --top-k 20
```

> [!WARNING]
> `--refinement-max-steps` は学習時のチェックポイントと同じか小さく設定してください。学習時より大きい値を設定すると、`loop_embedding.weight` の形状エラー（次元不一致）が発生します。

## 📁 プロジェクト構造

```
NExtIMS/
├── setup.py                           # パッケージ設定 (v1.0)
├── requirements.txt                   # Python依存関係
├── README.md                          # このファイル
│
├── docs/
│   ├── spec_v4.2_minimal_iterative.md    # 技術仕様書
│   ├── PREDICTION_GUIDE.md               # 予測ガイド
│   └── BDE_AWARE_PREDICTION.md           # BDE対応予測
│
├── scripts/
│   ├── download_bondnet_pretrained.py    # Phase 0: BonDNetモデル取得
│   ├── download_bde_db2.py               # Phase 0: BDE-db2ダウンロード
│   ├── train_bondnet_bde_db2.py          # Phase 0: BonDNet学習
│   ├── train_gnn_minimal.py              # GNN学習 (Main, ACT Refinement対応)
│   ├── evaluate_minimal.py               # 評価
│   ├── predict_gnn_minimal.py            # 推論 (ACT Refinement対応)
│   ├── predict_single.py                 # 単一予測 (レガシー)
│   └── predict_batch.py                  # バッチ予測 (レガシー)
│
├── src/
│   ├── models/
│   │   ├── qcgn2oei_minimal.py          # GNNモデル (v1.0 Architecture)
│   │   └── modules.py                    # 共通モジュール
│   ├── data/
│   │   ├── nist_dataset.py              # NIST17データローダー
│   │   ├── graph_generator_minimal.py   # グラフ生成
│   │   ├── features_qcgn.py             # 特徴量抽出 (34/10 dim)
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
    └── processed/
        └── bde_cache/                    # BDE計算キャッシュ
```

## 🔬 使用例

### 例1: エタノールのスペクトル予測

```bash
nextims-predict "CCO" --model models/qcgn2oei_best.pth --visualize --output ethanol_spectrum.png
```

### 例2: バッチ予測（CSV入力）

**molecules.csv**:
```csv
smiles,id,name
CCO,mol_001,ethanol
CC(C)O,mol_002,isopropanol
```

```bash
nextims-predict-batch \
    --input molecules.csv \
    --output predictions.csv \
    --model models/qcgn2oei_best.pth
```

## 🔧 設定

主なハイパーパラメータは `scripts/train_gnn_minimal.py` の引数、またはモデル内のデフォルト値として定義されています。

- **Node Dim**: 34
- **Edge Dim**: 10
- **Hidden Dim**: 768
- **Layers**: 14
- **Heads**: 24

## 🧪 テスト

```bash
# 全テスト実行
pytest tests/
```

## 📝 ライセンス

MIT License

## 🤝 貢献

Issue、Pull Requestを歓迎します！

## 📧 連絡先

- GitHub Issues: https://github.com/turnDeep/NExtIMS/issues

---

**バージョン**: NExtIMS v1.0
**最終更新**: 2025-12-03
**ステータス**: Released
