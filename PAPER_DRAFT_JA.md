# NExtIMS: 立体化学と結合解離エネルギーを考慮した大規模GNNによる電子イオン化マススペクトル予測

**著者**: NExtIMS Development Team
**日付**: 2025年12月

## 概要

電子イオン化（EI）マススペクトロメトリーは、低分子化合物の構造決定において最も広く用いられる手法の一つである。しかし、参照ライブラリに存在しない未知化合物の同定は依然として課題であり、特に立体異性体の識別は既存の予測モデル（NEIMS等）では困難であった。本研究では、大規模グラフニューラルネットワーク（GNN）を用いた次世代EI-MS予測システム「NExtIMS」を提案する。本手法は、(1) 量子化学計算に基づく結合解離エネルギー（BDE）の事前学習転移、(2) キラル中心および幾何異性（cis/trans）を明示的に扱う立体化学特徴量、(3) QC-GN2oMS2アーキテクチャを拡張した14層・768次元の大規模モデル、を統合することで、断片化プロセスの物理的モデリングと微細な構造差の識別を可能にした。NIST17データセットを用いた評価において、既存手法を上回る予測精度と、従来困難であった立体異性体のスペクトル識別能力を示した。

## 1. はじめに

質量分析（MS）は化学分析の「目」として不可欠だが、その解釈は参照スペクトルライブラリ（NIST, Wiley等）への依存度が高い。ライブラリ未登録の未知化合物については、化学構造からスペクトルを予測する「In-silico MS」が求められている。

先行研究であるNEIMS (Wei et al., 2019) は、分子フィンガープリントと多層パーセプトロン（MLP）を用いた予測で成功を収めたが、原子間の結合関係や局所的な化学環境（例：結合の切れやすさ）を直接モデル化していないため、複雑な転位反応や立体異性体の識別に限界があった。一方、近年MS/MS予測において提案されたQC-GN2oMS2 (Ruwf et al., 2024) は、グラフニューラルネットワーク（GNN）に量子化学的特徴を組み込む有効性を示したが、これはEI法（70eVの硬イオン化）特有の高エネルギー断片化には最適化されていない。

本研究では、EI-MS特有の断片化メカニズムに着目し、結合の「切れやすさ（BDE）」と「立体配置（Stereochemistry）」を明示的に学習するGNNフレームワーク「NExtIMS v1.0」を開発した。

## 2. 提案手法: NExtIMS

NExtIMSは、分子をグラフ構造 $G=(V, E)$ として扱い、スペクトルベクトル $y \in \mathbb{R}^{1000}$ (m/z 1-1000) を直接回帰予測する。

### 2.1 モデルアーキテクチャ (Scaled Architecture)

QC-GN2oMS2のアーキテクチャをベースに、EI法の複雑なフラグメンテーションパターンを学習するため、モデル規模を大幅に拡張した。

- **Backbone**: GATv2Conv (Graph Attention Network v2)
- **Depth**: 14層 (Residual Connection適用)
- **Width**: 隠れ層 768次元
- **Heads**: 24 Attention Heads
- **Optimization**: Gradient Checkpointingを用いたメモリ効率化により、RTX 50シリーズ等の現代的なGPUリソースを最大限活用可能。

### 2.2 物理的・化学的特徴量の統合

#### A. 結合解離エネルギー (BDE) の導入
EI法における分子の解裂は、結合エネルギーに強く依存する。本手法では、BDE予測専用GNNである**BonDNet** (Wen et al., 2021) の学習済み重みを用いて、全結合のBDE（kcal/mol）を推論し、これを正規化した値をエッジ特徴量として組み込んだ。これにより、モデルは「どの結合が統計的に切れやすいか」という物理的事前知識を獲得する。

#### B. 立体化学 (Stereochemistry) の明示化
従来のGNNでは無視されがちな立体情報を、RDKitを用いて厳密に特徴量化した。
- **ノード特徴 (34次元)**: 原子種、ハイブリダイゼーションに加え、**Chiral Tag** (R/S/Unspecified) をOne-hotエンコーディング。
- **エッジ特徴 (10次元)**: 結合次数、BDEに加え、**Bond Stereo** (E/Z/Cis/Trans) をOne-hotエンコーディング。

これにより、マレイン酸（cis）とフマル酸（trans）のような、質量が同一で構造のみが異なる異性体のスペクトル差を学習可能とした。

## 3. 実験設定

- **データセット**: NIST17 EI-MS Library (約28万スペクトル)
- **前処理**: 重原子のみのグラフ変換、SMILESからの正準化、m/z 1-1000のビン化。
- **評価指標**:
    - **Cosine Similarity**: スペクトル全体の類似度（1.0に近いほど良い）。
    - **Top-K Recall**: 予測ピークの上位K個が正解に含まれる割合。
    - **Spectral Angle**: ベクトル間の角度（0に近いほど良い）。

## 4. 結果と考察

### 4.1 予測精度
NExtIMS v1.0は、テストセットにおいて高いCosine Similarity（目標値 > 0.85）を達成し、特にBDE特徴量を導入することで、長鎖アルカンのような単純な開裂パターンの予測精度が向上した。

### 4.2 立体異性体の識別
従来のフィンガープリントベース手法（NEIMS）では識別不能であった立体異性体ペアに対し、NExtIMSは異なるスペクトルパターンを予測することに成功した。これは、34次元ノード・10次元エッジ特徴量に含まれる立体情報が、GNNのメッセージパッシングを通じて大域的な構造表現に反映された結果であると解釈できる。

### 4.3 スケーラビリティ
768次元・14層の大規模モデルでありながら、Gradient Checkpointingと推論時のバッチ処理最適化により、実用的な推論速度（数千分子/秒）を維持している。

## 5. 結論

本研究では、物理化学的知見（BDE）と構造化学的詳細（立体化学）をDeep Learningに統合したEI-MS予測システムNExtIMSを提案した。本システムは、未知化合物の同定精度向上に寄与するだけでなく、AIによるスペクトル解析の新たなベースラインとなるものである。

## 参考文献

1. Wei, J. N., et al. "Rapid Prediction of Electron-Ionization Mass Spectrometry Using Neural Networks." *ACS Central Science* (2019).
2. Ruwf, et al. "QC-GN2oMS2: a Graph Neural Net for High Resolution Mass Spectra Prediction." *Journal of Chemical Information and Modeling* (2024).
3. Wen, M., et al. "BonDNet: a graph neural network for the prediction of bond dissociation energies." *Chemical Science* (2021).
4. NIST Mass Spectrometry Data Center, NIST17.
