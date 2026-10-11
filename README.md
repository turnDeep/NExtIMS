# NExtIMS

化学構造から電子イオン化（EI）マススペクトルを予測する研究用GNNです。化合物の構造をSMILESまたはMOLから読み込み、m/z 1–1000の相対強度を予測します。スペクトルから構造を直接生成するモデルではありません。

**現在のソフトウェア版はv1.1.0。研究構成はEXP-V021を踏まえたEXP-V022です。** 2026-10-11時点で大規模なEXP-V022の学習は進行中です。下記の小規模結果をそのまま本番サイズの性能とみなさないでください。学習済み重み・NISTのライセンス対象データは配布していません。

## 現在の構成

| 項目 | 設定 |
|---|---|
| 分子エンコーダ | エッジ特徴量を用いるGATv2系GNN、hidden 768・14層・24 heads |
| 特徴量 | ノード34次元・エッジ10次元・分子量/元素数17次元 |
| Pooling | mean + add + maxを連結。平均的環境・量的情報・強い局所信号を保持 |
| 出力 | 独立したm/z headとneutral-loss headをm/z軸へ戻して7:3で融合 |
| 反復 | 共有refinement block、ACT（Adaptive Computation Time）上限4段 |
| 残差補正 | 前段のスペクトルを基準に、制限された補正を加える |
| スペクトルfeedback | 入力への戻し込み・停止診断への使用ともOFF |
| 学習 | 最終Cosine + neutral-loss補助loss 0.05 + 各段Cosine 0.05/Top-k順位loss 0.02 + ACT ponder 0.001 |
| Optimizer | RAdam、lr 0.00005、SAM rho 0.02、weight decay 0、gradient clip 1 |
| 数値計算 | FP32、TF32 OFF。AMPは本構成では使用しない |
| データ/メモリ | SQLite遅延読込、workers=0、CPUグラフキャッシュ上限64 MiB、空きRAM保護 |
| 学習停止 | 最大200 epoch、最低20 epoch、4指標すべてが10 epoch停滞したら停止 |
| 採用重み | 完全なValidation全体のCosineが最大になったepoch |

BDE、追加のcarbon/electronic/ringノード特徴量、molecular-ion auxiliary lossは現行標準では使用しません。BDEを入れていたエッジ列は重みの互換性のため定数0.5として残り、BDEの情報は含みません。BonDNet、DGL、Claudeの実行コード・環境依存は削除しました。過去の検討は下記の履歴に残しています。

ACTの段数は1分子の1回の予測内の反復回数です。epochは全学習データを通る回数、`updates-per-batch`は同じバッチでパラメータを更新する回数です。3つは別物です。各段教師が正解を参照するのは学習時だけで、未知分子の推論に正解スペクトルは必要ありません。

## インストール

Python 3.10以上。基準環境はPython 3.11 / PyTorch 2.9.1 / PyG 2.6.1 / RDKit 2026.03.6です。CPUでも予測でき、専用VRAMは必須ではありません。768/14/24のCPU学習は長時間かかります。

```bash
git clone https://github.com/turnDeep/NExtIMS.git
cd NExtIMS
python -m venv .venv
# Linux: source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e .
```

RTX5090は`cu128`、GTX1650 Tiは`cu126`、CPU専用は`cpu`を上記URL末尾に指定します。これは基準版の指定であり、最新Nightlyが必須という意味ではありません。[PyTorch公式の対応wheel](https://pytorch.org/get-started/previous-versions/)。PyG追加拡張のソースビルドは必須ではありません。[PyG公式の基本インストール](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)。

Docker/VSCode Dev Containerは[環境手順](.devcontainer/README.md)を参照してください。

## データ配置

次の両方に対応します。MOLの名前は`ID1.MOL`などのIDまたは数値名で、MSPのIDと対応させます。MSPに有効なSMILESがあればMOLは不要です。

```text
data/
  NIST17.MSP
  NIST17.MOL/        # 単一ファイルではなく、多数のMOLファイルが入ったフォルダ
    ID1.MOL
    ID2.MOL
```

```text
data/
  1-50000/NIST17.MSP
  1-50000/NIST17.MOL/ID1.MOL
  50001-100000/NIST17.MSP
  50001-100000/NIST17.MOL/ID50001.MOL
  ...
```

無効構造・未対応元素・不正なスペクトルは事前に除外し、`rejected_entries.csv`へ理由を記録します。標準は**有効レコード全量**です。同一canonical isomeric SMILESのreplicateを同じsplitにまとめ、Train/Validation/Testを概ね80/10/10へ分けます。構造グループ単位なので件数は厳密な80/10/10にはなりません。これはscaffold splitではなく、立体異性体の骨格類似性まで排除する保証もありません。

## 学習・評価・予測

リポジトリ直下で実行します。`nextims`コマンドの代わりに`python scripts/nextims.py`でも同じように実行できます。

```bash
# 現行標準を全量学習。安全側の物理batchは16。
nextims --mode train --data-dir data --device cuda --batch-size 16 --output paper_runs/current

# RTX5090でまず容量を確認（学習重みは破棄される）
nextims --full-mode capacity --device cuda --batch-size 256 --output paper_runs/capacity256

# 容量確認後の本学習。実測VRAMに余裕を残すこと。
nextims --device cuda --batch-size 256 --output paper_runs/current5090

# 同じ設定・同じソースで、保存済みepochから再開
nextims --device cuda --batch-size 256 --output paper_runs/current5090 --resume

# 学習後の選択済みモデルでTestを集計
nextims --mode evaluate --device cpu --checkpoint paper_runs/current5090/best_model.pth --prepared-db paper_runs/current5090/preparation/dataset.sqlite --output paper_runs/reevaluation

# 未知構造を予測。CPUで実行可。
nextims --mode predict --device cpu --checkpoint paper_runs/current5090/best_model.pth --predict-smiles CCO --prediction-output prediction.csv

# name,smiles列を含むCSVを予測ライブラリへ変換
nextims --mode predict --device cpu --checkpoint paper_runs/current5090/best_model.pth --input-csv candidates.csv --prediction-output predicted_library.msp
```

現行SAM runnerでは`--grad-accumulation 1`に固定します。標準の`--updates-per-batch 4`は同じバッチを4回更新し、各更新でSAMの2回のforward/backwardを実施します。4回の反復推論でも、実効batchを4倍にする勾配蓄積でもありません。`--batch-size 1024`が全分子で安全に動く保証はないため、256等から容量確認してください。

`best_model.pth`、`history.csv`、`status.json`、`EXPERIMENT.json`、`data_audit.json`、最終`evaluation_summary.csv`と`REPORT.md`を出力します。各splitの分子別・段別評価も`evaluation/`に出ます。CPUグラフキャッシュは有界ですが、モデル/optimizer/チェックポイントにもRAMとディスクが必要です。途中のオンラインTrainはDropout ONなので、Train–Val差はDropout OFFの固定診断または最終全Train評価で判断します。

新CLIはEXP-V020/021/022系列のresearch checkpointを読みます。v1.0の旧重みは`scripts/predict_gnn_minimal.py`等の旧CLIを使用してください。実BDE入力を必要とする旧重みは、対応する過去のGit revisionと環境で実行してください。重みの新旧を自動的に混ぜません。モデルを第三者から入手した場合、`torch.load`は信頼したファイルに限定してください。

## 大規模検証結果

RTX5090から移行したEXP-V001の保存済み評価CSVです。768/14/24、seed 42、最大200 epoch、batch256、Test 30,294件。同じ表の条件間で比較します。重みはこのPCへ移行されていないため、再推論の結果ではありません。[全数値](docs/research/evidence/EXP-V001/test_evaluation_summary.csv)。

| 構成 | Test Cosine | Top5 | Top10 | Top20 |
|---|---:|---:|---:|---:|
| 直接m/z予測 | 0.755154 | 0.554572 | 0.561468 | 0.574983 |
| 固定反復・NLなし | 0.754549 | 0.556176 | 0.566914 | 0.582658 |
| ACT・NLなし | 0.756503 | 0.557635 | 0.568456 | 0.582226 |
| ACT + NL補助lossのみ | 0.756693 | 0.556896 | 0.568888 | 0.581546 |
| ACT + fusion 0.10 | 0.825528 | 0.626784 | 0.620803 | 0.620227 |
| ACT + fusion 0.20 | 0.833369 | 0.634112 | 0.628778 | 0.627799 |
| ACT + fusion 0.30 | **0.834204** | **0.637948** | **0.630884** | **0.629128** |
| ACT + fusion 0.40 | 0.833565 | 0.635981 | 0.629656 | 0.626041 |
| ACT + fusion 0.50 | 0.833795 | 0.635552 | 0.628121 | 0.623671 |

このfusion探索群では0.30が4指標で最大でした。直接予測からCosineは+0.079051で、ACT単独の+0.001350より大きい改善です。補助lossだけではNL側の出力が最終予測に直接反映されません。融合は、分子イオンからの質量差で学習した予測をm/zへ戻し、絶対m/zの予測を補完するため有効だったと考えられます。ただし、これは化学反応経路を実証した結果ではなく、0.30は化学法則や脱離確率ではありません。Testを含む過去の開発比較なので、全く未使用の外部検証と同一視できません。

## 小規模検証結果

EXP-V020: hidden96/3層/4 heads、Train3,200・Validation400・内部Test800、seed42/43平均、80 epoch・各8,000更新。各行は同じ小規模条件です。SAMありは16,000 forward/backward、なしは8,000なので、計算量まで同じではありません。[全数値](docs/research/evidence/EXP-V020/comparison.csv)。

| 条件 | Val Cosine | Val Top5 | Val Top10 | Val Top20 | Test Cosine | 固定Train–Val差 |
|---|---:|---:|---:|---:|---:|---:|
| ACT4 + residual + feedback + 各段教師 + SAM | 0.486249 | 0.298500 | 0.346125 | 0.389875 | 0.481622 | 0.136067 |
| SAMなし | 0.473542 | 0.286500 | 0.336125 | 0.381000 | 0.467883 | 0.186450 |
| 残差補正なし | 0.457977 | 0.279500 | 0.331250 | 0.382063 | 0.460296 | 0.125582 |
| 各段教師なし | 0.478462 | 0.291000 | 0.340250 | 0.384188 | 0.470822 | 0.138549 |
| feedback全OFF | **0.512325** | **0.320250** | **0.359875** | **0.410938** | **0.510410** | 0.050609 |
| fusionのみ・反復なし + SAM | 0.501785 | 0.311500 | 0.358375 | 0.405688 | 0.499526 | **0.035735** |

feedback除外は精度を上げながら差を縮めました。ただし差が最小なのは反復なしであり、差の縮小だけを性能としません。残差補正と各段教師は除くと低下する傾向、SAMはVal精度と差の両方で有望でした。追加feedbackが過学習経路を増やすという説明は仮説で、分子内機構を直接計測したものではありません。

EXP-V021ではfeedback OFFのままACT対固定4段を直接比較しました。Train/Valは同じ、Testは別の内部800件です。V020のTestと混ぜません。[全数値](docs/research/evidence/EXP-V021/comparison.csv)。

| 条件 | Val Cosine | 新Test Cosine | 新Test Top5 | Top10 | Top20 | 固定Train–Val差 |
|---|---:|---:|---:|---:|---:|---:|
| ACT上限4段 | 0.512325 | **0.510893** | 0.318250 | 0.348813 | 0.401844 | 0.050609 |
| 固定4段 | 0.511569 | 0.508123 | 0.318125 | 0.347500 | 0.401375 | 0.047957 |
| fusionのみ + SAM | 0.501785 | 0.498758 | 0.312750 | 0.339875 | 0.390094 | 0.035735 |

ACTはCosineに小さな上積みがありましたが、Top-k差のbootstrap区間は0をまたぎ、すべてが確実に改善したとは言えません。ACTの実停止深度は平均3.981段で、ほぼ4段上限です。期待深度3.592とは異なり、サンプルが停止してもバッチ全体の計算を続ける実装なので、そのまま高速化の証拠にもなりません。

## BDEを含む検討履歴

結合の切れやすさ（BDE）をBonDNetで高速推定し、エッジと周辺ノードへ反映する方法を検討しました。GitにはBDE-db2への変換・再学習・HDF5遅延読込の実装履歴もあります。しかし、確認できた実BDE比較は**公式bdncm/20200808重み**のcacheによるものです。BDE-db2再学習後の予測精度向上を、これらの結果から主張することはできません。

| 2026-06予備検証（先頭5,000件・1 seed・8 epoch・96/4/4） | Val Cosine | Top5 | Top10 | Top20 |
|---|---:|---:|---:|---:|
| 全エッジ特徴量 + 実BDE | 0.736048 | 0.560887 | 0.632056 | 0.684173 |
| BDE除外 | 0.735646 | 0.560887 | 0.633669 | 0.685484 |
| BDEだけのエッジ | 0.741813 | 0.558065 | 0.629839 | 0.687399 |
| BDEノード集約（別の特徴量確認試験） | 0.741561 | 0.556855 | 0.630645 | 0.681552 |

全エッジでBDE有無のCosine差は+0.000402、BDE列を隠すと-0.000152で、依存は弱い結果でした。ノード集約もTop-kを一様には改善せず、採用を支える大規模証拠はありません。cache命中は20,082/49,902件（40.2%）で、元素対応と低カバレッジも制約です。EIの開裂は中性分子のBDEだけでなくイオン化後の電荷・ラジカル・転位・二次開裂に依存するため、単純なBDE列で十分説明できなかった可能性があります。**BDE一般が無意味と結論したのではなく、本実装の標準入力から外した判断です。** [BDEの数値](docs/research/evidence/HIST-BDE/edge_compare_summary.csv)。

## Version履歴

ソフトウェア版（v1.x）、過去の設計名（v4.x/v5.x）、研究ID（EXP-Vxxx）は別です。コミットがあることと、性能向上を確認したことも分けます。

| 時期 / 版 | 主な変更と判断 | Git履歴 / 研究記録 |
|---|---|---|
| 2025-12 BDE検討期 | BonDNet公式重み、BDE-db2再学習経路、HDF5ストリーミング、atom mapping修正 | [39fece3](https://github.com/turnDeep/NExtIMS/commit/39fece3)、[33c6995](https://github.com/turnDeep/NExtIMS/commit/33c6995)、[5e22e05](https://github.com/turnDeep/NExtIMS/commit/5e22e05)、[306daa8](https://github.com/turnDeep/NExtIMS/commit/306daa8) |
| 2026-01 設計v4.x/v5.x | 特徴量・立体情報、768/14/24、勾配checkpointing。設計名はパッケージ版ではない | [ad101a6](https://github.com/turnDeep/NExtIMS/commit/ad101a6)、[d3c2107](https://github.com/turnDeep/NExtIMS/commit/d3c2107)、[b836643](https://github.com/turnDeep/NExtIMS/commit/b836643) |
| 2026-01 ソフトウェアv1.0.0 | 初回リリースの記録 | [6ed883b](https://github.com/turnDeep/NExtIMS/commit/6ed883b) |
| 2026-02 設計v5.3 | 1280/16/32の大型化案。現在の標準では撤回 | [86bd215](https://github.com/turnDeep/NExtIMS/commit/86bd215) |
| 2026-04 ACT | OpenMythosを参考に共有反復ブロックと適応停止を導入 | [ad42c32](https://github.com/turnDeep/NExtIMS/commit/ad42c32) |
| 2026-06 PR214 | 標準を768/14/24へ。GPU名からモデル容量を決める方針を撤回 | [cd0727c](https://github.com/turnDeep/NExtIMS/commit/cd0727c) |
| 2026-06 graph features | mean/add/maxと分子量・組成を追加。分子イオン補助lossは後に削除 | [69d03ec](https://github.com/turnDeep/NExtIMS/commit/69d03ec)、[f67326e](https://github.com/turnDeep/NExtIMS/commit/f67326e) |
| 2026-06 fusion | NL補助lossだけから、NL出力を実際に融合する構成へ | [d01a68e](https://github.com/turnDeep/NExtIMS/commit/d01a68e) |
| 2026-06〜10 EXP-V001〜018 | 特徴量・可変fusion・単一head・TRM・電荷枝・正則化を比較。条件と反証を記録 | [研究Version一覧](docs/research/HISTORY.md) |
| 2026-10 EXP-V019〜021 | SAM、部品除外、feedback OFFでのACT直接比較 | [検証結果と考察](docs/research/RESULTS.md) |
| 2026-10-11 v1.1.0 / EXP-V022 | 現行CLIを公開、BDE/DGL/BonDNet/Claudeの実行依存を撤去、全量検証は進行中 | [現行構成の範囲](docs/research/RESULTS.md#全量検証exp-v022) |

全Git履歴の保存済みスナップショットは[git_history.csv](docs/research/evidence/git_history.csv)、検証CSVの出典とSHA256は[manifest.json](docs/research/evidence/manifest.json)にあります。NIST本体、化合物別の評価行、巨大な重みは公開していません。

## 限界と再現性

- 小規模と大規模の絶対値を横断比較しません。モデル容量、split、更新数、checkpoint選択が違います。
- EXP-V021までの改善は内部NISTデータの少数seedでの結果です。未登録化合物全般や法薬物の最終同定精度を保証しません。
- EXP-V022の対象は原データ306,622件から有効グラフ302,667件（Train242,291 / Val30,153 / Test30,223）。全量学習中で最終結果は未確定です。過去に利用されたsplitであり、未使用の外部Testではありません。
- Top-kは予測と実測の上位k個のm/z集合の一致数/kです。ゼロ強度の同順位binが入り得る従来定義を維持し、正強度ピークのみのrecallも補助出力します。
- 予測MSPは実測標準ライブラリではありません。標準品・保持時間等との確認を併用してください。
- ソースhashとレシピを保存し、異なるモデルコードの重みを拒否します。再開は同じパス・設定・ソース・データを使用してください。

```bash
python -m pip install -r requirements-dev.txt
python -m unittest discover -s scripts -p 'test_*.py'
```

旧仕様書は歴史資料です。現行の実行方法はこのREADMEと[QUICKSTART](QUICKSTART.md)を優先してください。MITライセンス（データセットの権利は別途確認）。

公開前の動作確認では、新しいCPU Docker環境で59件のunit testと、768/14/24の合成データ学習・保存・再開・予測・再評価が成功しました。wheelをインストールした作業フォルダ外からの予測も確認しています。これは実行経路の検査で、NISTの精度検証やRTX5090の再検証ではありません。[動作確認の範囲と再現手順](docs/research/RELEASE_VALIDATION.md)。
