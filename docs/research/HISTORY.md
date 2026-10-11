# Research Version History

2026-10-11にGit履歴と手元の集計CSVを照合した公開版です。
**EXP番号は検証設計の番号であって、リリース番号や性能順位ではありません。**
過去に小規模で有望だった案も、後続検証や実機報告で撤回されています。
この公開作業で全学習を再実行したわけではありません。

## Gitと研究記録

Gitのmainには2026-06-11のfusion採用までが反映済みでした。
その後のEXPはローカルの実験コード・報告・集計として保存されていました。
v1.1.0で現行runnerとその根拠を公開します。過去のソフトウェア版と
設計名の対応は[READMEの履歴](../../README.md#version履歴)を参照してください。
全590コミットの公開前スナップショットは[evidence/git_history.csv](evidence/git_history.csv)です。

| ID | 検証内容 | 確認した結果・現在の扱い | 公開証拠 |
|---|---|---|---|
| EXP-V001 | RTX5090・直接予測/固定反復/ACT/NL loss/fusion/段数 | fusion0.30が探索群の最大。ACT単独は小さい効果。重み未移行、CSV解析のみ | [RTX5090](evidence/EXP-V001/test_evaluation_summary.csv) |
| EXP-V002 | 残差スペクトル補正、品質教師あり停止、exit loss | 品質停止の優位性は不安定。残差の方が有望。共通再検証の数値で比較可 | [V008](evidence/EXP-V008/comparison_all.csv) |
| EXP-V003 | 残差 + 自己予測スペクトルfeedback | 初期短期screenで一部改善。後続V020ではfeedback除外が改善し、現在OFF | [V008](evidence/EXP-V008/comparison_all.csv)、[V020](evidence/EXP-V020/comparison.csv) |
| EXP-V004 | 各段のCosine + Top-k弱教師 | Top-k学習を追加。初期のCosine差は小さいがV020で除外すると低下 | [V008](evidence/EXP-V008/comparison_all.csv)、[V020](evidence/EXP-V020/comparison.csv) |
| EXP-V005 | m/zと質量差を同時に扱う単一decoder | 共通screenではdual fusionより低下。現在は独立2 headを維持 | [V008](evidence/EXP-V008/comparison_all.csv) |
| EXP-V006 | TRM由来のanswer/latent反復と分割学習 | 更新数を増やした条件の優位は、同一更新数では維持されず。標準化しない | [V008](evidence/EXP-V008/comparison_all.csv) |
| EXP-V007 | 同じバッチを4回optimizer更新するportable runner | 動作検証後、V008/V009で予算比較。反復段数/勾配蓄積とは別 | [V008](evidence/EXP-V008/comparison_all.csv)、[V009](evidence/EXP-V009/comparison.csv) |
| EXP-V008 | V001〜007を同一小規模条件で再比較 | Train1581/Val200/Test196、2 seed。full予算とupdate_matchedを別集計 | [比較一覧](evidence/EXP-V008/comparison_all.csv) |
| EXP-V009 | V004 80epoch 対 V007 20epoch、各4000更新 | checkpoint機会を20候補に合わせるとV007 Test0.447292対V0040.444666。圧倒的差ではない | [比較](evidence/EXP-V009/comparison.csv) |
| EXP-V010 | 同じバッチreplayから、別microbatchの更新へ | r01はsmoke、r02はaccuracy screen。同じ更新数でもサンプル露出数が違うため単純比較しない | [保存集計](evidence/EXP-V010/) |
| EXP-V011 | 7:3を保ち反復/ACT/残差/feedbackをfactorial比較 | 反復なしfusionを強い対照として再確認。残差固定4段の改善はfusion導入ほど大きくない | [集計](evidence/EXP-V011/condition_means.csv)、[差](evidence/EXP-V011/gap_means.csv) |
| EXP-V012 | 電荷競合・fragment survival・条件付き転位 | 電荷競合が有望。他2案は全指標の一貫改善なし。電荷位置の実測ではない | [集計](evidence/EXP-V012/condition_means.csv) |
| EXP-V013 | 固定残差4段×電荷枝の2×2比較 | 追加2000件Cosine: fusion0.389926→両方0.418303。ただしgapは0.006951→0.011900で広がる | [集計](evidence/EXP-V013/condition_means.csv)、[差](evidence/EXP-V013/gap_means.csv) |
| EXP-V014 | V013に2回予測・一致制約・weight decay・late SAM | 2回予測は改善。一致制約の上積みなし。late SAMは低下。後の初期からSAMとは条件が違う | [集計](evidence/EXP-V014/condition_means.csv) |
| EXP-V015 | fragment表現、候補均衡、多尺度、差分loss、query | 一様な大幅改善なし。pair差分はVal Cosine-0.013001、Top-kも低下 | [V014との比較](evidence/EXP-V015/comparison_with_previous.csv) |
| EXP-V016 | 物理batchと勾配蓄積 | r03で全10学習確認。同一epochでは更新数差が大きい。同一256更新で大batchはCosineとTop-kがtrade-off | [比較](evidence/EXP-V016/comparison.csv) |
| EXP-V017 | GTX1650 Ti・全量index/容量と既存学習曲線の監査 | 容量確認まで。V007の全量学習曲線を新たに得た結果ではない | [範囲](RESULTS.md#全量検証exp-v022) |
| EXP-V018 | Dropout増加、EMA、plateau LR、2モデルensemble | 単一モデルは「Valを下げずgap縮小」のgateを満たさず。ensembleは別の2モデル費用で精度改善 | [比較](evidence/EXP-V018/comparison.csv) |
| EXP-V019 | 初期からSAM、random depth、蒸留、cycle/SWA | SAMが2 seedのVal精度/gap gateを満たし新内部Testでも改善。random depthはTest seed43で低下 | [比較](evidence/EXP-V019/comparison.csv) |
| EXP-V020 | ACT・残差・feedback・各段教師・SAMを1つずつ除外 | feedback全OFFが最有力。残差/各段教師/SAMは保持。ACTのTop-k優位は未確定 | [比較](evidence/EXP-V020/comparison.csv) |
| EXP-V021 | feedback OFFでACT対固定4段 | 新内部TestでACT Cosine+0.002770。Top-k差は不確定、ほぼ4段停止 | [比較](evidence/EXP-V021/comparison.csv) |
| EXP-V022 | 768/14/24・全有効データ・feedback OFF・ACT4・残差・各段教師・SAM | 学習中。全量モデルの最終精度やbaselineとの差は未確定 | [計画と現在の範囲](RESULTS.md#全量検証exp-v022) |

V012〜015の主screenは96/3/4、Train1581/Val200、3 seed、30epoch/1500更新。
追加2000件は既存RTX5090 Test由来で、完全に新規な外部データではありません。
V018以降はTrain3200/Val400、96/3/4、2 seed、80epoch/8000更新が主条件です。
この2系列やRTX5090の絶対値を同じ条件として順位づけしません。

## BDEと旧特徴量の履歴

| 検討 | 証拠と判断 |
|---|---|
| BonDNet公式重みの利用 | bdncm/20200808の実cacheに20,082/49,902レコードが命中（40.2%）。元素制約と欠損が大きい |
| BDE-db2再学習 | Gitに変換・再学習・mapping修正・HDF5読込の実装あり。公式cache検証を再学習済みモデルの性能証拠と混同しない |
| BDEエッジ | 全edge対no_bdeでCosine+0.000402。occlusionでは影響が非常に小さく、現在不使用 |
| BDEノード集約 | 5k/8epochで0.736048→0.741561。ただしTop5/Top10/Top20はすべてbaselineより低く、普遍的改善ではない |
| carbon/electronic/ring | 5k screenでは0.754531、Top5/10/20 0.572984/0.641129/0.690927。ユーザーの実機では著しい低下が報告され撤回。小規模結果だけで標準化しない教訓 |
| distance特徴量 | 5kで0.735024、baseline0.736048より低下。現在不使用 |
| 全特徴量追加 | 小規模の最良選択組合せより低い。表現の重複・ノイズや学習難度増加の可能性。原因確定ではない |
| molecular-ion auxiliary loss | 69d03ecで導入後、f67326eで削除。分子量/組成入力は維持、分子イオン補助lossは現行から除外 |

[実BDE比較](evidence/HIST-BDE/edge_compare_summary.csv)、
[edge occlusion](evidence/HIST-BDE/full_edge_occlusion_results.csv)、
[旧特徴量ablation](evidence/HIST-FEATURE/feature_ablation_results.csv)。

## その他の旧prototype

以下はEXP番号以前のHIST系列です。独立した最終Testの確証とはせず、
採用に至らなかったものも残しています。

| 系列 | 保存結果・採用しない理由 |
|---|---|
| ランダム50k固定fusion探索 | 3epoch程度の弱学習条件では0.50のVal Cosine0.403835、0.30は0.400862。最適weightは条件依存。後のRTX5090比較では0.30が最大なので現行は0.30を維持 |
| ACT連動の制限付き可変fusion | 固定0.50からCosine-0.000074、Top20-0.000942。複雑化を正当化する一貫改善なし |
| consensus/motif | raw50k Cosine0.400959、consensus0.392716、motif追加0.396636。consensus内では改善してもrawを全指標で超えない |
| 水素シフトfusion | proxy条件でCosineは微増するがTop5/Top20は低下。十分学習した機構の否定ではなく本proxyを不採用 |
| fragment cascade | proxyの小さなCosine/Top5変化とTop10/Top20 trade-off。化学反応の物理的再現が確認されたものではない |
| 10案Docker再確認 | [数値と実施範囲](evidence/HIST-DOCKER10/docker_recheck_10ideas_summary.csv)。未学習/推論proxyを完全再学習の性能と混同しない |
| EMA/ensemble/peak presence/experts/target peak loss/augmentation/sampling | 旧HISTの個別試行は条件が異なり、この公開版では統一再監査していない。未採用であり「検証済み最良」とは扱わない。EMA/ensembleは後のV018で改めて検証 |
| MoLFormer/LoRA等の文献案 | 研究候補の議論と、実装・精度検証済みを区別。現行runnerには組み込まれていない |

数値の原CSVとhashは[evidence/manifest.json](evidence/manifest.json)で追跡できます。
モデル重みやNISTの分子別データは非公開のローカル証拠として保持されています。

## 旧仕様書

`docs/spec_v*.md`、`docs/spec_v4.0_pytorch_bde-db2.md`、
`docs/spec_v4.1_pytorch_bde-db2_filtered.md`等は当時の設計資料です。
そこにあるDGL/BonDNet導入や大型モデル推奨は現行の手順ではありません。
削除されたBDE実行コードは過去のGit revisionから復元できます。
