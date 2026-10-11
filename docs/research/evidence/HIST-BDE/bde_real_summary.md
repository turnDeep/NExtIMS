> Historical report: not current defaults; see ../../HISTORY.md for later reversals.

# Real BDE Cache Verification (BonDNet bdncm/20200808)

## BDE cache
- Supported species from the official model: C, F, H, Li, O
- Supported unique SMILES before prediction: 11,881
- Cached molecules: 11,601
- Failed during BonDNet prediction: 15
- Cached BDE bond values: 89,988
- Full NIST valid-entry cache hit rate: 20,082 / 49,902 = 40.2%
- Full NIST unique-SMILES cache hit rate: 11,224 / 31,469 = 35.7%

## Retraining comparison
Conditions: first 5,000 MSP entries, seed 42, val split 0.1, hidden 96, layers 4, heads 4, 8 epochs, ACT refinement, neutral-loss fusion weight 0.30.

| mode | val cosine | val Top5 | val Top10 | val Top20 | train cosine | train-val gap |
|---|---:|---:|---:|---:|---:|---:|
| full_edge | 0.736048 | 0.560887 | 0.632056 | 0.684173 | 0.719868 | -0.016180 |
| no_bde | 0.735646 | 0.560887 | 0.633669 | 0.685484 | 0.719883 | -0.015764 |
| bond_order_only | 0.739401 | 0.558468 | 0.628024 | 0.687903 | 0.722489 | -0.016912 |
| bde_only | 0.741813 | 0.558065 | 0.629839 | 0.687399 | 0.722775 | -0.019038 |

## Full-edge occlusion: edge features
Baseline full-edge validation: cosine 0.735995, Top10 0.632056.

| feature | delta cosine | delta Top5 | delta Top10 | delta Top20 | prevalence |
|---|---:|---:|---:|---:|---:|
| edge_in_ring | 0.000269 | 0.000403 | -0.000000 | 0.000101 | 0.316642 |
| group_edge_core_all | 0.000069 | 0.002016 | 0.001210 | -0.000706 | 0.629047 |
| edge_stereo_E | 0.000032 | 0.000000 | 0.000403 | 0.000000 | 0.015813 |
| edge_stereo_cis | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| edge_stereo_trans | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| edge_stereo_any | -0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| edge_stereo_Z | -0.000005 | 0.000000 | 0.000000 | 0.000101 | 0.005271 |
| edge_bond_order | -0.000136 | 0.001210 | -0.000202 | -0.001008 | 1.000000 |
| edge_BDE_normalized | -0.000152 | -0.000403 | -0.000000 | -0.000101 | 0.946913 |
| edge_is_conjugated | -0.000221 | -0.001210 | 0.000403 | -0.000907 | 0.252636 |
| group_edge_all | -0.000222 | 0.002419 | 0.000605 | 0.000101 | 0.351619 |
| group_edge_stereo_all | -0.000514 | 0.000806 | -0.000000 | -0.000605 | 0.166667 |
| edge_stereo_none | -0.000554 | 0.001210 | -0.000202 | -0.000302 | 0.978916 |

## Interpretation
- Real BDE cache did not improve full-edge over no-BDE in this controlled run: full_edge val cosine 0.736048 vs no_bde 0.735646, a +0.000402 difference.
- bde_only was highest on val cosine at 0.741813, but Top5/Top10 were lower than full_edge/no_bde, so BDE alone changes ranking behavior rather than uniformly improving all metrics.
- In full_edge occlusion, zeroing edge_BDE_normalized changed cosine by -0.000152, effectively no useful reliance.
- The practical bottleneck remains architecture/fusion: BDE as a raw GAT edge column is weakly used, even when real cache values exist.
