> Historical report: not current defaults; see ../../HISTORY.md for later reversals.

# Node Feature Group Verification

## Implemented feature groups
- BDE node aggregation: adjacent BDE min/mean/max/std, low-BDE fraction, real-BDE availability flag
- Carbon role: methyl, methylene, methine, quaternary carbon, terminal atom, branch point, hetero-adjacent carbon
- Distance: graph distance to nearest ring, hetero atom, and aromatic atom with availability flags
- Electronic: Gasteiger charge, Crippen logP/MR atom contributions, EState, outer electrons
- Ring detail: ring size 3/4/5/6/7+, ring count, multi-ring, bridgehead-like, aromatic-ring atom

## Screening: 2^5 factorial
Conditions: first 2,000 MSP entries, seed 42, val split 0.1, hidden 96, layers 4, heads 4, 5 epochs.

| rank | feature groups | val cosine | delta | Top10 | gap | node dim |
|---:|---|---:|---:|---:|---:|---:|
| 1 | bde__carbon__distance | 0.646796 | +0.032442 | 0.624365 | +0.000664 | 53 |
| 2 | bde__carbon__distance__electronic__ring | 0.643318 | +0.028963 | 0.595939 | -0.009954 | 67 |
| 3 | carbon__electronic__ring | 0.642610 | +0.028255 | 0.614721 | -0.002312 | 55 |
| 4 | carbon__electronic | 0.642217 | +0.027863 | 0.610152 | -0.009052 | 46 |
| 5 | bde__carbon__distance__ring | 0.638979 | +0.024625 | 0.605584 | +0.004813 | 62 |
| 6 | bde__distance__electronic__ring | 0.637865 | +0.023510 | 0.608629 | -0.004898 | 60 |
| 7 | distance__electronic__ring | 0.637777 | +0.023422 | 0.601523 | -0.001877 | 54 |
| 8 | carbon__distance__ring | 0.635990 | +0.021636 | 0.597462 | -0.004753 | 56 |
| 9 | carbon__distance | 0.635750 | +0.021395 | 0.598985 | +0.002724 | 47 |
| 10 | bde__distance__electronic | 0.635740 | +0.021385 | 0.618782 | +0.002392 | 51 |
| 11 | carbon__ring | 0.635084 | +0.020730 | 0.600508 | -0.000674 | 50 |
| 12 | carbon | 0.635026 | +0.020671 | 0.600000 | -0.000866 | 41 |

### Single-feature screening
| group | val cosine | delta | Top10 delta |
|---|---:|---:|---:|
| bde | 0.618713 | +0.004359 | -0.012183 |
| carbon | 0.635026 | +0.020671 | -0.000508 |
| distance | 0.624056 | +0.009701 | +0.001015 |
| electronic | 0.621967 | +0.007613 | -0.008122 |
| ring | 0.629659 | +0.015304 | -0.004061 |

### Three-way interactions
| triple | val cosine | delta | 3-way interaction |
|---|---:|---:|---:|
| bde+carbon+distance | 0.646796 | +0.032442 | +0.031402 |
| bde+distance+ring | 0.632108 | +0.017754 | +0.021369 |
| bde+carbon+ring | 0.619046 | +0.004692 | +0.013324 |
| distance+electronic+ring | 0.637777 | +0.023422 | +0.012669 |
| carbon+distance+ring | 0.635990 | +0.021636 | +0.010945 |
| bde+distance+electronic | 0.635740 | +0.021385 | +0.010436 |
| carbon+electronic+ring | 0.642610 | +0.028255 | +0.009235 |
| bde+electronic+ring | 0.620852 | +0.006497 | +0.005650 |
| bde+carbon+electronic | 0.629501 | +0.015147 | +0.002951 |
| carbon+distance+electronic | 0.630264 | +0.015910 | -0.010475 |

## Confirmation: 5,000 entries / 8 epochs
Baseline is the previous full-edge run with the same data size, seed, architecture, ACT refinement, and neutral-loss fusion weight 0.30.

| feature groups | node dim | val cosine | delta | Top5 | Top10 | Top20 | gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| carbon__electronic__ring | 55 | 0.754531 | +0.018483 | 0.572984 | 0.641129 | 0.690927 | -0.023613 |
| all | 67 | 0.746950 | +0.010903 | 0.556452 | 0.623589 | 0.681855 | -0.024736 |
| carbon | 41 | 0.746616 | +0.010568 | 0.562903 | 0.631855 | 0.682560 | -0.026898 |
| bde__carbon__distance | 53 | 0.745587 | +0.009540 | 0.566129 | 0.631855 | 0.687399 | -0.027243 |
| ring | 43 | 0.743721 | +0.007673 | 0.564113 | 0.633468 | 0.682863 | -0.024581 |
| bde | 40 | 0.741561 | +0.005513 | 0.556855 | 0.630645 | 0.681552 | -0.023616 |
| carbon__electronic | 46 | 0.740243 | +0.004195 | 0.554839 | 0.632661 | 0.679435 | -0.021137 |
| baseline | 34 | 0.736048 | +0.000000 | 0.560887 | 0.632056 | 0.684173 | -0.016180 |
| electronic | 39 | 0.735995 | -0.000053 | 0.559274 | 0.636089 | 0.680444 | -0.021389 |
| distance | 40 | 0.735024 | -0.001024 | 0.545968 | 0.622581 | 0.680746 | -0.019066 |

## Interpretation
- Best confirmed setting: carbon + electronic + ring detail, with val cosine +0.018483 over baseline.
- Carbon role features are the best single group and remain useful in larger confirmation.
- BDE node aggregation is mildly positive alone and participates in the best screening triple, but the larger confirmation favored carbon + electronic + ring detail over BDE-containing combos.
- Distance features looked useful in screening synergy but were negative alone in the larger confirmation, so they should not be adopted as a default yet.
- Adding every feature group is worse than the selected combo, suggesting feature bloat/noise still matters.
