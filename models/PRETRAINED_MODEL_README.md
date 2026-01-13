# BonDNet Pretrained Model

Downloaded from: https://zenodo.org/records/15117901

## Model Information

**Dataset:** BDNCM (Bond Dissociation for Neutral and Charged Molecules)
- Training data: 60,000+ unique homolytic and heterolytic bond dissociations
- Molecules: Neutral and charged species
- Elements: C, H, O, N (primary coverage)
- BDE types: Homolytic and heterolytic
- Performance: MAE ~0.022 eV (~0.51 kcal/mol)

**Paper:** BonDNet: a graph neural network for the prediction of bond dissociation
energies for charged molecules
- Published: Chemical Science, 2021
- DOI: 10.1039/D0SC05251E
- GitHub: https://github.com/mjwen/bondnet

## Usage for Transfer Learning

This pretrained model can be used as a starting point for fine-tuning on BDE-db2:

```bash
# Fine-tune on BDE-db2 dataset
python scripts/train_bondnet_bde_db2.py \
    --data-dir data/processed/bondnet_training/ \
    --pretrained models/bondnet_pretrained.pth \
    --output models/bondnet_bde_db2_finetuned.pth \
    --epochs 50 \
    --lr 0.0001
```

## Benefits of Transfer Learning

1. **Faster training**: 50-100 epochs instead of 200+
2. **Better generalization**: Pretrained on diverse molecules
3. **Lower learning rate**: 0.0001 (1/10 of from-scratch training)
4. **Retained knowledge**: C, H, O, N atom features already learned

## Note on Element Coverage

The pretrained model was trained primarily on C, H, O, N molecules.
For BDE-db2's additional elements (S, Cl, F, P, Br, I), fine-tuning will:
- Retain learned features for C, H, O, N
- Learn new features for halogens and other elements
- Achieve better performance than training from scratch

## Download Date

Model downloaded: 1768305675.9377275
