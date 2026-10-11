# Current NExtIMS Quick Start

The current recipe is documented in [README.md](README.md). Legacy scripts and
specifications do not define its defaults.

```bash
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e .
nextims --help
nextims --mode prepare --data-dir data --output paper_runs/current
nextims --prepared-db paper_runs/current/preparation/dataset.sqlite --device cuda --output paper_runs/current
nextims --mode predict --device cpu --checkpoint paper_runs/current/best_model.pth --predict-smiles CCO --prediction-output prediction.csv
```

Use `cu126` for the validated GTX1650 Ti runtime, or `cpu` for CPU-only use.
Default physical batch is16, accumulation1, same-batch updates4, FP32 SAM.
Use a new output for changed data/settings. `--resume` requires identical settings.
Preparing separately and then training must use the prepared-db path consistently
on any subsequent resume. No BDE computation, pretrained weights or NIST data
are bundled. Input layout and assessment limits are in the main README.
