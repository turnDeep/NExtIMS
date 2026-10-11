# NExtIMS Container

The image installs stable PyTorch2.9.1, PyG2.6.1 and RDKit2026.03.6.
No DGL, BonDNet, Claude, Node.js or compulsory PyG extension compilation.
The workspace is mounted once; data are not copied into the image.

```bash
# RTX5090-capable wheel; requires a compatible NVIDIA host driver
docker build -f .devcontainer/Dockerfile -t nextims:current .
docker run --rm -it --gpus all --shm-size=1gb -v "$PWD:/workspace" nextims:current
python -m pip install --no-deps -e .
nextims --device cuda --batch-size 16 --output paper_runs/current
```

In PowerShell use `-v "${PWD}:/workspace"`. In VSCode choose Reopen in Container.
The development container needs NVIDIA GPU passthrough. For CPU-only use:

```bash
docker build -f .devcontainer/Dockerfile --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu -t nextims:cpu .
docker run --rm -it -v "$PWD:/workspace" nextims:cpu
python -m pip install --no-deps -e .
nextims --mode predict --device cpu --checkpoint paper_runs/current/best_model.pth --predict-smiles CCO
```

For GTX1650 Ti select build argument
`TORCH_INDEX=https://download.pytorch.org/whl/cu126`.
Workers are0 and graph cache is bounded; large shared memory is not a substitute
for RAM/VRAM. Do not transfer the data or checkpoints through Git. For inference,
transfer your trusted `best_model.pth` and the exact matching source revision.
