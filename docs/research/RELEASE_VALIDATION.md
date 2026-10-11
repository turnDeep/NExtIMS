# v1.1.0 Runtime Validation

Date: 2026-10-11. These checks establish that the published runtime can execute;
they do not establish prediction accuracy or generalization on NIST or unknown
compounds. Experimental efficacy is documented separately in RESULTS.md.

## Completed Checks

| Check | Result | Scope |
|---|---|---|
| Native Windows unit tests | 57 tests passed | Before two additional evidence-integrity tests were added |
| Fresh CPU Docker build | Passed | Python 3.11, torch 2.9.1+cpu, PyG 2.6.1, RDKit 2026.03.6; no DGL or BonDNet |
| Final CPU Docker unit tests | 59 tests passed | Runtime, defaults, checkpoint compatibility, split handling, export integrity |
| Editable installation and console entry point | Passed | `nextims --help` and full-size synthetic pipeline |
| Full-size synthetic training | Passed | hidden 768, 14 layers, 24 heads; one epoch, four SAM optimizer updates |
| Save, reload, completed-run resume, prediction, reevaluation | Passed | Same checkpoint and source hashes; CSV/MSP prediction requires no target spectrum |
| Wheel build | Passed | `nextims-1.1.0-py3-none-any.whl` |
| Installed-wheel prediction outside the checkout | Passed | Installed into an isolated container; `nextims` predicted from `/tmp` using the synthetic checkpoint |
| Frozen runtime source bytes in Git index | Passed | All 12 frozen Python files match their raw working-file Git hashes |
| Evidence export audit | Passed | Aggregate row counts and SHA256 match manifest; no compound-level evaluation rows exported |

The synthetic smoke input contains 30 unique structures with artificial spectra.
Its group split has 27 training records, one validation record and two test
records. Metrics from that input are **not** reported as research performance.
The smoke covers the actual model size and algorithmic path, not full-dataset
training time or worst-case memory use.

## Reproduction

From the repository root, build the CPU variant of the same Dockerfile:

```bash
docker build -f .devcontainer/Dockerfile --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu -t nextims-release:cpu .
docker run --rm --cpus 4 --memory 5g -v "$PWD:/workspace" -w /workspace nextims-release:cpu python scripts/verify_release.py --output work/release_qa
```

`verify_release.py` installs the package, checks the console entry point, runs the
unit tests and the full-size synthetic pipeline, and writes logs plus a JSON
receipt. Its checkpoint and synthetic data remain local under the requested
output directory, not in the published evidence archive. The smoke needs several
GiB of host memory even though it runs on CPU.

Build an installable wheel separately:

```bash
python -m pip install build
python -m build --wheel --outdir work/release_wheel
```

The wheel was also installed with `pip install --no-deps` into a separate fresh
CPU container with the runtime dependencies already present. Prediction used the
installed package from `/opt/venv/lib/python3.11/site-packages`, not the checkout.

## Not Claimed

- The default CUDA 12.8 image was not freshly built and tested in this release
  check. The completed new Docker validation used the CPU variant.
- No new physical RTX5090 run was performed for this publication. The large
  RTX5090 results in README are recovered historical evaluation CSVs, not new
  inference from migrated weights.
- EXP-V022 full NIST training on GTX1650 Ti remains in progress. It uses an
  unchanged, separate frozen runtime. Final validation and test metrics are not
  available yet.
- The synthetic smoke does not guarantee that a particular physical batch size
  fits all NIST structures. Use the capacity mode before a large run.
- Old checkpoints that require actual BDE inputs need their historical Git
  revision and dependencies. Current inputs do not recreate missing BDE values.
