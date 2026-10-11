"""Streaming MSP/MOL preparation and bounded-memory graph construction."""

import csv
from contextlib import closing
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from torch.utils.data import Dataset
from torch_geometric.data import Data
from tqdm import tqdm

from .features import QCGNFeaturizer

ELEMENTS = ["C", "O", "H", "N", "S", "Si", "P", "Cl", "F", "I",
            "Se", "As", "B", "Br", "Sn"]
FEATURIZER = QCGNFeaturizer(use_bde=False)
SCHEMA_VERSION = 1


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_id(value):
    value = str(value).strip().upper()
    if value.startswith("ID"):
        value = value[2:]
    return str(int(value)) if value.isdigit() else value


def discover_inputs(data_dir, msp=None, mol_dir=None):
    root = Path(data_dir).resolve()
    if msp:
        sources = [Path(msp).resolve()]
    else:
        flat = [p for p in root.iterdir() if p.is_file() and p.name.upper() == "NIST17.MSP"]
        sources = flat or sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".msp")
    if not sources or any(not p.is_file() for p in sources):
        raise ValueError("No MSP found. Use --data-dir or --msp to specify the real data.")
    mol_root = Path(mol_dir).resolve() if mol_dir else root
    if not mol_root.is_dir():
        raise ValueError("--mol-dir must be a DIRECTORY containing individual MOL files.")
    return root, sources, mol_root


def iter_msp(path):
    """Keep only one record in memory; enforce declared peak counts."""
    record = None
    declared = None
    seen = 0
    ordinal = 0
    numeric_pair = re.compile(r"^\s*([+-]?[\d.]+(?:[eE][+-]?\d+)?)\s+([+-]?[\d.]+(?:[eE][+-]?\d+)?)")
    with Path(path).open(encoding="utf-8-sig", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.lower().startswith("name:"):
                if record is not None:
                    record["peak_count_ok"] = declared is not None and declared == seen
                    yield record
                ordinal += 1
                record = {"name": line.split(":", 1)[1].strip(), "ordinal": ordinal, "peaks": {}}
                declared, seen = None, 0
                continue
            if record is None:
                continue
            if ":" in line and not numeric_pair.match(line):
                key, value = line.split(":", 1)
                key = key.lower().replace(" ", "").replace("_", "")
                if key == "numpeaks":
                    try:
                        declared = int(value.strip())
                    except ValueError:
                        declared = -1
                elif key in {"id", "smiles", "inchikey", "formula"}:
                    record[key] = value.strip()
                continue
            if declared is None:
                continue
            for pair in line.split(";"):
                match = numeric_pair.match(pair)
                if not match:
                    continue
                mz, intensity = map(float, match.groups())
                seen += 1
                if math.isfinite(mz) and math.isfinite(intensity) and intensity > 0:
                    bin_mz = int(round(mz))
                    if 1 <= bin_mz <= 1000:
                        record["peaks"][bin_mz] = record["peaks"].get(bin_mz, 0.0) + intensity
        if record is not None:
            record["peak_count_ok"] = declared is not None and declared == seen
            yield record


def connect(path):
    db = sqlite3.connect(str(path))
    db.execute("PRAGMA cache_size=-8192")
    db.execute("PRAGMA temp_store=FILE")
    return db


def index_mols(db, mol_root):
    db.execute("CREATE TABLE mols (parent TEXT, identifier TEXT, path TEXT, PRIMARY KEY(parent, identifier))")
    db.execute("CREATE INDEX mol_id ON mols(identifier)")
    count = 0
    for folder, _dirs, files in os.walk(mol_root):
        for name in files:
            path = Path(folder) / name
            if path.suffix.lower() != ".mol":
                continue
            parent = str(path.parent.parent.resolve())
            identifier = normalize_id(path.stem)
            previous = db.execute("SELECT path FROM mols WHERE parent=? AND identifier=?", (parent, identifier)).fetchone()
            if previous:
                raise ValueError(f"Ambiguous MOL ID {identifier}: {previous[0]} and {path}")
            db.execute("INSERT INTO mols VALUES (?, ?, ?)", (parent, identifier, str(path.resolve())))
            count += 1
            if count % 10000 == 0:
                db.commit()
                print(f"Indexed {count:,} MOL paths (on disk)", flush=True)
    db.commit()
    return count


def resolve_mol(db, source, identifier):
    if not identifier:
        raise ValueError("missing ID and valid SMILES; cannot safely join by row order")
    identifier = normalize_id(identifier)
    local = db.execute("SELECT path FROM mols WHERE parent=? AND identifier=?", (str(source.parent.resolve()), identifier)).fetchall()
    paths = local or db.execute("SELECT path FROM mols WHERE identifier=? LIMIT 2", (identifier,)).fetchall()
    if len(paths) != 1:
        raise ValueError(f"MOL ID {identifier}: {'ambiguous' if paths else 'not found'}")
    mol = Chem.MolFromMolFile(paths[0][0], sanitize=True, removeHs=False)
    if mol is None:
        raise ValueError("invalid MOL")
    return mol


def make_graph(smiles, peaks=None, sample_id=0):
    mol = Chem.MolFromSmiles(smiles)
    valid, reason = FEATURIZER.validate_molecule(mol)
    if not valid:
        raise ValueError(reason)
    nodes = np.array([FEATURIZER.get_atom_features(a) for a in mol.GetAtoms()])
    edges, edge_features = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        features = FEATURIZER.get_edge_features(bond)
        edges.extend([(i, j), (j, i)])
        edge_features.extend([features, features])
    edge_index = torch.tensor(edges, dtype=torch.long).reshape(-1, 2).t().contiguous()
    edge_attr = torch.tensor(np.array(edge_features), dtype=torch.float32).reshape(-1, 10)
    exact_mass = float(rdMolDescriptors.CalcExactMolWt(mol))
    counts = dict.fromkeys(ELEMENTS, 0.0)
    for atom in mol.GetAtoms():
        if atom.GetSymbol() in counts:
            counts[atom.GetSymbol()] += 1
        if atom.GetSymbol() != "H":
            counts["H"] += atom.GetTotalNumHs()
    features = [exact_mass / 1000, Descriptors.MolWt(mol) / 1000]
    features.extend(counts[e] / 100 for e in ELEMENTS)
    graph = Data(x=torch.tensor(nodes, dtype=torch.float32), edge_index=edge_index, edge_attr=edge_attr)
    graph.graph_features = torch.tensor([features], dtype=torch.float32)
    graph.molecular_ion_mz = torch.tensor([max(1, int(round(exact_mass)))], dtype=torch.long)
    graph.sample_id = torch.tensor([sample_id], dtype=torch.long)
    if peaks is not None:
        graph.y = torch.zeros(1, 1000)
        for mz, intensity in peaks:
            graph.y[0, int(mz) - 1] = float(intensity)
    return graph


def prepare_database(output, data_dir, msp, mol_dir, max_samples, seed, val_fraction, test_fraction):
    output.mkdir(parents=True, exist_ok=True)
    root, sources, mol_root = discover_inputs(data_dir, msp, mol_dir)
    manifest = {
        "schema": SCHEMA_VERSION, "data_dir": str(root), "mol_dir": str(mol_root),
        "sources": [{"path": str(p), "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in sources],
        "max_samples": max_samples, "split_seed": seed, "val_fraction": val_fraction,
        "test_fraction": test_fraction, "mol_root_mtime_ns": mol_root.stat().st_mtime_ns,
    }
    fingerprint = digest(json.dumps(manifest, sort_keys=True))
    path = output / "dataset.sqlite"
    if path.exists():
        with closing(connect(path)) as db:
            try:
                meta = dict(db.execute("SELECT key, value FROM meta"))
            except sqlite3.Error as error:
                raise ValueError("Incomplete dataset index. Use a NEW --output directory.") from error
            if meta.get("complete") != "1" or meta.get("fingerprint") != fingerprint:
                raise ValueError("Dataset/configuration changed or preparation incomplete. Use a NEW --output directory.")
            return path, json.loads(meta["summary"])
    db = connect(path)
    db.execute("CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT)")
    db.execute("INSERT INTO meta VALUES ('fingerprint', ?)", (fingerprint,))
    db.execute("CREATE TABLE samples(sample_id INTEGER PRIMARY KEY, source TEXT, ordinal INTEGER, identifier TEXT, name TEXT, smiles TEXT, peaks TEXT, priority TEXT, selected INTEGER DEFAULT 0, split TEXT)")
    mol_count = index_mols(db, mol_root)
    accepted, rejected = 0, 0
    with (output / "rejected_entries.csv").open("w", newline="", encoding="utf-8") as reject_file:
        writer = csv.writer(reject_file)
        writer.writerow(["source", "ordinal", "id", "name", "reason"])
        for source in sources:
            print(f"Streaming MSP: {source}", flush=True)
            for entry in tqdm(iter_msp(source), desc=source.parent.name, unit="entries"):
                try:
                    if not entry["peak_count_ok"]:
                        raise ValueError("missing or inconsistent Num Peaks")
                    if not entry["peaks"]:
                        raise ValueError("no positive peaks in m/z 1..1000")
                    mol = Chem.MolFromSmiles(entry.get("smiles", "")) if entry.get("smiles") else None
                    mol = mol if mol is not None else resolve_mol(db, source, entry.get("id"))
                    valid, reason = FEATURIZER.validate_molecule(mol)
                    if not valid:
                        raise ValueError(reason)
                    smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
                    # Reparse exactly as training does, including explicit H handling.
                    canonical_mol = Chem.MolFromSmiles(smiles)
                    valid, reason = FEATURIZER.validate_molecule(canonical_mol)
                    if not valid:
                        raise ValueError(reason)
                    maximum = max(entry["peaks"].values())
                    peaks = [(mz, val / maximum) for mz, val in sorted(entry["peaks"].items())]
                    source_key = str(source.relative_to(root)) if source.is_relative_to(root) else source.name
                    priority = digest(f"{seed}|sample|{source_key}|{entry['ordinal']}")
                    db.execute("INSERT INTO samples(source, ordinal, identifier, name, smiles, peaks, priority) VALUES (?, ?, ?, ?, ?, ?, ?)",
                               (source_key, entry["ordinal"], entry.get("id", ""), entry["name"], smiles, json.dumps(peaks), priority))
                    accepted += 1
                except (ValueError, RuntimeError) as error:
                    rejected += 1
                    writer.writerow([str(source), entry["ordinal"], entry.get("id", ""), entry["name"], str(error)])
                if (accepted + rejected) % 2000 == 0:
                    db.commit()
    if accepted == 0:
        db.commit()
        db.close()
        raise ValueError("Zero valid MSP/MOL pairs. Inspect rejected_entries.csv; check ID*.MOL filenames and --mol-dir.")
    db.execute("CREATE INDEX sample_priority ON samples(priority)")
    limit = max_samples if max_samples > 0 else accepted
    db.execute("UPDATE samples SET selected=1 WHERE sample_id IN (SELECT sample_id FROM samples ORDER BY priority LIMIT ?)", (limit,))
    db.create_function("group_split", 1, lambda s: split_for_smiles(s, seed, val_fraction, test_fraction))
    db.execute("UPDATE samples SET split=group_split(smiles) WHERE selected=1")
    db.execute("CREATE INDEX sample_split ON samples(split)")
    counts = dict(db.execute("SELECT split, COUNT(*) FROM samples WHERE selected=1 GROUP BY split"))
    if any(counts.get(s, 0) == 0 for s in ["train", "validation", "test"]):
        db.commit()
        db.close()
        raise ValueError(f"Empty split {counts}. Increase --max-samples; use a NEW output after changing preparation options.")
    summary = {**manifest, "fingerprint": fingerprint, "indexed_mols": mol_count, "valid_entries": accepted,
               "rejected_entries": rejected, "selected_entries": sum(counts.values()), "split_counts": counts,
               "canonical_groups": db.execute("SELECT COUNT(DISTINCT smiles) FROM samples WHERE selected=1").fetchone()[0],
               "split_policy": "canonical isomeric SMILES groups; deterministic SHA256; no exact-structure leakage",
               "input_reuse_limit": "MSP size/mtime checked; MOL content changes require a new output directory"}
    with (output / "selected_samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "source", "ordinal", "id", "name", "smiles", "split"])
        writer.writerows(db.execute("SELECT sample_id, source, ordinal, identifier, name, smiles, split FROM samples WHERE selected=1 ORDER BY sample_id"))
    db.execute("INSERT INTO meta VALUES ('summary', ?)", (json.dumps(summary),))
    db.execute("INSERT INTO meta VALUES ('complete', '1')")
    db.commit()
    db.close()
    (output / "dataset_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path, summary


def split_for_smiles(smiles, seed, val_fraction, test_fraction):
    uniform = int(digest(f"{seed}|group|{smiles}")[:16], 16) / 2**64
    if uniform < test_fraction:
        return "test"
    return "validation" if uniform < test_fraction + val_fraction else "train"


class DiskDataset(Dataset):
    def __init__(self, path, split):
        self.path = path
        self.db = connect(path)
        self.ids = np.array([r[0] for r in self.db.execute("SELECT sample_id FROM samples WHERE split=? ORDER BY sample_id", (split,))], dtype=np.int64)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        sample_id = int(self.ids[index])
        smiles, peaks = self.db.execute("SELECT smiles, peaks FROM samples WHERE sample_id=?", (sample_id,)).fetchone()
        return make_graph(smiles, json.loads(peaks), sample_id)

    def close(self):
        self.db.close()
