"""Target-free primary ion candidates for the EXP-V014 charge branch.

Only primary events are used by charge competition. Omit the unused secondary
and rearrangement states from the screen to bound batch memory at full scale.
"""

from collections import Counter
from itertools import combinations, islice

import torch
from rdkit import Chem
from torch_geometric.data import Data

ELEMENTS = ("C", "H", "N", "O", "S", "F", "Cl", "Br", "I", "Si", "P")
PT = Chem.GetPeriodicTable()
MAX_EVENTS = 33


class IonData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == "frag_members":
            return torch.tensor([[self.frag_mz.numel()], [self.num_nodes]])
        if key == "event_index":
            return self.frag_mz.numel()
        if key == "frag_graph":
            return 1
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in {"frag_members", "event_index"}:
            return 1
        return super().__cat_dim__(key, value, *args, **kwargs)


class PrimaryCandidates:
    def __init__(self, smiles):
        self.mol = Chem.MolFromSmiles(smiles)
        if self.mol is None:
            raise ValueError("Invalid candidate SMILES")
        self.atoms, self.bonds = list(self.mol.GetAtoms()), list(self.mol.GetBonds())
        self.all_atoms = frozenset(range(len(self.atoms)))
        self.adj = {i: [] for i in self.all_atoms}
        for bond in self.bonds:
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            self.adj[a].append((b, bond.GetIdx()))
            self.adj[b].append((a, bond.GetIdx()))
        self.states, self.index, self.events = [], {}, []
        self.parent_formula = self.formula(self.all_atoms)
        self.parent_mass = self.mass(self.parent_formula)
        self.eligible = (len(Chem.GetMolFrags(self.mol)) == 1
                         and Chem.GetFormalCharge(self.mol) == 0
                         and 1 <= round(self.parent_mass - 0.00054858) <= 1000)

    def formula(self, atoms):
        result = Counter()
        for i in atoms:
            atom = self.atoms[i]
            result[atom.GetSymbol()] += 1
            if atom.GetSymbol() != "H":
                result["H"] += int(atom.GetTotalNumHs())
        return result

    @staticmethod
    def mass(formula):
        return sum(PT.GetMostCommonIsotopeMass(PT.GetAtomicNumber(e)) * n
                   for e, n in formula.items())

    def components(self, removed):
        remaining, groups = set(self.all_atoms), []
        while remaining:
            stack, group = [min(remaining)], set()
            while stack:
                node = stack.pop()
                if node in group:
                    continue
                group.add(node)
                stack.extend(n for n, b in self.adj[node]
                             if b not in removed and n in remaining and n not in group)
            remaining -= group
            groups.append(frozenset(group))
        return sorted(groups, key=lambda s: tuple(sorted(s)))

    def add_state(self, atoms, fallback=False):
        key = tuple(sorted(atoms))
        if key in self.index:
            return self.index[key]
        formula = self.formula(atoms)
        mass = self.mass(formula)
        mz = int(round(mass - 0.00054858))
        if not 1 <= mz <= 1000:
            if not fallback:
                return None
            # The charge correction is disabled for this graph; retain an
            # inert root so PyG batching and chemical normalization stay valid.
            mz = min(1000, max(1, mz))
        nuclear = sum(PT.GetAtomicNumber(e) * n for e, n in formula.items())
        odd = int((nuclear - 1) % 2)
        internal = [b for b in self.bonds
                    if b.GetBeginAtomIdx() in atoms and b.GetEndAtomIdx() in atoms]
        boundary = sum((b.GetBeginAtomIdx() in atoms) != (b.GetEndAtomIdx() in atoms)
                       for b in self.bonds)
        basic = [mass / 1000, mass / max(self.parent_mass, 1), len(atoms) / len(self.atoms), 0.]
        basic += [formula[e] / 100 for e in ELEMENTS]
        basic += [boundary / 10]
        charged = [odd, sum(self.atoms[i].GetIsAromatic() for i in atoms) / len(atoms)]
        charged += [formula[e] / max(len(atoms), 1) for e in ("N", "O", "S")]
        charged += [sum(b.GetIsConjugated() for b in internal) / max(len(internal), 1),
                    sum(b.GetBondTypeAsDouble() > 1 for b in internal) / max(len(internal), 1),
                    sum(self.atoms[i].GetFormalCharge() for i in atoms) / 4]
        index = len(self.states)
        self.index[key] = index
        self.states.append(dict(atoms=atoms, mz=mz, basic=basic, charged=charged))
        return index

    def enumerate(self):
        root = self.add_state(self.all_atoms, fallback=True)
        self.events.append((root, root, False))
        if not self.eligible:
            return self
        cuts = [(b.GetIdx(),) for b in self.bonds if not b.IsInRing()]
        ring = [b.GetIdx() for b in self.bonds if b.IsInRing()]
        cuts.extend(islice(combinations(ring, 2), 64))
        for cut in cuts:
            if len(self.events) >= MAX_EVENTS:
                break
            groups = self.components(set(cut))
            if len(groups) != 2:
                continue
            first, second = self.add_state(groups[0]), self.add_state(groups[1])
            if first is not None and second is not None:
                self.events.append((first, second, True))
        return self

    def attach(self, graph):
        if graph.num_nodes != len(self.atoms):
            raise ValueError("Candidate and graph atom order/count must match")
        result = IonData(**graph.to_dict())
        members = [(f, a) for f, state in enumerate(self.states) for a in sorted(state["atoms"])]
        result.frag_members = torch.tensor(members, dtype=torch.long).t().contiguous()
        result.frag_basic = torch.tensor([s["basic"] for s in self.states], dtype=torch.float32)
        result.frag_charged = torch.tensor([s["charged"] for s in self.states], dtype=torch.float32)
        result.frag_mz = torch.tensor([s["mz"] for s in self.states], dtype=torch.long)
        result.frag_graph = torch.zeros(len(self.states), dtype=torch.long)
        result.event_index = torch.tensor([e[:2] for e in self.events], dtype=torch.long).t().contiguous()
        result.event_paired = torch.tensor([e[2] for e in self.events], dtype=torch.bool)
        result.ion_eligible = torch.tensor([int(self.eligible)])
        result.candidate_counts = torch.tensor([[len(self.states), len(self.events)]])
        return result


def attach(graph, smiles):
    return PrimaryCandidates(smiles).enumerate().attach(graph)
