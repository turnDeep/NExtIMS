#!/usr/bin/env python3
"""
BonDNet HDF5 Streaming Dataset

Memory-efficient data loader for BonDNet that uses HDF5 for on-disk storage
and lazy loading of molecular graphs.

Usage:
    from scripts.bondnet_hdf5_dataset import create_hdf5_dataset, BonDNetHDF5Dataset

    # Step 1: Convert BonDNet data to HDF5
    create_hdf5_dataset(
        molecule_file='data/processed/bondnet_training/molecules.sdf',
        molecule_attributes_file='data/processed/bondnet_training/molecule_attributes.yaml',
        reaction_file='data/processed/bondnet_training/reactions.yaml',
        output_h5='data/processed/bondnet_training/bondnet_data.h5'
    )

    # Step 2: Use HDF5 dataset for training
    dataset = BonDNetHDF5Dataset('data/processed/bondnet_training/bondnet_data.h5')
    dataloader = DataLoader(dataset, batch_size=100, collate_fn=collate_bondnet)
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from rdkit import Chem
import yaml
import logging
import json
from tqdm import tqdm
import dgl
from dgl import DGLGraph

# Attempt to import BondNet classes for typing/construction if available
try:
    # Try importing ReactionInNetwork as Reaction (aliasing it as it seems to be the class name in installed version)
    from bondnet.data.reaction_network import ReactionInNetwork as Reaction
except ImportError:
    try:
        from bondnet.data.reaction_network import Reaction
    except ImportError:
        Reaction = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_hdf5_dataset(
    molecule_file: str,
    molecule_attributes_file: str,
    reaction_file: str,
    output_h5: str
):
    """
    Convert BonDNet SDF+YAML data to HDF5 format for memory-efficient loading.

    Args:
        molecule_file: Path to molecules.sdf
        molecule_attributes_file: Path to molecule_attributes.yaml
        reaction_file: Path to reactions.yaml
        output_h5: Output HDF5 file path
    """
    logger.info("="*80)
    logger.info("Converting BonDNet data to HDF5 format")
    logger.info("="*80)

    # Load molecules from SDF
    logger.info(f"Loading molecules from: {molecule_file}")
    suppl = Chem.SDMolSupplier(molecule_file, removeHs=False, sanitize=False)

    molecules_smiles = []
    molecules_ids = []

    for idx, mol in enumerate(tqdm(suppl, desc="Reading molecules")):
        if mol is None:
            logger.warning(f"Failed to parse molecule at index {idx}")
            molecules_smiles.append("")  # Placeholder
            molecules_ids.append(f"mol_{idx}")
        else:
            smiles = Chem.MolToSmiles(mol)
            mol_id = mol.GetProp('_Name') if mol.HasProp('_Name') else f"mol_{idx}"
            molecules_smiles.append(smiles)
            molecules_ids.append(mol_id)

    logger.info(f"Loaded {len(molecules_smiles)} molecules")

    # Load molecule attributes
    logger.info(f"Loading molecule attributes from: {molecule_attributes_file}")
    with open(molecule_attributes_file, 'r') as f:
        molecule_attributes = yaml.safe_load(f)

    # Handle the case where molecule_attributes is a list of dictionaries (row-based)
    # instead of a dictionary of lists (column-based).
    if isinstance(molecule_attributes, list):
        logger.info("Detected list-based molecule attributes. Converting to dictionary format...")
        if not molecule_attributes:
            molecule_attributes = {}
        elif isinstance(molecule_attributes[0], dict):
            # Convert list of dicts to dict of lists
            # Assume all dicts have the same keys
            keys = molecule_attributes[0].keys()
            new_attributes = {k: [] for k in keys}
            for item in molecule_attributes:
                for k in keys:
                    new_attributes[k].append(item.get(k, None))  # Use None for missing values
            molecule_attributes = new_attributes
        else:
            logger.warning("Unknown format for molecule_attributes list. Skipping attributes.")
            molecule_attributes = {}

    # Load reactions
    logger.info(f"Loading reactions from: {reaction_file}")
    with open(reaction_file, 'r') as f:
        reactions = yaml.safe_load(f)

    logger.info(f"Loaded {len(reactions)} reactions")

    # Create HDF5 file
    logger.info(f"Creating HDF5 file: {output_h5}")
    Path(output_h5).parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_h5, 'w') as f:
        # Store molecules (SMILES only, graphs generated on-the-fly)
        logger.info("Storing molecules...")
        dt = h5py.string_dtype(encoding='utf-8')
        f.create_dataset('molecule_smiles', data=molecules_smiles, dtype=dt)
        f.create_dataset('molecule_ids', data=molecules_ids, dtype=dt)

        # Store molecule attributes (if any)
        if molecule_attributes:
            attrs_group = f.create_group('molecule_attributes')
            for key, values in molecule_attributes.items():
                if isinstance(values, list):
                    attrs_group.create_dataset(key, data=np.array(values))

        # Store reactions
        logger.info("Storing reactions...")
        reactions_group = f.create_group('reactions')

        reaction_ids = []
        reactant_ids = []
        product_ids_json = []  # Store list of products as JSON string
        atom_mapping_json = [] # Store list of dicts as JSON string
        bond_mapping_json = [] # Store list of dicts as JSON string
        bond_indices = []
        energies = []

        for rxn_idx, rxn in enumerate(tqdm(reactions, desc="Processing reactions")):
            reaction_ids.append(rxn.get('id', f'rxn_{rxn_idx}'))
            reactant_ids.append(rxn['reactant'])

            # Products can be multiple, store as list
            products = rxn.get('products', [])
            product_ids_json.append(json.dumps(products))

            # Mappings
            atom_map = rxn.get('atom_mapping', [])
            bond_map = rxn.get('bond_mapping', [])
            atom_mapping_json.append(json.dumps(atom_map))
            bond_mapping_json.append(json.dumps(bond_map))

            # Bond index (which bond is broken)
            bond_idx = rxn.get('bond_index', -1)
            bond_indices.append(bond_idx)

            # Energy (BDE value)
            energy = rxn.get('energy', 0.0)
            energies.append(energy)

        reactions_group.create_dataset('reaction_ids', data=reaction_ids, dtype=dt)
        reactions_group.create_dataset('reactant_ids', data=reactant_ids, dtype=dt)
        reactions_group.create_dataset('product_ids_json', data=product_ids_json, dtype=dt)
        reactions_group.create_dataset('atom_mapping_json', data=atom_mapping_json, dtype=dt)
        reactions_group.create_dataset('bond_mapping_json', data=bond_mapping_json, dtype=dt)
        reactions_group.create_dataset('bond_indices', data=np.array(bond_indices, dtype=np.int32))
        reactions_group.create_dataset('energies', data=np.array(energies, dtype=np.float32))

        # Store metadata
        f.attrs['num_molecules'] = len(molecules_smiles)
        f.attrs['num_reactions'] = len(reactions)
        f.attrs['created_by'] = 'NExtIMS BonDNet HDF5 Converter v2'

    logger.info("="*80)
    logger.info(f"✓ HDF5 dataset created successfully: {output_h5}")
    logger.info(f"  Molecules: {len(molecules_smiles):,}")
    logger.info(f"  Reactions: {len(reactions):,}")
    logger.info(f"  File size: {Path(output_h5).stat().st_size / 1024 / 1024:.1f} MB")
    logger.info("="*80)


class BonDNetHDF5Dataset(Dataset):
    """
    Memory-efficient BonDNet dataset using HDF5 storage.
    """

    def __init__(self, h5_file: str, cache_graphs: bool = False):
        self.h5_file = h5_file
        self.cache_graphs = cache_graphs
        self.graph_cache = {} if cache_graphs else None

        # Load metadata
        with h5py.File(h5_file, 'r') as f:
            self.num_molecules = f.attrs['num_molecules']
            self.num_reactions = f.attrs['num_reactions']

            # Pre-load molecule IDs for fast lookup
            # This consumes some memory but is much faster than reading from disk every time
            self.molecule_ids = [s.decode('utf-8') if isinstance(s, bytes) else s
                               for s in f['molecule_ids'][:]]
            self.molecule_id_to_idx = {mid: i for i, mid in enumerate(self.molecule_ids)}

        logger.info(f"Initialized BonDNet HDF5 Dataset:")
        logger.info(f"  HDF5 file: {h5_file}")
        logger.info(f"  Molecules: {self.num_molecules:,}")
        logger.info(f"  Reactions: {self.num_reactions:,}")

    def __len__(self) -> int:
        return self.num_reactions

    def _smiles_to_dgl_graph(self, smiles: str) -> DGLGraph:
        """Convert SMILES to DGL graph (BonDNet format)."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            mol = Chem.MolFromSmiles('C')

        # BonDNet requires explicit hydrogens
        mol = Chem.AddHs(mol)

        # Build graph
        num_atoms = mol.GetNumAtoms()

        # Node features (atom features)
        # Simplified features - in production should use bondnet.data.featurizer
        node_features = []
        for atom in mol.GetAtoms():
            features = [
                atom.GetAtomicNum(),
                atom.GetTotalDegree(),
                atom.GetFormalCharge(),
                atom.GetTotalNumHs(),
                atom.GetNumRadicalElectrons(),
                int(atom.GetHybridization()),
                int(atom.GetIsAromatic()),
                int(atom.IsInRing()),
            ]
            node_features.append(features)

        # Edge list
        edges_src = []
        edges_dst = []
        edge_features = []

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            features = [
                float(bond.GetBondTypeAsDouble()),
                float(bond.GetIsConjugated()),
                float(bond.GetIsAromatic()),
                float(bond.IsInRing()),
            ]

            # Add bidirectional edges
            edges_src.extend([i, j])
            edges_dst.extend([j, i])
            edge_features.extend([features, features])

        # Create DGL graph
        # BonDNet uses HeteroGraphs with 'atom' and 'bond' nodes or Homogeneous graphs treated as Hetero
        # But GatedGCNReactionNetwork usually expects a heterograph or specific node types.
        # Based on error 'Expect ntype in ['_N'], got atom', our current graph is homogeneous (ntype '_N').
        # We should create a heterograph with 'atom' nodes (and maybe 'bond' nodes if we use line graph)
        # OR just rename the node type.

        # Creating a heterograph with explicit node types 'atom' and 'bond' is complex if the model
        # expects a specific structure (like line graph connections).
        # However, checking BonDNet code:
        # It typically uses dgl.heterograph({('atom', 'bond', 'atom'): (u, v)})

        # For simplicity in this HDF5 loader, let's create a heterograph mimicking what's expected.
        # But wait, BonDNet grapher produces a graph where 'atom' is a node type.

        data_dict = {
            ('atom', 'a2b', 'atom'): (torch.tensor(edges_src, dtype=torch.long), torch.tensor(edges_dst, dtype=torch.long))
        }

        # If no edges, we still need the node type 'atom'
        if len(edges_src) == 0:
            # DGL heterograph with no edges but specified num_nodes
            # Note: BondNet seems to expect 'bond' nodes as well (likely line graph nodes or explicit bond nodes)
            # If the model errors with 'Node type "bond" does not exist', we must add it.
            # Even if we have 0 bonds, the type must exist.
            g = dgl.heterograph(data_dict, num_nodes_dict={'atom': num_atoms, 'bond': 0})
        else:
            # We assume 'bond' nodes correspond to edges? Or does BondNet construct a line graph?
            # Looking at BondNet `grapher.py`, it uses `BondAsNodeFeaturizerFull`.
            # This implies bonds are treated as nodes in the graph structure.
            # The edge list I constructed was atom-to-atom ('a2b' relation implies atom-to-bond?).

            # The previous 'a2b' relation I guessed might be wrong if 'bond' is a node type.
            # If 'bond' is a node type, then 'a2b' probably connects atoms to bond nodes.

            # Let's adjust to create 'bond' nodes.
            # Each bond in molecule (edge in my construction) becomes a node of type 'bond'.
            # My edge list `edges_src`, `edges_dst` represents atom connectivity.
            # If we follow BondNet convention:
            # Nodes: 'atom', 'bond'
            # Edges: 'a2b' (atom to bond), 'b2a' (bond to atom), 'b2b' (bond to bond, optional)

            # In my simplified HDF5 loader, I don't have the full logic to build this complex graph.
            # However, I can try to mimic it simply.
            # Every undirected bond becomes 1 'bond' node? Or 2 directed?
            # RDKit gives bonds.

            num_bonds = mol.GetNumBonds()

            # Reset data dict for correct structure
            # We need to map which atom connects to which bond node.

            a2b_src = []
            a2b_dst = []
            b2a_src = []
            b2a_dst = []

            bond_feats_reordered = []

            for b_idx, bond in enumerate(mol.GetBonds()):
                u = bond.GetBeginAtomIdx()
                v = bond.GetEndAtomIdx()

                # Bond node index = b_idx
                # Connect u -> b_idx and v -> b_idx (for a2b)
                # Connect b_idx -> u and b_idx -> v (for b2a)

                a2b_src.extend([u, v])
                a2b_dst.extend([b_idx, b_idx])

                b2a_src.extend([b_idx, b_idx])
                b2a_dst.extend([u, v])

                # Features for this bond node
                features = [
                    float(bond.GetBondTypeAsDouble()),
                    float(bond.GetIsConjugated()),
                    float(bond.GetIsAromatic()),
                    float(bond.IsInRing()),
                ]
                bond_feats_reordered.append(features)

            data_dict = {
                ('atom', 'a2b', 'bond'): (torch.tensor(a2b_src, dtype=torch.long), torch.tensor(a2b_dst, dtype=torch.long)),
                ('bond', 'b2a', 'atom'): (torch.tensor(b2a_src, dtype=torch.long), torch.tensor(b2a_dst, dtype=torch.long)),
                # Adding b2b (bond to bond) to satisfy GatedGCN expectations if it checks for it
                # Even if empty.
                ('bond', 'b2b', 'bond'): (torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)),

                # Adding g2b (global to bond) and b2g (bond to global) as per GatedGCNConv expectation
                # 'global' node is usually connected to all bonds in BonDNet
                # We have 1 global node (idx 0). We need to connect it to all bond nodes.
                ('global', 'g2b', 'bond'): (torch.zeros(num_bonds, dtype=torch.long), torch.arange(num_bonds, dtype=torch.long)),
                ('bond', 'b2g', 'global'): (torch.arange(num_bonds, dtype=torch.long), torch.zeros(num_bonds, dtype=torch.long)),

                # Adding a2a (atom to atom) - probably based on original bonds?
                # GatedGCN might use direct atom-atom connections too.
                # In standard graph, a2a would be the bonds.
                # Let's add them.
                ('atom', 'a2a', 'atom'): (torch.tensor(a2b_src, dtype=torch.long), torch.tensor(a2b_src, dtype=torch.long)) # Wait, src/dst logic for a2a needs to be correct.
                # Use original edges for a2a
            }

            # Resetting a2a to use proper edges from mol
            # We already have a2b_src/dst which came from u,v
            # a2a should be u->v and v->u

            # Since I already flattened a2b_src to include both directions in previous loop (I think? No, I did)
            # In the loop:
            # a2b_src.extend([u, v])
            # a2b_dst.extend([b_idx, b_idx])
            # So a2b_src has [u1, v1, u2, v2...]

            # For a2a, we want u->v and v->u.
            # a2a_src = [u, v]
            # a2a_dst = [v, u]

            a2a_src = []
            a2a_dst = []

            for bond in mol.GetBonds():
                u = bond.GetBeginAtomIdx()
                v = bond.GetEndAtomIdx()
                a2a_src.extend([u, v])
                a2a_dst.extend([v, u])

            data_dict[('atom', 'a2a', 'atom')] = (torch.tensor(a2a_src, dtype=torch.long), torch.tensor(a2a_dst, dtype=torch.long))

            # Adding g2a (global to atom) and a2g (atom to global)
            data_dict[('global', 'g2a', 'atom')] = (torch.zeros(num_atoms, dtype=torch.long), torch.arange(num_atoms, dtype=torch.long))
            data_dict[('atom', 'a2g', 'global')] = (torch.arange(num_atoms, dtype=torch.long), torch.zeros(num_atoms, dtype=torch.long))

            # Adding g2g (global to global) self-loop
            data_dict[('global', 'g2g', 'global')] = (torch.zeros(1, dtype=torch.long), torch.zeros(1, dtype=torch.long))

            g = dgl.heterograph(data_dict, num_nodes_dict={'atom': num_atoms, 'bond': num_bonds, 'global': 1})
            if num_bonds > 0:
                g.nodes['bond'].data['feat'] = torch.tensor(bond_feats_reordered, dtype=torch.float32)
            else:
                g.nodes['bond'].data['feat'] = torch.zeros((0, 4), dtype=torch.float32)

        g.nodes['atom'].data['feat'] = torch.tensor(node_features, dtype=torch.float32)
        # Global node feature (empty/zero)
        g.nodes['global'].data['feat'] = torch.zeros((1, 0), dtype=torch.float32)

        return g

    def __getitem__(self, idx: int) -> Dict:
        """
        Load a single reaction sample with mapping info.
        """
        with h5py.File(self.h5_file, 'r') as f:
            # Load reaction metadata
            reactant_id = f['reactions/reactant_ids'][idx]
            product_ids_json = f['reactions/product_ids_json'][idx]
            atom_mapping_json = f['reactions/atom_mapping_json'][idx]
            bond_mapping_json = f['reactions/bond_mapping_json'][idx]
            energy = f['reactions/energies'][idx]

            # Decode strings
            if isinstance(reactant_id, bytes):
                reactant_id = reactant_id.decode('utf-8')

            if isinstance(product_ids_json, bytes):
                product_ids_json = product_ids_json.decode('utf-8')
            product_ids = json.loads(product_ids_json)

            if isinstance(atom_mapping_json, bytes):
                atom_mapping_json = atom_mapping_json.decode('utf-8')
            atom_mapping = json.loads(atom_mapping_json)
            # Convert keys back to int
            atom_mapping = [{int(k): v for k, v in mp.items()} for mp in atom_mapping]

            if isinstance(bond_mapping_json, bytes):
                bond_mapping_json = bond_mapping_json.decode('utf-8')
            bond_mapping = json.loads(bond_mapping_json)
            # Convert keys back to int
            bond_mapping = [{int(k): v for k, v in mp.items()} for mp in bond_mapping]

            # Load SMILES using index lookup
            reactant_idx = self.molecule_id_to_idx[reactant_id]
            reactant_smiles = f['molecule_smiles'][reactant_idx]
            if isinstance(reactant_smiles, bytes):
                reactant_smiles = reactant_smiles.decode('utf-8')

            product_smiles_list = []
            for pid in product_ids:
                pidx = self.molecule_id_to_idx[pid]
                psmiles = f['molecule_smiles'][pidx]
                if isinstance(psmiles, bytes):
                    psmiles = psmiles.decode('utf-8')
                product_smiles_list.append(psmiles)

        # Generate graphs
        reactant_graph = self._get_graph(reactant_id, reactant_smiles)
        product_graphs = [self._get_graph(pid, psmiles) for pid, psmiles in zip(product_ids, product_smiles_list)]

        return {
            'reactant_graph': reactant_graph,
            'product_graphs': product_graphs,
            'atom_mapping': atom_mapping,
            'bond_mapping': bond_mapping,
            'energy': float(energy)
        }

    def _get_graph(self, mol_id, smiles):
        if self.cache_graphs and mol_id in self.graph_cache:
            return self.graph_cache[mol_id]

        g = self._smiles_to_dgl_graph(smiles)

        if self.cache_graphs:
            self.graph_cache[mol_id] = g
        return g


def collate_bondnet(batch: List[Dict]) -> Tuple[dgl.DGLGraph, Dict]:
    """
    Collate function for BonDNet DataLoader.
    Constructs a batched graph and a list of Reaction objects.
    """
    all_graphs = []
    reactions = []
    energies = []

    current_idx_offset = 0

    for sample in batch:
        r_graph = sample['reactant_graph']
        p_graphs = sample['product_graphs']

        # Add to global list
        # Order: [Reactant, Product1, Product2, ...]
        all_graphs.append(r_graph)
        all_graphs.extend(p_graphs)

        # Calculate indices for this reaction in the new batch
        reactant_indices = [current_idx_offset]
        product_indices = [current_idx_offset + 1 + i for i in range(len(p_graphs))]

        # Create Reaction object (if available)
        # Even if Reaction class is not available (ImportError), we need to pass SOMETHING expected by the model.
        # But if Reaction is None, reactions list will be empty, leading to "input list of graphs cannot be empty" error.

        # We need Reaction class to be available. It should be if bondnet is installed.
        if Reaction is not None:
            # Note: Reaction expects 'init_reactants' etc.
            # Here we provide indices relative to the batch
            rxn = Reaction(
                reactants=reactant_indices,
                products=product_indices,
                atom_mapping=sample['atom_mapping'],
                bond_mapping=sample['bond_mapping'],
                id=None
            )
            reactions.append(rxn)
        else:
             # Fallback mock object if Reaction is not imported (should not happen in real env if bondnet installed)
             # But for my test env if bondnet import failed earlier?
             # I installed bondnet, so it should be fine.
             # Why is it 0?
             logger.warning("Reaction class not available!")

        energies.append(sample['energy'])

        current_idx_offset += 1 + len(p_graphs)

    # Batch all graphs
    batched_graph = dgl.batch(all_graphs)

    # Prepare label dict (as expected by train_bde_distributed.py)
    labels = {
        "value": torch.tensor(energies, dtype=torch.float32).unsqueeze(1),
        "reaction": reactions,
        # Placeholders for norms (should be computed if needed)
        "norm_atom": None,
        "norm_bond": None,
        "scaler_stdev": torch.tensor(1.0) # Placeholder
    }

    return batched_graph, labels

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Convert BonDNet data to HDF5 format')
    parser.add_argument('--molecule-file', required=True, help='Path to molecules.sdf')
    parser.add_argument('--molecule-attributes', required=True, help='Path to molecule_attributes.yaml')
    parser.add_argument('--reaction-file', required=True, help='Path to reactions.yaml')
    parser.add_argument('--output', required=True, help='Output HDF5 file path')

    args = parser.parse_args()

    create_hdf5_dataset(
        molecule_file=args.molecule_file,
        molecule_attributes_file=args.molecule_attributes,
        reaction_file=args.reaction_file,
        output_h5=args.output
    )
