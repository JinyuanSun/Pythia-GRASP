

import os
import glob
import gzip
import argparse
import warnings
from tqdm import tqdm
from joblib import Parallel, delayed
from Bio import BiopythonDeprecationWarning
from Bio.PDB.Polypeptide import index_to_one

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from model import AMPNN
from pdb_utils import read_pdb_to_protbb, get_neighbor

warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)
pythia_root_dir = os.path.dirname(os.path.abspath(__file__))


def get_torch_model(checkpoint_path, device='cuda'):
    model = AMPNN(
        embed_dim=128,
        edge_dim=27,
        node_dim=28,
        dropout=0.0,
        layer_nums=3,
        token_num=21,
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device(device)))
    model.eval()
    model.to(device)
    return model


def calculate_plddt(pdb_file):
    b_factors = []
    open_func = gzip.open if pdb_file.endswith(".pdb.gz") else open
    with open_func(pdb_file, "rt" if pdb_file.endswith(".gz") else "r") as f:
        for line in f:
            if line.startswith("ATOM"):
                b_factor = float(line[60:66])
                b_factors.append(b_factor)
    return np.mean(b_factors)


def apply_ddfep_opt_fixed(pythia_predictions):
    aa_weights = {
        'A': -0.374, 'C': 0.279, 'D': -0.588, 'E': -0.250, 'F': 0.450,
        'G': -0.730, 'H': -0.063, 'I': 0.549, 'K': -0.416, 'L': 0.333,
        'M': 0.732, 'N': -0.427, 'P': -0.267, 'Q': -0.393, 'R': 0.306,
        'S': -0.545, 'T': -0.494, 'V': 0.089, 'W': 1.063, 'Y': 0.746
    }
    delta_weight = 0.145

    corrected_predictions = {}
    for mutation, pythia_score in pythia_predictions.items():
        from_aa = mutation[0]
        to_aa = mutation[-1]
        correction = (
            -1 * delta_weight * pythia_score
            + aa_weights[to_aa] * 1
            + aa_weights[from_aa] * -1
        )
        corrected_predictions[mutation] = -1 * correction

    return corrected_predictions


def make_one_scan(
    pdb_file,
    torch_models,
    device='cuda',
    output_dir=None,
    save_pt=False,
    save_csv=True,
    apply_correction=True
):
    protbb = read_pdb_to_protbb(pdb_file)
    node_features, edge_features, sequence = get_neighbor(protbb, noise_level=0.0)
    model_outputs = []

    with torch.no_grad():
        for model in torch_models:
            logits, _ = model(node_features.to(device), edge_features.to(device))
            probabilities = F.softmax(logits, dim=-1).detach().cpu().numpy()
            model_outputs.append(probabilities)

    mutation_scores = {}

    for position, aa in enumerate(protbb.seq):
        energy = np.zeros(21)
        from_aa_index = int(aa.item())

        for prob in model_outputs:
            energy += -np.log(prob[position] / prob[position][from_aa_index])

        from_aa = index_to_one(from_aa_index)
        for i in range(20):
            to_aa = index_to_one(i)
            mutation_name = f"{from_aa}{position+1}{to_aa}"
            mutation_scores[mutation_name] = energy[i]

    corrected_scores = apply_ddfep_opt_fixed(mutation_scores) if apply_correction else mutation_scores

    pdb_basename = os.path.basename(pdb_file)
    output_base = os.path.join(output_dir, os.path.splitext(pdb_basename)[0]) if output_dir else pdb_file.replace(".pdb", "")

    if save_pt:
        output_dict = {
            mutation: (np.float16(mutation_scores[mutation]), np.float16(corrected_scores[mutation]))
            for mutation in mutation_scores
        }
        output_path = output_base + "_pred_mask.pt"
        torch.save(output_dict, output_path)
        print(f"Saved prediction (PT): {output_path}")

    if save_csv:
        output_path = output_base + "_pred_mask.csv"
        df = pd.DataFrame({
            "Mutation": list(mutation_scores.keys()),
            "PythiaScore": list(mutation_scores.values()),
            "CorrectedScore": list(corrected_scores.values())
        })
        df.to_csv(output_path, index=False)
        print(f"Saved prediction (CSV): {output_path}")


def main(args):
    input_dir = args.input_dir
    pdb_filename = args.pdb_filename
    output_dir = args.output_dir
    check_plddt = args.check_plddt
    plddt_cutoff = args.plddt_cutoff
    n_jobs = args.n_jobs
    device = args.device
    save_pt = args.save_pt
    apply_correction = args.apply_correction

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    torch_model_c = get_torch_model(os.path.join(pythia_root_dir, "../pythia-c.pt"), device)
    torch_model_p = get_torch_model(os.path.join(pythia_root_dir, "../pythia-p.pt"), device)
    torch_models = [torch_model_c, torch_model_p]

    if input_dir:
        pdb_files = glob.glob(os.path.join(input_dir, '*.pdb'))
        print(f"Found {len(pdb_files)} PDB files in {input_dir}")

        if check_plddt:
            pdb_files = [
                pdb for pdb in tqdm(pdb_files, desc="Checking pLDDT")
                if calculate_plddt(pdb) > plddt_cutoff
            ]
            print(f"Found {len(pdb_files)} PDB files with pLDDT > {plddt_cutoff}")

        Parallel(n_jobs=n_jobs)(
            delayed(make_one_scan)(
                pdb_file, torch_models, device,
                output_dir=output_dir,
                save_pt=save_pt,
                save_csv=True,
                apply_correction=apply_correction
            )
            for pdb_file in tqdm(pdb_files, desc="Processing PDB files")
        )

    if pdb_filename:
        make_one_scan(
            pdb_filename, torch_models, device,
            output_dir=output_dir,
            save_pt=save_pt,
            save_csv=True,
            apply_correction=apply_correction
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pythia MBC Scoring Pipeline")
    parser.add_argument('--input_dir', type=str, default=None, help='Directory containing PDB files.')
    parser.add_argument('--pdb_filename', type=str, default=None, help='Single PDB file to process.')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for saving results.')
    parser.add_argument('--check_plddt', action='store_true', help='Check pLDDT value and filter.')
    parser.add_argument('--plddt_cutoff', type=float, default=95.0, help='Minimum pLDDT cutoff for filtering.')
    parser.add_argument('--n_jobs', type=int, default=2, help='Number of parallel jobs.')
    parser.add_argument('--device', type=str, default="cuda:0", help='Device for computation.')
    parser.add_argument('--save_pt', action='store_true', help='Save prediction results in .pt format.')
    parser.add_argument('--apply_correction', action='store_true', help='Apply MBC correction to pythiascore.')
    args = parser.parse_args()
    main(args)